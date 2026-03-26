from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .config import (
    ENV_TRAIN_DOUBLE_TARGET,
    ENV_TRAIN_MODEL_MODE,
    ENV_TRAIN_MSE_TERM,
    apply_train_overrides,
    load_config,
    resolve_train_override_cli_env,
)
from .datasets import FrameDataset, SegmentDataset
from .features import (
    meta_feature_names,
    pair_feature_names,
    pass1_feature_names,
    resolve_feature_profile,
    self_feature_names,
)
from .manifest import build_manifest
from .models import Phase1Net, Phase2Net, Phase2_1Net, Phase3Net
from .utils import (
    compute_psnr_from_mse_torch,
    ensure_dir,
    huber_loss_masked,
    inverse_log_bits,
    inverse_log_mse,
    regression_metrics,
    save_json,
    set_seed,
    to_device,
    train_val_test_split,
)


def setup_distributed() -> tuple[int, int, bool]:
    """返回 (local_rank, world_size, use_ddp)。非 torchrun 或单进程时 use_ddp=False。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, False
    if not torch.cuda.is_available():
        raise RuntimeError("检测到 WORLD_SIZE>1 但当前无 CUDA，无法进行 DDP 训练。")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    backend = "nccl"
    dist.init_process_group(backend=backend)
    return local_rank, dist.get_world_size(), True


def cleanup_distributed(use_ddp: bool) -> None:
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def mse_term_normalized(cfg: dict) -> str:
    return str(cfg["loss"].get("mse_term", "log_mse")).lower().strip()


def is_vmaf_distortion(cfg: dict) -> bool:
    return mse_term_normalized(cfg) == "vmaf"


def _distortion_loss_dir_suffix(cfg: dict) -> str:
    """由 loss.mse_term 区分输出目录：log_mse → _logmse，psnr → _psnr，vmaf → _vmaf。"""
    term = mse_term_normalized(cfg)
    if term == "psnr":
        return "_psnr"
    if term == "vmaf":
        return "_vmaf"
    return "_logmse"


def is_double_mode(cfg: dict) -> bool:
    return str(cfg.get("model", {}).get("mode", "single")).lower().strip() == "double"


def double_target(cfg: dict) -> str:
    """bits | distortion（仅 mode=double）"""
    return str(cfg.get("model", {}).get("double_target", "bits")).lower().strip()


def phase2_variant(cfg: dict) -> str:
    return str(cfg.get("model", {}).get("phase2_variant", "flat")).lower().strip()


def model_mode_tag(cfg: dict) -> str:
    """与 evaluate_loader 返回的 model_mode 一致。"""
    if not is_double_mode(cfg):
        return "single"
    return "double_bits" if double_target(cfg) == "bits" else "double_distortion"


def train_metrics_stub_when_skip_full_train_eval(cfg: dict, train_log: dict) -> dict:
    """eval_full_train_each_epoch=false 时占位，避免 history 缺字段。"""
    base = {
        "loss": float(train_log["loss"]),
        "by_frame_type": {},
        "by_temporal_layer": {},
        "model_mode": model_mode_tag(cfg),
        "_skipped_full_train_eval": True,
        "mse_term": mse_term_normalized(cfg),
    }
    if is_double_mode(cfg) and double_target(cfg) == "bits":
        return {**base, "bits": {}, "mse": {}, "psnr": {}}
    if is_double_mode(cfg) and double_target(cfg) == "distortion":
        if is_vmaf_distortion(cfg):
            return {**base, "vmaf": {}}
        return {**base, "bits": {}, "mse": {}, "psnr": {}}
    if is_vmaf_distortion(cfg):
        return {**base, "bits": {}, "vmaf": {}}
    return {**base, "bits": {}, "mse": {}, "psnr": {}}


def model_head_out_dim(cfg: dict) -> int:
    return 1 if is_double_mode(cfg) else 2


def _double_mode_dir_suffix(cfg: dict) -> str:
    """double 专用目录后缀：_double_bits | _double_psnr | _double_mse | _double_vmaf"""
    dt = double_target(cfg)
    if dt == "bits":
        return "_double_bits"
    if dt != "distortion":
        raise ValueError(f'model.double_target 应为 "bits" 或 "distortion"，当前为 {dt!r}')
    term = mse_term_normalized(cfg)
    if term == "psnr":
        return "_double_psnr"
    if term == "vmaf":
        return "_double_vmaf"
    return "_double_mse"


def phase_output_dirname(cfg: dict, phase: int) -> str:
    """输出目录名：phase{N} + pass1/no_pass1 后缀 + 单头失真后缀 或 double 后缀。"""
    base = f"phase{phase}"
    if phase == 2:
        v = phase2_variant(cfg)
        if v not in ("flat", "phase2_1"):
            raise ValueError(f'model.phase2_variant 必须为 "flat" 或 "phase2_1"，当前为 {v!r}')
        if v != "flat":
            base = v
    data_cfg = cfg["data"]
    if is_double_mode(cfg):
        dist_suf = _double_mode_dir_suffix(cfg)
    else:
        dist_suf = _distortion_loss_dir_suffix(cfg)
    if data_cfg.get("use_pass1_features", False):
        suf = str(data_cfg.get("output_phase_pass1_suffix", "_pass1"))
        return f"{base}{suf}{dist_suf}"
    suf = str(data_cfg.get("output_phase_no_pass1_suffix", "") or "")
    return f"{base}{suf}{dist_suf}"


def split_manifest(manifest, cfg):
    split_by_col = cfg["data"]["split_by_col"]
    train_keys, val_keys, test_keys = train_val_test_split(
        manifest[split_by_col].unique().tolist(),
        cfg["data"]["train_ratio"],
        cfg["data"]["val_ratio"],
        cfg["data"]["test_ratio"],
        cfg["seed"],
    )
    train_df = manifest[manifest[split_by_col].isin(train_keys)].reset_index(drop=True)
    eval_keys = list(dict.fromkeys(list(val_keys) + list(test_keys)))
    eval_df = manifest[manifest[split_by_col].isin(eval_keys)].reset_index(drop=True)
    return train_df, eval_df


def build_model(cfg: dict, phase: int):
    feature_profile = resolve_feature_profile(cfg, phase)
    self_dim = len(self_feature_names(feature_profile))
    pair_dim = len(pair_feature_names(feature_profile))
    meta_dim = len(meta_feature_names(feature_profile))
    pass1_dim = len(pass1_feature_names(cfg)) if cfg["data"].get("use_pass1_features", False) else 0
    head_out = model_head_out_dim(cfg)

    if phase == 1:
        return Phase1Net(self_dim=self_dim, meta_dim=meta_dim, pass1_dim=pass1_dim, cfg=cfg, out_dim=head_out)
    if phase == 2:
        v = phase2_variant(cfg)
        if v == "flat":
            return Phase2Net(
                self_dim=self_dim,
                pair_dim=pair_dim,
                meta_dim=meta_dim,
                pass1_dim=pass1_dim,
                cfg=cfg,
                out_dim=head_out,
            )
        if v == "phase2_1":
            return Phase2_1Net(
                self_dim=self_dim,
                pair_dim=pair_dim,
                meta_dim=meta_dim,
                pass1_dim=pass1_dim,
                cfg=cfg,
                out_dim=head_out,
            )
        raise ValueError(f'model.phase2_variant 必须为 "flat" 或 "phase2_1"，当前为 {v!r}')
    if phase == 3:
        return Phase3Net(self_dim=self_dim, pair_dim=pair_dim, meta_dim=meta_dim, pass1_dim=pass1_dim, cfg=cfg, head_out_dim=head_out)
    raise ValueError(f"Unsupported phase: {phase}")


def _dataloader_common_kwargs(train_cfg: dict) -> dict:
    """persistent_workers + prefetch 减轻「等 worker」；远端盘随机读仍可能很慢，见文档说明。"""
    nw = int(train_cfg["num_workers"])
    kwargs: dict = {
        "num_workers": nw,
        "pin_memory": True,
        "drop_last": False,
    }
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(2, int(train_cfg.get("prefetch_factor", 4)))
    return kwargs


def make_dataloaders(
    manifest,
    cfg,
    phase: int,
    *,
    distributed_train: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    train_df, eval_df = split_manifest(manifest, cfg)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    dl_kw = _dataloader_common_kwargs(train_cfg)

    if phase in (1, 2):
        train_ds = FrameDataset(manifest, cfg, train_df, phase=phase)
        eval_ds = FrameDataset(manifest, cfg, eval_df, phase=phase)
        batch_size = int(train_cfg[f"batch_size_phase{phase}"])
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            if distributed_train
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **dl_kw,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            **dl_kw,
        )
        return train_loader, eval_loader

    if phase == 3:
        train_ds = SegmentDataset(manifest, cfg, train_df)
        eval_ds = SegmentDataset(manifest, cfg, eval_df)
        batch_size = int(train_cfg["batch_size_phase3"])
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            if distributed_train
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **dl_kw,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            **dl_kw,
        )
        return train_loader, eval_loader

    raise ValueError(f"Unsupported phase: {phase}")


def _effective_huber_delta_psnr(cfg: dict) -> float:
    """mse_term=psnr 时专用：默认 train.huber_delta * huber_delta_psnr_scale，避免 dB 空间仍用 log 域 delta。"""
    lc = cfg["loss"]
    fixed = lc.get("huber_delta_psnr")
    if fixed is not None:
        return float(fixed)
    scale = float(lc.get("huber_delta_psnr_scale", 5.0))
    return float(cfg["train"]["huber_delta"]) * scale


def _effective_huber_delta_vmaf(cfg: dict) -> float:
    """mse_term=vmaf 时专用：默认 train.huber_delta * huber_delta_vmaf_scale。"""
    lc = cfg["loss"]
    fixed = lc.get("huber_delta_vmaf")
    if fixed is not None:
        return float(fixed)
    scale = float(lc.get("huber_delta_vmaf_scale", 2.0))
    return float(cfg["train"]["huber_delta"]) * scale


def _huber_distortion_term(
    pred_log_mse: torch.Tensor,
    target_log_mse: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
    cfg: dict,
) -> torch.Tensor:
    """第二维：log(mse) 空间 Huber，或 MSE→PSNR 后 PSNR 空间 Huber，或直接 VMAF 空间 Huber。"""
    term = mse_term_normalized(cfg)
    if term in ("log_mse", "mse", ""):
        return huber_loss_masked(pred_log_mse, target_log_mse, mask, delta)
    if term == "psnr":
        max_v = float(cfg["eval"]["max_psnr_value"])
        delta_psnr = _effective_huber_delta_psnr(cfg)
        mse_p = inverse_log_mse(pred_log_mse)
        mse_t = inverse_log_mse(target_log_mse)
        psnr_p = compute_psnr_from_mse_torch(mse_p, max_v)
        psnr_t = compute_psnr_from_mse_torch(mse_t, max_v)
        return huber_loss_masked(psnr_p, psnr_t, mask, delta_psnr)
    if term == "vmaf":
        delta_v = _effective_huber_delta_vmaf(cfg)
        return huber_loss_masked(pred_log_mse, target_log_mse, mask, delta_v)
    raise ValueError(
        f'loss.mse_term 必须是 "log_mse"、"psnr" 或 "vmaf"，当前为 {cfg["loss"].get("mse_term")!r}'
    )


def compute_loss(batch, outputs, cfg, phase: int):
    delta = float(cfg["train"]["huber_delta"])
    bits_w = float(cfg["loss"]["bits_weight"])
    mse_w = float(cfg["loss"]["mse_weight"])
    aux_w = float(cfg["loss"]["aux_weight"])
    mse_term = mse_term_normalized(cfg)

    if phase in (1, 2):
        pred = outputs["pred"]
        target = batch["target"]
        mask = batch["valid_mask"]
        if is_double_mode(cfg):
            if double_target(cfg) == "bits":
                loss_bits = huber_loss_masked(pred[..., 0:1], target[..., 0:1], mask, delta)
                loss = bits_w * loss_bits
                logs = {
                    "loss": float(loss.item()),
                    "loss_bits": float(loss_bits.item()),
                    "loss_mse": 0.0,
                    "mse_term": mse_term,
                    "model_mode": "double_bits",
                }
                return loss, logs
            loss_mse = _huber_distortion_term(pred[..., 0:1], target[..., 1:2], mask, delta, cfg)
            loss = mse_w * loss_mse
            logs = {
                "loss": float(loss.item()),
                "loss_bits": 0.0,
                "mse_term": mse_term,
                "model_mode": "double_distortion",
            }
            if mse_term == "vmaf":
                logs["loss_vmaf"] = float(loss_mse.item())
                logs["loss_mse"] = 0.0
                logs["distortion_huber_delta"] = _effective_huber_delta_vmaf(cfg)
            else:
                logs["loss_mse"] = float(loss_mse.item())
                if mse_term == "psnr":
                    logs["distortion_huber_delta"] = _effective_huber_delta_psnr(cfg)
            return loss, logs

        loss_bits = huber_loss_masked(pred[..., 0:1], target[..., 0:1], mask, delta)
        loss_mse = _huber_distortion_term(pred[..., 1:2], target[..., 1:2], mask, delta, cfg)
        loss = bits_w * loss_bits + mse_w * loss_mse
        logs = {
            "loss": float(loss.item()),
            "loss_bits": float(loss_bits.item()),
            "mse_term": mse_term,
        }
        if mse_term == "vmaf":
            logs["loss_vmaf"] = float(loss_mse.item())
            logs["loss_mse"] = 0.0
            logs["distortion_huber_delta"] = _effective_huber_delta_vmaf(cfg)
        else:
            logs["loss_mse"] = float(loss_mse.item())
            if mse_term == "psnr":
                logs["distortion_huber_delta"] = _effective_huber_delta_psnr(cfg)
        return loss, logs

    if phase == 3:
        pred = outputs["pred"]
        aux = outputs["aux_pred"]
        target = batch["targets"]
        mask = batch["valid_loss_mask"]

        if is_double_mode(cfg):
            if double_target(cfg) == "bits":
                loss_bits = huber_loss_masked(pred[..., 0:1], target[..., 0:1], mask, delta)
                loss_aux_bits = huber_loss_masked(aux[..., 0:1], target[..., 0:1], mask, delta)
                loss = bits_w * loss_bits + aux_w * loss_aux_bits
                logs = {
                    "loss": float(loss.item()),
                    "loss_bits": float(loss_bits.item()),
                    "loss_mse": 0.0,
                    "loss_aux_bits": float(loss_aux_bits.item()),
                    "loss_aux_mse": 0.0,
                    "mse_term": mse_term,
                    "model_mode": "double_bits",
                }
                return loss, logs
            loss_mse = _huber_distortion_term(pred[..., 0:1], target[..., 1:2], mask, delta, cfg)
            loss_aux_mse = _huber_distortion_term(aux[..., 0:1], target[..., 1:2], mask, delta, cfg)
            loss = mse_w * loss_mse + aux_w * loss_aux_mse
            logs = {
                "loss": float(loss.item()),
                "loss_bits": 0.0,
                "loss_aux_bits": 0.0,
                "mse_term": mse_term,
                "model_mode": "double_distortion",
            }
            if mse_term == "vmaf":
                logs["loss_vmaf"] = float(loss_mse.item())
                logs["loss_aux_vmaf"] = float(loss_aux_mse.item())
                logs["loss_mse"] = 0.0
                logs["loss_aux_mse"] = 0.0
                logs["distortion_huber_delta"] = _effective_huber_delta_vmaf(cfg)
            else:
                logs["loss_mse"] = float(loss_mse.item())
                logs["loss_aux_mse"] = float(loss_aux_mse.item())
                if mse_term == "psnr":
                    logs["distortion_huber_delta"] = _effective_huber_delta_psnr(cfg)
            return loss, logs

        loss_bits = huber_loss_masked(pred[..., 0:1], target[..., 0:1], mask, delta)
        loss_mse = _huber_distortion_term(pred[..., 1:2], target[..., 1:2], mask, delta, cfg)
        loss_aux_bits = huber_loss_masked(aux[..., 0:1], target[..., 0:1], mask, delta)
        loss_aux_mse = _huber_distortion_term(aux[..., 1:2], target[..., 1:2], mask, delta, cfg)

        loss_main = bits_w * loss_bits + mse_w * loss_mse
        loss_aux = bits_w * loss_aux_bits + mse_w * loss_aux_mse
        loss = loss_main + aux_w * loss_aux
        logs = {
            "loss": float(loss.item()),
            "loss_bits": float(loss_bits.item()),
            "loss_aux_bits": float(loss_aux_bits.item()),
            "mse_term": mse_term,
        }
        if mse_term == "vmaf":
            logs["loss_vmaf"] = float(loss_mse.item())
            logs["loss_aux_vmaf"] = float(loss_aux_mse.item())
            logs["loss_mse"] = 0.0
            logs["loss_aux_mse"] = 0.0
            logs["distortion_huber_delta"] = _effective_huber_delta_vmaf(cfg)
        else:
            logs["loss_mse"] = float(loss_mse.item())
            logs["loss_aux_mse"] = float(loss_aux_mse.item())
            if mse_term == "psnr":
                logs["distortion_huber_delta"] = _effective_huber_delta_psnr(cfg)
        return loss, logs

    raise ValueError(f"Unsupported phase: {phase}")


def _group_metrics(bits_true, bits_pred, mse_true, mse_pred, psnr_true, psnr_pred, group_ids, label_fn):
    out = {}
    if len(group_ids) == 0:
        return out

    for gid in sorted(set(group_ids.tolist())):
        mask = group_ids == gid
        if not np.any(mask):
            continue
        out[label_fn(int(gid))] = {
            "count": int(mask.sum()),
            "bits": regression_metrics(bits_true[mask], bits_pred[mask]),
            "mse": regression_metrics(mse_true[mask], mse_pred[mask]),
            "psnr": regression_metrics(psnr_true[mask], psnr_pred[mask]),
        }
    return out


def _group_metrics_bits_only(bits_true, bits_pred, group_ids, label_fn):
    out = {}
    if len(group_ids) == 0:
        return out
    for gid in sorted(set(group_ids.tolist())):
        mask = group_ids == gid
        if not np.any(mask):
            continue
        out[label_fn(int(gid))] = {
            "count": int(mask.sum()),
            "bits": regression_metrics(bits_true[mask], bits_pred[mask]),
        }
    return out


def _group_metrics_distortion_only(mse_true, mse_pred, psnr_true, psnr_pred, group_ids, label_fn):
    out = {}
    if len(group_ids) == 0:
        return out
    for gid in sorted(set(group_ids.tolist())):
        mask = group_ids == gid
        if not np.any(mask):
            continue
        out[label_fn(int(gid))] = {
            "count": int(mask.sum()),
            "mse": regression_metrics(mse_true[mask], mse_pred[mask]),
            "psnr": regression_metrics(psnr_true[mask], psnr_pred[mask]),
        }
    return out


def _mean_aggregate_metrics(
    bits_true: np.ndarray,
    bits_pred: np.ndarray,
    mse_true: np.ndarray,
    mse_pred: np.ndarray,
    psnr_true: np.ndarray,
    psnr_pred: np.ndarray,
    sequences: np.ndarray,
) -> dict:
    """全局样本均值差 + 按序列先求均值再对序列取平均的宏平均差（预测 - 真实）。"""
    n = int(len(bits_true))
    if n == 0:
        return {}

    def _triplet(t_true: np.ndarray, t_pred: np.ndarray) -> dict:
        return {
            "mean_true": float(np.mean(t_true)),
            "mean_pred": float(np.mean(t_pred)),
            "diff_pred_minus_true": float(np.mean(t_pred - t_true)),
        }

    out: dict = {
        "global_sample_means": {
            "bits": _triplet(bits_true, bits_pred),
            "mse": _triplet(mse_true, mse_pred),
            "psnr": _triplet(psnr_true, psnr_pred),
        },
    }

    uniq = np.unique(sequences)
    if len(uniq) == 0:
        return out

    def _macro_diff(t_true: np.ndarray, t_pred: np.ndarray) -> dict:
        diffs = []
        for s in uniq:
            m = sequences == s
            if not np.any(m):
                continue
            diffs.append(float(np.mean(t_pred[m]) - np.mean(t_true[m])))
        return {
            "mean_diff_pred_minus_true": float(np.mean(diffs)) if diffs else 0.0,
            "num_sequences": int(len(diffs)),
        }

    out["macro_by_sequence"] = {
        "bits": _macro_diff(bits_true, bits_pred),
        "mse": _macro_diff(mse_true, mse_pred),
        "psnr": _macro_diff(psnr_true, psnr_pred),
    }
    return out


def _mean_aggregate_bits_only(
    bits_true: np.ndarray,
    bits_pred: np.ndarray,
    sequences: np.ndarray,
) -> dict:
    n = int(len(bits_true))
    if n == 0:
        return {}

    def _triplet(t_true: np.ndarray, t_pred: np.ndarray) -> dict:
        return {
            "mean_true": float(np.mean(t_true)),
            "mean_pred": float(np.mean(t_pred)),
            "diff_pred_minus_true": float(np.mean(t_pred - t_true)),
        }

    out: dict = {
        "global_sample_means": {"bits": _triplet(bits_true, bits_pred)},
    }
    uniq = np.unique(sequences)
    if len(uniq) == 0:
        return out

    def _macro_diff(t_true: np.ndarray, t_pred: np.ndarray) -> dict:
        diffs = []
        for s in uniq:
            m = sequences == s
            if not np.any(m):
                continue
            diffs.append(float(np.mean(t_pred[m]) - np.mean(t_true[m])))
        return {
            "mean_diff_pred_minus_true": float(np.mean(diffs)) if diffs else 0.0,
            "num_sequences": int(len(diffs)),
        }

    out["macro_by_sequence"] = {"bits": _macro_diff(bits_true, bits_pred)}
    return out


def _mean_aggregate_distortion_only(
    mse_true: np.ndarray,
    mse_pred: np.ndarray,
    psnr_true: np.ndarray,
    psnr_pred: np.ndarray,
    sequences: np.ndarray,
) -> dict:
    n = int(len(mse_true))
    if n == 0:
        return {}

    def _triplet(t_true: np.ndarray, t_pred: np.ndarray) -> dict:
        return {
            "mean_true": float(np.mean(t_true)),
            "mean_pred": float(np.mean(t_pred)),
            "diff_pred_minus_true": float(np.mean(t_pred - t_true)),
        }

    out: dict = {
        "global_sample_means": {
            "mse": _triplet(mse_true, mse_pred),
            "psnr": _triplet(psnr_true, psnr_pred),
        },
    }
    uniq = np.unique(sequences)
    if len(uniq) == 0:
        return out

    def _macro_diff(t_true: np.ndarray, t_pred: np.ndarray) -> dict:
        diffs = []
        for s in uniq:
            m = sequences == s
            if not np.any(m):
                continue
            diffs.append(float(np.mean(t_pred[m]) - np.mean(t_true[m])))
        return {
            "mean_diff_pred_minus_true": float(np.mean(diffs)) if diffs else 0.0,
            "num_sequences": int(len(diffs)),
        }

    out["macro_by_sequence"] = {
        "mse": _macro_diff(mse_true, mse_pred),
        "psnr": _macro_diff(psnr_true, psnr_pred),
    }
    return out


def _group_metrics_vmaf_only(vmaf_true, vmaf_pred, group_ids, label_fn):
    out = {}
    if len(group_ids) == 0:
        return out
    for gid in sorted(set(group_ids.tolist())):
        mask = group_ids == gid
        if not np.any(mask):
            continue
        out[label_fn(int(gid))] = {
            "count": int(mask.sum()),
            "vmaf": regression_metrics(vmaf_true[mask], vmaf_pred[mask]),
        }
    return out


def _group_metrics_bits_vmaf(bits_true, bits_pred, vmaf_true, vmaf_pred, group_ids, label_fn):
    out = {}
    if len(group_ids) == 0:
        return out
    for gid in sorted(set(group_ids.tolist())):
        mask = group_ids == gid
        if not np.any(mask):
            continue
        out[label_fn(int(gid))] = {
            "count": int(mask.sum()),
            "bits": regression_metrics(bits_true[mask], bits_pred[mask]),
            "vmaf": regression_metrics(vmaf_true[mask], vmaf_pred[mask]),
        }
    return out


def _mean_aggregate_vmaf_only(vmaf_true: np.ndarray, vmaf_pred: np.ndarray, sequences: np.ndarray) -> dict:
    n = int(len(vmaf_true))
    if n == 0:
        return {}

    def _triplet(t_true: np.ndarray, t_pred: np.ndarray) -> dict:
        return {
            "mean_true": float(np.mean(t_true)),
            "mean_pred": float(np.mean(t_pred)),
            "diff_pred_minus_true": float(np.mean(t_pred - t_true)),
        }

    out: dict = {
        "global_sample_means": {"vmaf": _triplet(vmaf_true, vmaf_pred)},
    }
    uniq = np.unique(sequences)
    if len(uniq) == 0:
        return out

    def _macro_diff(t_true: np.ndarray, t_pred: np.ndarray) -> dict:
        diffs = []
        for s in uniq:
            m = sequences == s
            if not np.any(m):
                continue
            diffs.append(float(np.mean(t_pred[m]) - np.mean(t_true[m])))
        return {
            "mean_diff_pred_minus_true": float(np.mean(diffs)) if diffs else 0.0,
            "num_sequences": int(len(diffs)),
        }

    out["macro_by_sequence"] = {"vmaf": _macro_diff(vmaf_true, vmaf_pred)}
    return out


def _mean_aggregate_bits_vmaf(
    bits_true: np.ndarray,
    bits_pred: np.ndarray,
    vmaf_true: np.ndarray,
    vmaf_pred: np.ndarray,
    sequences: np.ndarray,
) -> dict:
    n = int(len(bits_true))
    if n == 0:
        return {}

    def _triplet(t_true: np.ndarray, t_pred: np.ndarray) -> dict:
        return {
            "mean_true": float(np.mean(t_true)),
            "mean_pred": float(np.mean(t_pred)),
            "diff_pred_minus_true": float(np.mean(t_pred - t_true)),
        }

    out: dict = {
        "global_sample_means": {
            "bits": _triplet(bits_true, bits_pred),
            "vmaf": _triplet(vmaf_true, vmaf_pred),
        },
    }
    uniq = np.unique(sequences)
    if len(uniq) == 0:
        return out

    def _macro_diff(t_true: np.ndarray, t_pred: np.ndarray) -> dict:
        diffs = []
        for s in uniq:
            m = sequences == s
            if not np.any(m):
                continue
            diffs.append(float(np.mean(t_pred[m]) - np.mean(t_true[m])))
        return {
            "mean_diff_pred_minus_true": float(np.mean(diffs)) if diffs else 0.0,
            "num_sequences": int(len(diffs)),
        }

    out["macro_by_sequence"] = {
        "bits": _macro_diff(bits_true, bits_pred),
        "vmaf": _macro_diff(vmaf_true, vmaf_pred),
    }
    return out


@torch.no_grad()
def evaluate_loader(model, loader, device, cfg, phase: int, *, rank: int = 0):
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    loss_meter = []
    bits_true_all = []
    bits_pred_all = []
    mse_true_all = []
    mse_pred_all = []
    vmaf_true_chunks: list[np.ndarray] = []
    vmaf_pred_chunks: list[np.ndarray] = []
    frame_type_all = []
    temporal_layer_all = []
    sequence_all = []

    for batch in tqdm(loader, desc="eval", leave=False, disable=(rank != 0)):
        batch = to_device(batch, device)
        outputs = model(batch)
        _, logs = compute_loss(batch, outputs, cfg, phase)
        loss_meter.append(logs["loss"])

        if phase in (1, 2):
            pred = outputs["pred"]
            target = batch["target"]
            valid_mask = batch["valid_mask"] > 0.5
            frame_type_ids = batch["frame_type_id"]
            temporal_layers = batch["temporal_layer"]
        else:
            pred = outputs["pred"]
            target = batch["targets"]
            valid_mask = batch["valid_loss_mask"] > 0.5
            frame_type_ids = batch["frame_type_ids"]
            temporal_layers = batch["temporal_layers"]

        vm_np = valid_mask.detach().cpu().numpy()
        if phase in (1, 2):
            seq_batch = batch["sequence"]
            seq_np = np.asarray(seq_batch, dtype=object)[vm_np]
        else:
            B, T = vm_np.shape
            seq_list = batch["sequence"]
            rows: list = []
            for b in range(B):
                s = seq_list[b]
                for t in range(T):
                    if vm_np[b, t]:
                        rows.append(s)
            seq_np = np.asarray(rows, dtype=object)

        pred = pred[valid_mask]
        target = target[valid_mask]
        frame_type_ids = frame_type_ids[valid_mask].detach().cpu().numpy()
        temporal_layers = temporal_layers[valid_mask].detach().cpu().numpy()

        if is_double_mode(cfg) and double_target(cfg) == "bits":
            bits_pred = inverse_log_bits(pred[:, 0]).detach().cpu().numpy()
            bits_true = inverse_log_bits(target[:, 0]).detach().cpu().numpy()
            bits_true_all.append(bits_true)
            bits_pred_all.append(bits_pred)
            frame_type_all.append(frame_type_ids)
            temporal_layer_all.append(temporal_layers)
            sequence_all.append(seq_np)
        elif is_double_mode(cfg) and double_target(cfg) == "distortion":
            if is_vmaf_distortion(cfg):
                vmaf_pred = pred[:, 0].detach().cpu().numpy()
                vmaf_true = target[:, 1].detach().cpu().numpy()
                vmaf_pred_chunks.append(vmaf_pred)
                vmaf_true_chunks.append(vmaf_true)
            else:
                mse_pred = inverse_log_mse(pred[:, 0]).detach().cpu().numpy()
                mse_true = inverse_log_mse(target[:, 1]).detach().cpu().numpy()
                mse_true_all.append(mse_true)
                mse_pred_all.append(mse_pred)
            frame_type_all.append(frame_type_ids)
            temporal_layer_all.append(temporal_layers)
            sequence_all.append(seq_np)
        else:
            bits_pred = inverse_log_bits(pred[:, 0]).detach().cpu().numpy()
            bits_true = inverse_log_bits(target[:, 0]).detach().cpu().numpy()
            bits_true_all.append(bits_true)
            bits_pred_all.append(bits_pred)
            if is_vmaf_distortion(cfg):
                vmaf_pred = pred[:, 1].detach().cpu().numpy()
                vmaf_true = target[:, 1].detach().cpu().numpy()
                vmaf_pred_chunks.append(vmaf_pred)
                vmaf_true_chunks.append(vmaf_true)
            else:
                mse_pred = inverse_log_mse(pred[:, 1]).detach().cpu().numpy()
                mse_true = inverse_log_mse(target[:, 1]).detach().cpu().numpy()
                mse_true_all.append(mse_true)
                mse_pred_all.append(mse_pred)
            frame_type_all.append(frame_type_ids)
            temporal_layer_all.append(temporal_layers)
            sequence_all.append(seq_np)

    bits_true_all = np.concatenate(bits_true_all, axis=0) if bits_true_all else np.zeros((0,))
    bits_pred_all = np.concatenate(bits_pred_all, axis=0) if bits_pred_all else np.zeros((0,))
    mse_true_all = np.concatenate(mse_true_all, axis=0) if mse_true_all else np.zeros((0,))
    mse_pred_all = np.concatenate(mse_pred_all, axis=0) if mse_pred_all else np.zeros((0,))
    frame_type_all = np.concatenate(frame_type_all, axis=0) if frame_type_all else np.zeros((0,), dtype=np.int64)
    temporal_layer_all = np.concatenate(temporal_layer_all, axis=0) if temporal_layer_all else np.zeros((0,), dtype=np.int64)
    sequence_concat = np.concatenate(sequence_all, axis=0) if sequence_all else np.asarray([], dtype=object)
    vmaf_true_arr = np.concatenate(vmaf_true_chunks, axis=0) if vmaf_true_chunks else np.zeros((0,))
    vmaf_pred_arr = np.concatenate(vmaf_pred_chunks, axis=0) if vmaf_pred_chunks else np.zeros((0,))

    from .utils import compute_psnr_from_mse

    loss_f = float(np.mean(loss_meter)) if loss_meter else 0.0
    ft_label = lambda x: {0: "I", 1: "P", 2: "B", 3: "UNK"}.get(x, f"UNK_{x}")

    if is_double_mode(cfg) and double_target(cfg) == "bits":
        mean_agg = {}
        if len(bits_true_all) > 0 and len(sequence_concat) == len(bits_true_all):
            mean_agg = _mean_aggregate_bits_only(bits_true_all, bits_pred_all, sequence_concat)
        metrics = {
            "loss": loss_f,
            "mse_term": mse_term_normalized(cfg),
            "bits": regression_metrics(bits_true_all, bits_pred_all) if len(bits_true_all) > 0 else {},
            "mse": {},
            "psnr": {},
            "by_frame_type": _group_metrics_bits_only(bits_true_all, bits_pred_all, frame_type_all, label_fn=ft_label),
            "by_temporal_layer": _group_metrics_bits_only(
                bits_true_all, bits_pred_all, temporal_layer_all, label_fn=lambda x: str(x)
            ),
            "model_mode": "double_bits",
        }
        if mean_agg:
            metrics["mean_aggregate"] = mean_agg
        return metrics

    if is_double_mode(cfg) and double_target(cfg) == "distortion":
        if is_vmaf_distortion(cfg):
            mean_agg = {}
            if len(vmaf_true_arr) > 0 and len(sequence_concat) == len(vmaf_true_arr):
                mean_agg = _mean_aggregate_vmaf_only(vmaf_true_arr, vmaf_pred_arr, sequence_concat)
            metrics = {
                "loss": loss_f,
                "mse_term": "vmaf",
                "vmaf": regression_metrics(vmaf_true_arr, vmaf_pred_arr) if len(vmaf_true_arr) > 0 else {},
                "by_frame_type": _group_metrics_vmaf_only(
                    vmaf_true_arr, vmaf_pred_arr, frame_type_all, label_fn=ft_label
                ),
                "by_temporal_layer": _group_metrics_vmaf_only(
                    vmaf_true_arr, vmaf_pred_arr, temporal_layer_all, label_fn=lambda x: str(x)
                ),
                "model_mode": "double_distortion",
            }
            if mean_agg:
                metrics["mean_aggregate"] = mean_agg
            return metrics

        psnr_true = compute_psnr_from_mse(mse_true_all, max_value=float(cfg["eval"]["max_psnr_value"]))
        psnr_pred = compute_psnr_from_mse(mse_pred_all, max_value=float(cfg["eval"]["max_psnr_value"]))
        mean_agg = {}
        if len(mse_true_all) > 0 and len(sequence_concat) == len(mse_true_all):
            mean_agg = _mean_aggregate_distortion_only(
                mse_true_all, mse_pred_all, psnr_true, psnr_pred, sequence_concat
            )
        metrics = {
            "loss": loss_f,
            "mse_term": mse_term_normalized(cfg),
            "bits": {},
            "mse": regression_metrics(mse_true_all, mse_pred_all) if len(mse_true_all) > 0 else {},
            "psnr": regression_metrics(psnr_true, psnr_pred) if len(psnr_true) > 0 else {},
            "by_frame_type": _group_metrics_distortion_only(
                mse_true_all, mse_pred_all, psnr_true, psnr_pred, frame_type_all, label_fn=ft_label
            ),
            "by_temporal_layer": _group_metrics_distortion_only(
                mse_true_all,
                mse_pred_all,
                psnr_true,
                psnr_pred,
                temporal_layer_all,
                label_fn=lambda x: str(x),
            ),
            "model_mode": "double_distortion",
        }
        if mean_agg:
            metrics["mean_aggregate"] = mean_agg
        return metrics

    if not is_double_mode(cfg) and is_vmaf_distortion(cfg):
        mean_agg = {}
        if len(bits_true_all) > 0 and len(sequence_concat) == len(bits_true_all):
            mean_agg = _mean_aggregate_bits_vmaf(
                bits_true_all, bits_pred_all, vmaf_true_arr, vmaf_pred_arr, sequence_concat
            )
        metrics = {
            "loss": loss_f,
            "mse_term": "vmaf",
            "bits": regression_metrics(bits_true_all, bits_pred_all) if len(bits_true_all) > 0 else {},
            "vmaf": regression_metrics(vmaf_true_arr, vmaf_pred_arr) if len(vmaf_true_arr) > 0 else {},
            "by_frame_type": _group_metrics_bits_vmaf(
                bits_true_all,
                bits_pred_all,
                vmaf_true_arr,
                vmaf_pred_arr,
                frame_type_all,
                label_fn=ft_label,
            ),
            "by_temporal_layer": _group_metrics_bits_vmaf(
                bits_true_all,
                bits_pred_all,
                vmaf_true_arr,
                vmaf_pred_arr,
                temporal_layer_all,
                label_fn=lambda x: str(x),
            ),
            "model_mode": "single",
        }
        if mean_agg:
            metrics["mean_aggregate"] = mean_agg
        return metrics

    psnr_true = compute_psnr_from_mse(mse_true_all, max_value=float(cfg["eval"]["max_psnr_value"]))
    psnr_pred = compute_psnr_from_mse(mse_pred_all, max_value=float(cfg["eval"]["max_psnr_value"]))

    mean_agg = {}
    if len(bits_true_all) > 0 and len(sequence_concat) == len(bits_true_all):
        mean_agg = _mean_aggregate_metrics(
            bits_true_all,
            bits_pred_all,
            mse_true_all,
            mse_pred_all,
            psnr_true,
            psnr_pred,
            sequence_concat,
        )

    metrics = {
        "loss": loss_f,
        "mse_term": mse_term_normalized(cfg),
        "bits": regression_metrics(bits_true_all, bits_pred_all) if len(bits_true_all) > 0 else {},
        "mse": regression_metrics(mse_true_all, mse_pred_all) if len(mse_true_all) > 0 else {},
        "psnr": regression_metrics(psnr_true, psnr_pred) if len(psnr_true) > 0 else {},
        "by_frame_type": _group_metrics(
            bits_true_all,
            bits_pred_all,
            mse_true_all,
            mse_pred_all,
            psnr_true,
            psnr_pred,
            frame_type_all,
            label_fn=ft_label,
        ),
        "by_temporal_layer": _group_metrics(
            bits_true_all,
            bits_pred_all,
            mse_true_all,
            mse_pred_all,
            psnr_true,
            psnr_pred,
            temporal_layer_all,
            label_fn=lambda x: str(x),
        ),
        "model_mode": "single",
    }
    if mean_agg:
        metrics["mean_aggregate"] = mean_agg
    return metrics


def _fmt_metric_val(x: float) -> str:
    if not np.isfinite(x):
        return "-"
    ax = abs(float(x))
    if ax >= 1e6:
        return f"{x:.3e}"
    if ax >= 1e4:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.2f}"
    return f"{x:.4g}"


def _fmt_reg_line(tag: str, m: dict) -> str:
    if not m:
        return f"{tag}: (无样本)"
    return (
        f"{tag}: MAE={_fmt_metric_val(m['mae'])}  RMSE={_fmt_metric_val(m['rmse'])}  "
        f"R²={m['r2']:.4f}"
    )


def _format_mean_aggregate_lines(ma: dict) -> list[str]:
    lines: list[str] = []
    g = ma.get("global_sample_means") or {}
    m = ma.get("macro_by_sequence") or {}
    if not g and not m:
        return lines

    lines.append("           -- 均值对比 (预测 - 真实) --")
    if g:
        parts: list[str] = []
        if "bits" in g and g.get("bits"):
            bg = g["bits"]
            parts.append(
                "bits: pred="
                f"{_fmt_metric_val(bg.get('mean_pred', float('nan')))}  "
                f"true={_fmt_metric_val(bg.get('mean_true', float('nan')))}  "
                f"diff={_fmt_metric_val(bg.get('diff_pred_minus_true', float('nan')))}"
            )
        if "mse" in g and g.get("mse"):
            mg = g["mse"]
            parts.append(
                "mse: pred="
                f"{_fmt_metric_val(mg.get('mean_pred', float('nan')))}  "
                f"true={_fmt_metric_val(mg.get('mean_true', float('nan')))}  "
                f"diff={_fmt_metric_val(mg.get('diff_pred_minus_true', float('nan')))}"
            )
        if "psnr" in g and g.get("psnr"):
            pg = g["psnr"]
            parts.append(
                "psnr: pred="
                f"{_fmt_metric_val(pg.get('mean_pred', float('nan')))}  "
                f"true={_fmt_metric_val(pg.get('mean_true', float('nan')))}  "
                f"diff={_fmt_metric_val(pg.get('diff_pred_minus_true', float('nan')))}"
            )
        if "vmaf" in g and g.get("vmaf"):
            vg = g["vmaf"]
            parts.append(
                "vmaf: pred="
                f"{_fmt_metric_val(vg.get('mean_pred', float('nan')))}  "
                f"true={_fmt_metric_val(vg.get('mean_true', float('nan')))}  "
                f"diff={_fmt_metric_val(vg.get('diff_pred_minus_true', float('nan')))}"
            )
        if parts:
            lines.append("           [全局样本均值]  " + "  |  ".join(parts))
    if m:
        # 序列数：优先 bits，否则 vmaf，否则 mse
        nseq = 0
        if m.get("bits"):
            nseq = int(m["bits"].get("num_sequences", 0))
        elif m.get("vmaf"):
            nseq = int(m["vmaf"].get("num_sequences", 0))
        elif m.get("mse"):
            nseq = int(m["mse"].get("num_sequences", 0))
        lines.append(
            "           [按序列宏平均] 每序列先算均值再对序列平均 diff=mean_s(mean_pred|s-mean_true|s)；"
            f" 序列数={nseq}"
        )
        m_parts: list[str] = []
        if m.get("bits"):
            b = m["bits"]
            m_parts.append(f"bits diff={_fmt_metric_val(b.get('mean_diff_pred_minus_true', float('nan')))}")
        if m.get("mse"):
            ms = m["mse"]
            m_parts.append(f"mse diff={_fmt_metric_val(ms.get('mean_diff_pred_minus_true', float('nan')))}")
        if m.get("psnr"):
            pp = m["psnr"]
            m_parts.append(f"psnr diff={_fmt_metric_val(pp.get('mean_diff_pred_minus_true', float('nan')))}")
        if m.get("vmaf"):
            vm = m["vmaf"]
            m_parts.append(f"vmaf diff={_fmt_metric_val(vm.get('mean_diff_pred_minus_true', float('nan')))}")
        if m_parts:
            lines.append("                         " + "  |  ".join(m_parts))
    return lines


def format_eval_metrics_block(name: str, metrics: dict) -> str:
    """将 evaluate_loader 返回的字典格式化为多行可读文本（含分组表）。"""
    lines: list[str] = []
    loss = metrics.get("loss", float("nan"))
    lines.append(f"  [{name}]  loss = {loss:.6f}")

    ma = metrics.get("mean_aggregate")
    if isinstance(ma, dict) and ma:
        lines.extend(_format_mean_aggregate_lines(ma))

    mse_term = str(metrics.get("mse_term", "")).lower().strip()
    model_mode = str(metrics.get("model_mode", ""))
    vmaf_pure = mse_term == "vmaf" and model_mode == "double_distortion"
    single_vmaf = mse_term == "vmaf" and model_mode == "single"

    for key in ("bits", "mse", "psnr", "vmaf"):
        if key in metrics and metrics[key]:
            lines.append(f"           {_fmt_reg_line(key, metrics[key])}")

    for block_title, key in (
        ("按帧类型", "by_frame_type"),
        ("按时域层 TL", "by_temporal_layer"),
    ):
        sub = metrics.get(key) or {}
        if not sub:
            continue
        lines.append(f"           -- {block_title} --")

        def sort_key(item):
            label, _ = item
            order = {"I": 0, "P": 1, "B": 2}
            if label in order:
                return (0, order[label])
            try:
                return (1, int(label))
            except ValueError:
                return (2, label)

        if vmaf_pure:
            header = f"           {'subset':<8} {'n':>7} | {'vmaf_RMSE':>12} {'vmaf_R2':>8}"
            lines.append(header)
            lines.append("           " + "-" * 42)
            for label, row in sorted(sub.items(), key=sort_key):
                v = row.get("vmaf") or {}
                n = int(row.get("count", 0))
                lines.append(
                    f"           {label:<8} {n:>7} | "
                    f"{_fmt_metric_val(v.get('rmse', float('nan'))):>12} {v.get('r2', float('nan')):>8.4f}"
                )
        elif single_vmaf:
            header = (
                f"           {'subset':<8} {'n':>7} | "
                f"{'bits_RMSE':>12} {'bits_R2':>8} | "
                f"{'vmaf_RMSE':>12} {'vmaf_R2':>8}"
            )
            lines.append(header)
            lines.append("           " + "-" * 60)
            for label, row in sorted(sub.items(), key=sort_key):
                b = row.get("bits") or {}
                v = row.get("vmaf") or {}
                n = int(row.get("count", 0))
                lines.append(
                    f"           {label:<8} {n:>7} | "
                    f"{_fmt_metric_val(b.get('rmse', float('nan'))):>12} {b.get('r2', float('nan')):>8.4f} | "
                    f"{_fmt_metric_val(v.get('rmse', float('nan'))):>12} {v.get('r2', float('nan')):>8.4f}"
                )
        else:
            header = (
                f"           {'subset':<8} {'n':>7} | "
                f"{'bits_RMSE':>12} {'bits_R2':>8} | "
                f"{'mse_RMSE':>10} {'psnr_RMSE':>10}"
            )
            lines.append(header)
            lines.append("           " + "-" * 64)

            for label, row in sorted(sub.items(), key=sort_key):
                b = row.get("bits") or {}
                mm = row.get("mse") or {}
                p = row.get("psnr") or {}
                n = int(row.get("count", 0))
                lines.append(
                    f"           {label:<8} {n:>7} | "
                    f"{_fmt_metric_val(b.get('rmse', float('nan'))):>12} {b.get('r2', float('nan')):>8.4f} | "
                    f"{_fmt_metric_val(mm.get('rmse', float('nan'))):>10} {_fmt_metric_val(p.get('rmse', float('nan'))):>10}"
                )

    return "\n".join(lines)


def print_epoch_metrics(train_log: dict, train_metrics: dict, eval_metrics: dict) -> None:
    """终端打印：优化 loss + 训练集 eval 指标 + eval。"""
    opt = train_log.get("loss", float("nan"))
    print(f"  [train]  opt_loss = {opt:.6f}")
    print(format_eval_metrics_block("train (eval)", train_metrics))
    print(format_eval_metrics_block("eval", eval_metrics))
    print("  (完整指标仍写入 history.json)")


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    cfg,
    phase: int,
    *,
    rank: int = 0,
    bench_steps: int | None = None,
):
    model.train()
    train_cfg = cfg["train"]
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"

    def train_step_batch(batch) -> float:
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            outputs = model(batch)
            loss, _ = compute_loss(batch, outputs, cfg, phase)

        loss_f = float(loss.item())
        if not math.isfinite(loss_f):
            raise RuntimeError(f"训练损失非有限值 (nan/inf)={loss_f}，请检查数据、学习率与 AMP 设置。")

        if amp_enabled:
            scaler.scale(loss).backward()
            if float(train_cfg["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=float(train_cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if float(train_cfg["grad_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=float(train_cfg["grad_clip"]))
            optimizer.step()

        return loss_f

    meter = []

    if bench_steps is not None and bench_steps > 0:
        data_ms_list: list[float] = []
        compute_ms_list: list[float] = []
        it = iter(loader)
        for step in range(bench_steps):
            try:
                t0 = time.perf_counter()
                batch = next(it)
            except StopIteration:
                if rank == 0:
                    print(f"[bench_data] DataLoader 在 step {step} 用尽（数据集比请求的步数短）。")
                break
            batch = to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            loss_f = train_step_batch(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            data_ms = (t1 - t0) * 1000.0
            compute_ms = (t2 - t1) * 1000.0
            data_ms_list.append(data_ms)
            compute_ms_list.append(compute_ms)
            meter.append(loss_f)
            if rank == 0:
                print(
                    f"[bench_data] step {step + 1}/{bench_steps}  "
                    f"data_ms={data_ms:.2f}  compute_ms={compute_ms:.2f}  loss={loss_f:.4f}"
                )

        if rank == 0 and data_ms_list:
            n = len(data_ms_list)
            d_mean = float(np.mean(data_ms_list))
            c_mean = float(np.mean(compute_ms_list))
            print(
                f"[bench_data] 汇总 (共 {n} 步):  data 均值={d_mean:.2f} ms  compute 均值={c_mean:.2f} ms"
            )
            if n >= 2:
                d_skip = float(np.mean(data_ms_list[1:]))
                c_skip = float(np.mean(compute_ms_list[1:]))
                print(
                    f"[bench_data] 汇总 (跳过第 1 步冷启动, {n - 1} 步):  "
                    f"data 均值={d_skip:.2f} ms  compute 均值={c_skip:.2f} ms"
                )
        return {"loss": float(np.mean(meter)) if meter else 0.0}

    pbar = tqdm(loader, desc="train", leave=False, disable=(rank != 0))
    for batch in pbar:
        batch = to_device(batch, device)
        loss_f = train_step_batch(batch)
        meter.append(loss_f)
        pbar.set_postfix(loss=f"{np.mean(meter):.4f}")

    return {"loss": float(np.mean(meter)) if meter else 0.0}


def main():
    parser = argparse.ArgumentParser(
        description="QP 状态预测训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（脚本循环改参数即可，无需改 YAML）:
  python -m qp_predictor.train --config my_config.yaml --phase 1 --model-mode single --mse-term psnr
  python -m qp_predictor.train --config my_config.yaml --phase 2 --model-mode double --double-target bits
  python -m qp_predictor.train --config my_config.yaml --phase 1 --model-mode double --double-target distortion --mse-term log_mse
未写 CLI 时可用环境变量: QP_TRAIN_MODEL_MODE, QP_TRAIN_DOUBLE_TARGET, QP_TRAIN_MSE_TERM
""",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument(
        "--model-mode",
        default=None,
        choices=["single", "double"],
        help="覆盖 model.mode；未设时可读环境变量 QP_TRAIN_MODEL_MODE。",
    )
    parser.add_argument(
        "--double-target",
        default=None,
        choices=["bits", "distortion"],
        dest="double_target",
        help="覆盖 model.double_target；未设时可读 QP_TRAIN_DOUBLE_TARGET。",
    )
    parser.add_argument(
        "--mse-term",
        default=None,
        metavar="TERM",
        dest="mse_term",
        help="覆盖 loss.mse_term：log_mse | mse | logmse | psnr | vmaf；未设时可读 QP_TRAIN_MSE_TERM。",
    )
    parser.add_argument(
        "--bench-data",
        "--bench_data",
        type=int,
        default=0,
        metavar="N",
        help="仅跑 N 个训练 step，打印每步 data_ms / compute_ms 后退出；0 表示正常训练。",
    )
    parser.add_argument(
        "--metrics-json",
        action="store_true",
        help="每个 epoch 仍打印 train/eval 的完整 JSON（调试）；默认使用紧凑表格。",
    )
    args = parser.parse_args()

    local_rank, world_size, use_ddp = setup_distributed()
    rank = dist.get_rank() if use_ddp else 0

    cfg = load_config(args.config)
    mm, mm_src = resolve_train_override_cli_env(args.model_mode, ENV_TRAIN_MODEL_MODE)
    dt, dt_src = resolve_train_override_cli_env(args.double_target, ENV_TRAIN_DOUBLE_TARGET)
    mt, mt_src = resolve_train_override_cli_env(args.mse_term, ENV_TRAIN_MSE_TERM)
    ov_msgs = apply_train_overrides(
        cfg,
        model_mode=mm,
        double_target=dt,
        mse_term=mt,
        model_mode_src=mm_src,
        double_target_src=dt_src,
        mse_term_src=mt_src,
    )
    if rank == 0 and ov_msgs:
        print("[train overrides]", " | ".join(ov_msgs))

    if is_double_mode(cfg):
        dt = double_target(cfg)
        if dt not in ("bits", "distortion"):
            raise ValueError(
                f'model.double_target 必须为 "bits" 或 "distortion"，当前为 {dt!r}（mode=double 时必填）'
            )
    set_seed(int(cfg["seed"]) + rank)

    manifest = build_manifest(cfg)
    model = build_model(cfg, phase=args.phase)

    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, eval_loader = make_dataloaders(
        manifest,
        cfg,
        args.phase,
        distributed_train=use_ddp,
        rank=rank,
        world_size=world_size,
    )

    if len(train_loader.dataset) == 0:
        raise RuntimeError("训练集样本数为 0：请检查 labels_csv、valid_train 与划分。")
    n_split_units = manifest[cfg["data"]["split_by_col"]].nunique()
    if len(eval_loader.dataset) == 0 and n_split_units >= 2:
        raise RuntimeError(
            "eval 集样本数为 0，但划分单位>=2：请调整 data.train_ratio/val_ratio/test_ratio，或检查 split_by_col。"
        )

    if use_ddp:
        find_unused = bool(cfg["train"].get("ddp_find_unused_parameters", False))
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )

    output_dir = Path(cfg["data"]["output_root"]) / phase_output_dirname(cfg, args.phase)
    if rank == 0:
        print(f"[output] checkpoint & history -> {output_dir}")
        ensure_dir(output_dir)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    if int(args.bench_data) > 0:
        if rank == 0:
            print(
                f"[bench_data] phase={args.phase}  steps={args.bench_data}  "
                f"(data_ms: next batch + to_device; compute_ms: forward/backward/step, CUDA 已 synchronize)"
            )
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            phase=args.phase,
            rank=rank,
            bench_steps=int(args.bench_data),
        )
        cleanup_distributed(use_ddp)
        return

    best_eval = float("inf")
    history = {"train": [], "eval": []}

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        if use_ddp:
            sampler = train_loader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n===== Phase {args.phase} | Epoch {epoch}/{cfg['train']['epochs']} =====")

        train_log = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg, phase=args.phase, rank=rank
        )

        eval_model = unwrap_model(model)
        if rank == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if cfg["train"].get("eval_full_train_each_epoch", True):
                train_metrics = evaluate_loader(eval_model, train_loader, device, cfg, args.phase, rank=rank)
            else:
                train_metrics = train_metrics_stub_when_skip_full_train_eval(cfg, train_log)
                print(
                    "  [train] 已跳过全训练集 eval（train.eval_full_train_each_epoch=false），"
                    "仅记录 opt_loss；eval 仍为完整评估。"
                )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            eval_metrics = evaluate_loader(eval_model, eval_loader, device, cfg, args.phase, rank=rank)
        else:
            train_metrics = None
            eval_metrics = None

        # 非 rank0 先到达 barrier；rank0 跑完整集 eval 后再汇合，避免同步顺序死锁
        if use_ddp:
            dist.barrier()

        if rank == 0:
            history["train"].append({"epoch": epoch, "opt_loss": train_log["loss"], **train_metrics})
            history["eval"].append({"epoch": epoch, **eval_metrics})

            if args.metrics_json:
                print(
                    "train:",
                    json.dumps(
                        {"opt_loss": train_log["loss"], **train_metrics},
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
                print("eval:", json.dumps(eval_metrics, indent=2, ensure_ascii=False))
                print("  (完整指标仍写入 history.json)")
            else:
                print_epoch_metrics(train_log, train_metrics, eval_metrics)

            ckpt = {
                "epoch": epoch,
                "phase": args.phase,
                "model_state": eval_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
            }
            torch.save(ckpt, output_dir / "last.pt")

            if eval_metrics["loss"] < best_eval:
                best_eval = eval_metrics["loss"]
                torch.save(ckpt, output_dir / "best.pt")

            save_json(history, output_dir / "history.json")

        if use_ddp:
            dist.barrier()

    if rank == 0:
        print(f"\nTraining done. Best eval loss = {best_eval:.6f}")

    cleanup_distributed(use_ddp)


if __name__ == "__main__":
    main()
