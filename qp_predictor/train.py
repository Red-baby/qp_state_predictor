from __future__ import annotations

import argparse
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

from .config import load_config
from .datasets import FrameDataset, SegmentDataset
from .features import pair_feature_names, pass1_feature_names, self_feature_names
from .manifest import build_manifest
from .models import Phase1Net, Phase2Net, Phase3Net
from .utils import (
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


def phase_output_dirname(cfg: dict, phase: int) -> str:
    """输出目录名：默认 phase{N}；开启 pass1 时为 phase{N}{output_phase_pass1_suffix}，关闭时可配 no_pass1 后缀。"""
    base = f"phase{phase}"
    data_cfg = cfg["data"]
    if data_cfg.get("use_pass1_features", False):
        suf = str(data_cfg.get("output_phase_pass1_suffix", "_pass1"))
        return f"{base}{suf}"
    suf = str(data_cfg.get("output_phase_no_pass1_suffix", "") or "")
    return f"{base}{suf}"


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
    val_df = manifest[manifest[split_by_col].isin(val_keys)].reset_index(drop=True)
    test_df = manifest[manifest[split_by_col].isin(test_keys)].reset_index(drop=True)
    return train_df, val_df, test_df


def build_model(cfg: dict, phase: int):
    self_dim = len(self_feature_names())
    pair_dim = len(pair_feature_names())
    meta_dim = 12
    pass1_dim = len(pass1_feature_names()) if cfg["data"].get("use_pass1_features", False) else 0

    if phase == 1:
        return Phase1Net(self_dim=self_dim, meta_dim=meta_dim, pass1_dim=pass1_dim, cfg=cfg)
    if phase == 2:
        return Phase2Net(self_dim=self_dim, pair_dim=pair_dim, meta_dim=meta_dim, pass1_dim=pass1_dim, cfg=cfg)
    if phase == 3:
        return Phase3Net(self_dim=self_dim, pair_dim=pair_dim, meta_dim=meta_dim, pass1_dim=pass1_dim, cfg=cfg)
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
    train_df, val_df, test_df = split_manifest(manifest, cfg)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    dl_kw = _dataloader_common_kwargs(train_cfg)

    if phase in (1, 2):
        train_ds = FrameDataset(manifest, cfg, train_df, phase=phase)
        val_ds = FrameDataset(manifest, cfg, val_df, phase=phase)
        test_ds = FrameDataset(manifest, cfg, test_df, phase=phase)
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
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            **dl_kw,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            **dl_kw,
        )
        return train_loader, val_loader, test_loader

    if phase == 3:
        train_ds = SegmentDataset(manifest, cfg, train_df)
        val_ds = SegmentDataset(manifest, cfg, val_df)
        test_ds = SegmentDataset(manifest, cfg, test_df)
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
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            **dl_kw,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            **dl_kw,
        )
        return train_loader, val_loader, test_loader

    raise ValueError(f"Unsupported phase: {phase}")


def compute_loss(batch, outputs, cfg, phase: int):
    delta = float(cfg["train"]["huber_delta"])
    bits_w = float(cfg["loss"]["bits_weight"])
    mse_w = float(cfg["loss"]["mse_weight"])
    aux_w = float(cfg["loss"]["aux_weight"])

    if phase in (1, 2):
        pred = outputs["pred"]
        target = batch["target"]
        mask = batch["valid_mask"]
        loss_bits = huber_loss_masked(pred[..., 0:1], target[..., 0:1], mask, delta)
        loss_mse = huber_loss_masked(pred[..., 1:2], target[..., 1:2], mask, delta)
        loss = bits_w * loss_bits + mse_w * loss_mse
        return loss, {
            "loss": float(loss.item()),
            "loss_bits": float(loss_bits.item()),
            "loss_mse": float(loss_mse.item()),
        }

    if phase == 3:
        pred = outputs["pred"]
        aux = outputs["aux_pred"]
        target = batch["targets"]
        mask = batch["valid_loss_mask"]

        loss_bits = huber_loss_masked(pred[..., 0:1], target[..., 0:1], mask, delta)
        loss_mse = huber_loss_masked(pred[..., 1:2], target[..., 1:2], mask, delta)
        loss_aux_bits = huber_loss_masked(aux[..., 0:1], target[..., 0:1], mask, delta)
        loss_aux_mse = huber_loss_masked(aux[..., 1:2], target[..., 1:2], mask, delta)

        loss_main = bits_w * loss_bits + mse_w * loss_mse
        loss_aux = bits_w * loss_aux_bits + mse_w * loss_aux_mse
        loss = loss_main + aux_w * loss_aux
        return loss, {
            "loss": float(loss.item()),
            "loss_bits": float(loss_bits.item()),
            "loss_mse": float(loss_mse.item()),
            "loss_aux_bits": float(loss_aux_bits.item()),
            "loss_aux_mse": float(loss_aux_mse.item()),
        }

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


@torch.no_grad()
def evaluate_loader(model, loader, device, cfg, phase: int, *, rank: int = 0):
    model.eval()
    loss_meter = []
    bits_true_all = []
    bits_pred_all = []
    mse_true_all = []
    mse_pred_all = []
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

        bits_pred = inverse_log_bits(pred[:, 0]).detach().cpu().numpy()
        bits_true = inverse_log_bits(target[:, 0]).detach().cpu().numpy()
        mse_pred = inverse_log_mse(pred[:, 1]).detach().cpu().numpy()
        mse_true = inverse_log_mse(target[:, 1]).detach().cpu().numpy()

        bits_true_all.append(bits_true)
        bits_pred_all.append(bits_pred)
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

    from .utils import compute_psnr_from_mse
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
        "loss": float(np.mean(loss_meter)) if loss_meter else 0.0,
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
            label_fn=lambda x: {0: "I", 1: "P", 2: "B", 3: "UNK"}.get(x, f"UNK_{x}"),
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
        bits_g = g.get("bits") or {}
        mse_g = g.get("mse") or {}
        psnr_g = g.get("psnr") or {}
        lines.append(
            "           [全局样本均值]  "
            f"bits: pred={_fmt_metric_val(bits_g.get('mean_pred', float('nan')))}  "
            f"true={_fmt_metric_val(bits_g.get('mean_true', float('nan')))}  "
            f"diff={_fmt_metric_val(bits_g.get('diff_pred_minus_true', float('nan')))}  |  "
            f"mse: pred={_fmt_metric_val(mse_g.get('mean_pred', float('nan')))}  "
            f"true={_fmt_metric_val(mse_g.get('mean_true', float('nan')))}  "
            f"diff={_fmt_metric_val(mse_g.get('diff_pred_minus_true', float('nan')))}  |  "
            f"psnr: pred={_fmt_metric_val(psnr_g.get('mean_pred', float('nan')))}  "
            f"true={_fmt_metric_val(psnr_g.get('mean_true', float('nan')))}  "
            f"diff={_fmt_metric_val(psnr_g.get('diff_pred_minus_true', float('nan')))}"
        )
    if m:
        bits_m = m.get("bits") or {}
        mse_m = m.get("mse") or {}
        psnr_m = m.get("psnr") or {}
        nseq = bits_m.get("num_sequences", 0)
        lines.append(
            "           [按序列宏平均] 每序列先算均值再对序列平均 diff=mean_s(mean_pred|s-mean_true|s)；"
            f" 序列数={nseq}"
        )
        lines.append(
            "                         "
            f"bits diff={_fmt_metric_val(bits_m.get('mean_diff_pred_minus_true', float('nan')))}  |  "
            f"mse diff={_fmt_metric_val(mse_m.get('mean_diff_pred_minus_true', float('nan')))}  |  "
            f"psnr diff={_fmt_metric_val(psnr_m.get('mean_diff_pred_minus_true', float('nan')))}"
        )
    return lines


def format_eval_metrics_block(name: str, metrics: dict) -> str:
    """将 evaluate_loader 返回的字典格式化为多行可读文本（含分组表）。"""
    lines: list[str] = []
    loss = metrics.get("loss", float("nan"))
    lines.append(f"  [{name}]  loss = {loss:.6f}")

    ma = metrics.get("mean_aggregate")
    if isinstance(ma, dict) and ma:
        lines.extend(_format_mean_aggregate_lines(ma))

    for key in ("bits", "mse", "psnr"):
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
        header = (
            f"           {'subset':<8} {'n':>7} | "
            f"{'bits_RMSE':>12} {'bits_R2':>8} | "
            f"{'mse_RMSE':>10} {'psnr_RMSE':>10}"
        )
        lines.append(header)
        lines.append("           " + "-" * 64)

        def sort_key(item):
            label, _ = item
            order = {"I": 0, "P": 1, "B": 2}
            if label in order:
                return (0, order[label])
            try:
                return (1, int(label))
            except ValueError:
                return (2, label)

        for label, row in sorted(sub.items(), key=sort_key):
            b = row.get("bits") or {}
            m = row.get("mse") or {}
            p = row.get("psnr") or {}
            n = int(row.get("count", 0))
            lines.append(
                f"           {label:<8} {n:>7} | "
                f"{_fmt_metric_val(b.get('rmse', float('nan'))):>12} {b.get('r2', float('nan')):>8.4f} | "
                f"{_fmt_metric_val(m.get('rmse', float('nan'))):>10} {_fmt_metric_val(p.get('rmse', float('nan'))):>10}"
            )

    return "\n".join(lines)


def print_epoch_metrics(train_log: dict, train_metrics: dict, val_metrics: dict) -> None:
    """终端打印：优化 loss + 训练集 eval（bits/mse/psnr）+ val。"""
    opt = train_log.get("loss", float("nan"))
    print(f"  [train]  opt_loss = {opt:.6f}")
    print(format_eval_metrics_block("train (eval)", train_metrics))
    print(format_eval_metrics_block("val", val_metrics))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
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
        help="每个 epoch 仍打印 train/val/test 的完整 JSON（调试）；默认使用紧凑表格。",
    )
    args = parser.parse_args()

    local_rank, world_size, use_ddp = setup_distributed()
    rank = dist.get_rank() if use_ddp else 0

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]) + rank)

    manifest = build_manifest(cfg)
    model = build_model(cfg, phase=args.phase)

    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader, test_loader = make_dataloaders(
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
    if len(val_loader.dataset) == 0 and n_split_units >= 2:
        raise RuntimeError(
            "验证集样本数为 0，但划分单位>=2：请调整 data.train_ratio/val_ratio/test_ratio，或检查 split_by_col。"
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

    best_val = float("inf")
    history = {"train": [], "val": [], "test_best": None}

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
            train_metrics = evaluate_loader(eval_model, train_loader, device, cfg, args.phase, rank=rank)
            val_metrics = evaluate_loader(eval_model, val_loader, device, cfg, args.phase, rank=rank)
        else:
            train_metrics = None
            val_metrics = None

        # 非 rank0 先到达 barrier；rank0 跑完整集 eval 后再汇合，避免同步顺序死锁
        if use_ddp:
            dist.barrier()

        if rank == 0:
            history["train"].append({"epoch": epoch, "opt_loss": train_log["loss"], **train_metrics})
            history["val"].append({"epoch": epoch, **val_metrics})

            if args.metrics_json:
                print(
                    "train:",
                    json.dumps(
                        {"opt_loss": train_log["loss"], **train_metrics},
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
                print("val:", json.dumps(val_metrics, indent=2, ensure_ascii=False))
                print("  (完整指标仍写入 history.json)")
            else:
                print_epoch_metrics(train_log, train_metrics, val_metrics)

            ckpt = {
                "epoch": epoch,
                "phase": args.phase,
                "model_state": eval_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
            }
            torch.save(ckpt, output_dir / "last.pt")

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(ckpt, output_dir / "best.pt")
                test_metrics = evaluate_loader(eval_model, test_loader, device, cfg, args.phase, rank=rank)
                history["test_best"] = {"epoch": epoch, **test_metrics}
                if args.metrics_json:
                    print("test(best):", json.dumps(test_metrics, indent=2, ensure_ascii=False))
                else:
                    print(format_eval_metrics_block("test (best val)", test_metrics))

            save_json(history, output_dir / "history.json")

        if use_ddp:
            dist.barrier()

    if rank == 0:
        print(f"\nTraining done. Best val loss = {best_val:.6f}")

    cleanup_distributed(use_ddp)


if __name__ == "__main__":
    main()
