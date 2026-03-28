from __future__ import annotations

import copy
import os
import yaml


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


DEFAULT_CONFIG = {
    "seed": 42,
    "data": {
        "labels_csv": "",
        "yuv_root": "",
        "cache_dir": "",
        "output_root": "",
        "sequence_col": "sequence",
        "poc_col": "poc",
        "qp_col": "qp",
        "bits_col": "bits",
        "psnr_col": "psnr",
        "mse_col": "mse",
        "group_cols": ["sequence"],
        "split_by_col": "sequence",
        "yuv_sequence_col": "sequence",
        "yuv_filename_template": "{sequence}.yuv",
        "width": 1280,
        "height": 720,
        "bit_depth": 8,
        "resize_width": 192,
        "resize_height": 108,
        "y_lowres_cache_suffix": ".ylow.npy",
        "y_lowres_cache_max_open_sequences": 2,
        "i_interval": 125,
        "gop_size": 16,
        "tail_hier_min": 4,
        "infer_refs_if_missing": True,
        "explicit_ref_columns": {
            "frame_type": None,
            "temporal_layer": None,
            "ref_poc_1": None,
            "ref_poc_2": None,
            "ref_qp_1": None,
            "ref_qp_2": None,
        },
        "intra_period_pos_col": None,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        # QP 输入与 pass1_delta_qp 分母：min-max 归一化区间 (x - min) / (max - min)
        "qp_norm_min": 30,
        "qp_norm_max": 255,
        # loss.mse_term=vmaf 时必填：CSV 中 VMAF 标签列名（如 pass2_vmaf）
        "vmaf_col": None,
        "pass1_columns": {
            "qp": "pass1_qp",
            "bits": "pass1_bits",
            "mse": "pass1_mse",
            "psnr": "pass1_psnr",
            # loss.mse_term=vmaf 时作为 pass1 向量第 3 维
            "vmaf": "pass1_vmaf",
        },
        # pass1 VMAF 输入归一化：pass1_vmaf / pass1_vmaf_norm_div（与 qp 归一化同量级）
        "pass1_vmaf_norm_div": 100.0,
        "phase1_tensor_cache_max_open_sequences": 4,
        "phase2_tensor_cache_max_open_sequences": 4,
    },
    "features": {
        "block_size": 8,
        "entropy_bins": 32,
        "edge_threshold": 0.08,
        "changed_threshold": 0.03,
        "pair_block_size": 8,
        # Phase 2/3：sidecar `<seq>.pair.npz`（由 preprocess_pair_cache 生成）
        "use_pair_cache": False,
        "pair_cache_suffix": ".pair.npz",
        "pair_cache_required": False,
        "pair_cache_fallback_online": True,
        "use_phase1_tensor_cache": False,
        "phase1_tensor_cache_required": False,
        "phase1_tensor_cache_suffix": ".phase1_{feature_profile}_{mse_term}.npz",
        "use_phase2_tensor_cache": False,
        "phase2_tensor_cache_required": False,
        "phase2_tensor_cache_suffix": ".phase2_{feature_profile}_{mse_term}.npz",
    },
    "train": {
        "num_workers": 4,
        "epochs": 40,
        "batch_size_phase1": 256,
        "batch_size_phase2": 256,
        "batch_size_phase3": 8,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-5,
        "huber_delta": 0.2,
        "grad_clip": 5.0,
        "amp": True,
        "device": "cuda",
        "save_every": 1,
        # 每 epoch 是否在训练集上跑完整 eval（与 val 一样扫全数据）；false 可明显省内存/时间，history 中 train 仅保留 opt_loss 占位
        "eval_full_train_each_epoch": True,
        "sequence_grouped_batches": False,
        # 多卡 DDP（torchrun）：batch_size_phase* 为每卡 batch；见 README 或 run_train_ddp.sh
        "ddp_find_unused_parameters": False,
    },
    "loss": {
        "bits_weight": 1.0,
        "mse_weight": 1.0,
        "aux_weight": 0.3,
        # 第二维失真项：log_mse（默认）| psnr | vmaf（直接回归 VMAF；模型该维输出即为 VMAF）
        "mse_term": "log_mse",
        # mse_term=psnr 时 Huber delta：未设 huber_delta_psnr 时为 train.huber_delta * huber_delta_psnr_scale（默认 5，约对应 dB 量级）
        "huber_delta_psnr": None,
        "huber_delta_psnr_scale": 5.0,
        # mse_term=vmaf 时：默认 train.huber_delta * huber_delta_vmaf_scale（约 0～100 分制）
        "huber_delta_vmaf": None,
        "huber_delta_vmaf_scale": 2.0,
    },
    "model": {
        # single：单头同时预测 bits + log(mse)；double：与 single 同架构但 head 仅 1 维，分两次训练专用于 bits 或失真
        "mode": "single",
        # mode=double 时必填：bits（_double_bits）| distortion（_double_psnr | _double_mse | _double_vmaf，由 loss.mse_term）
        "double_target": "bits",
        "phase2_variant": "flat",
        "self_hidden": 128,
        "edge_hidden": 128,
        "state_hidden": 64,
        "head_hidden": 128,
        "dropout": 0.1,
        "state_dim": 32,
    },
    "eval": {
        "max_psnr_value": 255.0,
    },
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f)
    return _deep_update(DEFAULT_CONFIG, user_cfg or {})


def is_double_bits_cfg(cfg: dict) -> bool:
    """mode=double 且 double_target=bits：仅训码率，pass1/target 不含失真先验。"""
    m = str(cfg.get("model", {}).get("mode", "single")).lower().strip()
    d = str(cfg.get("model", {}).get("double_target", "bits")).lower().strip()
    return m == "double" and d == "bits"


def normalize_mse_term(term: str) -> str:
    """将命令行/环境的失真项别名统一为 YAML 使用的 loss.mse_term。"""
    t = str(term).lower().strip()
    if t == "psnr":
        return "psnr"
    if t == "vmaf":
        return "vmaf"
    if t in ("log_mse", "logmse", "mse", ""):
        return "log_mse"
    raise ValueError(
        f'loss.mse_term 必须是 log_mse、mse、logmse、psnr 或 vmaf 之一，当前为 {term!r}'
    )


def resolve_train_override_cli_env(cli_val: str | None, env_key: str) -> tuple[str | None, str | None]:
    """命令行优先，否则读环境变量。返回 (值, 来源) 来源为 cli | env；未覆盖为 (None, None)。"""
    if cli_val is not None:
        s = str(cli_val).strip()
        if s != "":
            return s, "cli"
    v = os.environ.get(env_key)
    if v is not None and str(v).strip() != "":
        return str(v).strip(), "env"
    return None, None


# 训练/评估脚本共用：未传 CLI 时可由 shell export 循环注入
ENV_TRAIN_MODEL_MODE = "QP_TRAIN_MODEL_MODE"
ENV_TRAIN_DOUBLE_TARGET = "QP_TRAIN_DOUBLE_TARGET"
ENV_TRAIN_MSE_TERM = "QP_TRAIN_MSE_TERM"


def apply_train_overrides(
    cfg: dict,
    *,
    model_mode: str | None = None,
    double_target: str | None = None,
    mse_term: str | None = None,
    model_mode_src: str | None = None,
    double_target_src: str | None = None,
    mse_term_src: str | None = None,
) -> list[str]:
    """
    在 load_config 之后覆盖 model.mode / model.double_target / loss.mse_term。
    返回人类可读日志行（含来源 cli|env），便于确认与 YAML 的差异。
    """
    msgs: list[str] = []

    if model_mode is not None:
        m = str(model_mode).lower().strip()
        if m not in ("single", "double"):
            raise ValueError(f'model.mode 必须是 single 或 double，当前为 {model_mode!r}')
        cfg["model"]["mode"] = m
        msgs.append(f"model.mode={m} ({model_mode_src or 'cli'})")

    if double_target is not None:
        d = str(double_target).lower().strip()
        if d not in ("bits", "distortion"):
            raise ValueError(f'model.double_target 必须是 bits 或 distortion，当前为 {double_target!r}')
        cfg["model"]["double_target"] = d
        msgs.append(f"model.double_target={d} ({double_target_src or 'cli'})")

    if mse_term is not None:
        nt = normalize_mse_term(mse_term)
        cfg["loss"]["mse_term"] = nt
        msgs.append(f"loss.mse_term={nt} ({mse_term_src or 'cli'})")

    return msgs
