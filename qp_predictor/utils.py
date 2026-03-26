from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

TEMPORAL_LAYER_BUCKETS: tuple[int, ...] = (0, 1, 2, 3, 4, 6)


def qp_norm_span(data_cfg: dict) -> float:
    """QP 线性缩放到约 [0,1] 所用的区间长度 max(qp_norm_max - qp_norm_min, eps)。"""
    lo = float(data_cfg.get("qp_norm_min", 30))
    hi = float(data_cfg.get("qp_norm_max", 255))
    return max(hi - lo, 1e-6)


def normalize_qp(q: float | int, data_cfg: dict) -> float:
    """将 QP 按 data.qp_norm_min / qp_norm_max 做 min-max 归一化（默认 30～255）。"""
    lo = float(data_cfg.get("qp_norm_min", 30))
    return (float(q) - lo) / qp_norm_span(data_cfg)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def huber_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    mask = mask.float()
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    diff = pred - target
    abs_diff = diff.abs()
    delta_t = torch.tensor(delta, device=pred.device, dtype=pred.dtype)
    quadratic = torch.minimum(abs_diff, delta_t)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta_t * linear
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
    return (loss * mask).sum() / denom


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = np.mean(np.abs(y_true - y_pred) / denom)
    r2 = r2_score_np(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


def train_val_test_split(items: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    uniq = list(sorted(set(items)))
    rnd = random.Random(seed)
    rnd.shuffle(uniq)
    n = len(uniq)
    if n == 0:
        return [], [], []

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    n_train = max(0, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    n_test = n - n_train - n_val

    train_items = uniq[:n_train]
    val_items = uniq[n_train : n_train + n_val]
    test_items = uniq[n_train + n_val :]

    # round() 在序列数较少时常把 val/test 划成 0，导致验证集为空或指标无意义；在可能时补出非空划分
    if n >= 2 and len(val_items) == 0 and len(train_items) > 1:
        val_items = [train_items.pop()]
    if n >= 3 and len(test_items) == 0 and len(train_items) > 1:
        test_items = [train_items.pop()]

    return train_items, val_items, test_items


def to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(v, device) for v in batch)
    return batch


def frame_type_to_id(frame_type: str) -> int:
    mapping = {"I": 0, "P": 1, "B": 2, "UNK": 3}
    return mapping.get(str(frame_type).upper(), 3)


def frame_type_onehot(frame_type: str) -> np.ndarray:
    idx = frame_type_to_id(frame_type)
    arr = np.zeros(4, dtype=np.float32)
    arr[idx] = 1.0
    return arr


def temporal_layer_onehot(temporal_layer: int) -> np.ndarray:
    layer = int(temporal_layer)
    # 兼容旧模板里的 TL5，将其并入最高层桶。
    if layer == 5:
        layer = 6
    arr = np.zeros(len(TEMPORAL_LAYER_BUCKETS), dtype=np.float32)
    if layer in TEMPORAL_LAYER_BUCKETS:
        arr[TEMPORAL_LAYER_BUCKETS.index(layer)] = 1.0
    return arr


def compute_psnr_from_mse(mse, max_value: float = 255.0):
    mse = np.asarray(mse)
    mse = np.clip(mse, 1e-8, None)
    return 10.0 * np.log10((max_value ** 2) / mse)


def compute_psnr_from_mse_torch(mse: torch.Tensor, max_value: float = 255.0) -> torch.Tensor:
    """与 ``compute_psnr_from_mse`` 一致，供训练图中反传使用。"""
    mse = mse.clamp(min=1e-8)
    return 10.0 * torch.log10((max_value ** 2) / mse)


def inverse_log_bits(x):
    if isinstance(x, torch.Tensor):
        return torch.exp(x) - 1.0
    return np.exp(x) - 1.0


def inverse_log_mse(x, eps: float = 1e-6):
    if isinstance(x, torch.Tensor):
        return torch.exp(x) - eps
    return np.exp(x) - eps
