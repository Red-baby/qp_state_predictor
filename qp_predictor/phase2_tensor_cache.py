from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .config import is_double_bits_cfg, normalize_mse_term
from .features import meta_feature_names, normalize_feature_profile, pair_feature_names, pass1_feature_names, self_feature_names


def phase2_tensor_cache_loss_tag(cfg: dict) -> str:
    """文件名第二段：double+bits 固定为 bits（与 phase2_vmaf_vmaf 对称）；否则为归一化后的 loss.mse_term。"""
    if is_double_bits_cfg(cfg):
        return "bits"
    return normalize_mse_term(str(cfg.get("loss", {}).get("mse_term", "log_mse")))


def phase2_tensor_cache_filename(cfg: dict, sequence: str, feature_profile: str) -> str:
    suffix_tmpl = str(
        cfg.get("features", {}).get("phase2_tensor_cache_suffix", ".phase2_{feature_profile}_{mse_term}.npz")
    )
    profile = normalize_feature_profile(feature_profile)
    term = phase2_tensor_cache_loss_tag(cfg)
    suffix = suffix_tmpl.format(feature_profile=profile, mse_term=term)
    return f"{sequence}{suffix}"


class Phase2TensorCacheManager:
    """加载 Phase 2 预打包张量缓存，并用 LRU 限制 worker 常驻内存。"""

    _REQUIRED_KEYS = {
        "present_mask",
        "self_feats",
        "meta_feats",
        "qp",
        "target",
        "temporal_layer",
        "valid_mask",
        "pass1_feats",
        "ref_feats",
        "pair_feats",
        "ref_qps",
        "ref_valid_mask",
        "ref_pass1_feats",
        "meta_dim",
        "self_dim",
        "pair_dim",
        "pass1_dim",
    }

    def __init__(self, cfg: dict, feature_profile: str):
        self.cfg = cfg
        self.cache_dir = Path(cfg["data"]["cache_dir"])
        self.feature_profile = normalize_feature_profile(feature_profile)
        self.cache_loss_tag = phase2_tensor_cache_loss_tag(cfg)
        self.required = bool(cfg.get("features", {}).get("phase2_tensor_cache_required", False))
        self.max_open = max(1, int(cfg.get("data", {}).get("phase2_tensor_cache_max_open_sequences", 4)))
        self.self_dim = len(self_feature_names(self.feature_profile))
        self.meta_dim = len(meta_feature_names(self.feature_profile))
        self.pair_dim = len(pair_feature_names(self.feature_profile))
        self.pass1_dim = len(pass1_feature_names(cfg))
        self._stores: OrderedDict[str, dict[str, Any] | None] = OrderedDict()

    def _evict_if_needed(self) -> None:
        while len(self._stores) >= self.max_open:
            self._stores.popitem(last=False)

    def _validate(self, payload: dict[str, Any], path: Path) -> None:
        missing = self._REQUIRED_KEYS - set(payload.keys())
        if missing:
            raise KeyError(f"{path} missing keys: {sorted(missing)}")
        if int(np.asarray(payload["self_dim"]).item()) != self.self_dim:
            raise ValueError(f"{path} self_dim mismatch")
        if int(np.asarray(payload["meta_dim"]).item()) != self.meta_dim:
            raise ValueError(f"{path} meta_dim mismatch")
        if int(np.asarray(payload["pair_dim"]).item()) != self.pair_dim:
            raise ValueError(f"{path} pair_dim mismatch")
        if int(np.asarray(payload["pass1_dim"]).item()) != self.pass1_dim:
            raise ValueError(f"{path} pass1_dim mismatch")

    def load(self, sequence: str) -> dict[str, Any] | None:
        seq = str(sequence)
        if seq in self._stores:
            self._stores.move_to_end(seq)
            return self._stores[seq]

        path = self.cache_dir / phase2_tensor_cache_filename(self.cfg, seq, self.feature_profile)
        if not path.is_file():
            if self.required:
                raise FileNotFoundError(f"phase2 tensor cache required but missing: {path}")
            self._evict_if_needed()
            self._stores[seq] = None
            return None

        data = np.load(path, allow_pickle=False)
        try:
            payload = {k: np.asarray(data[k]) for k in data.files}
        finally:
            data.close()
        self._validate(payload, path)
        self._evict_if_needed()
        self._stores[seq] = payload
        return payload
