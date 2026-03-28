from __future__ import annotations

import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import (
    extract_pair_features,
    extract_self_features,
    meta_feature_names,
    pair_feature_names,
    pass1_feature_names,
    resolve_feature_profile,
    resolve_phase3_pair_profile,
    self_feature_storage_key,
)
from .config import is_double_bits_cfg
from .graph import build_segment_topo_order
from .pair_cache import PairCacheManager
from .phase1_tensor_cache import Phase1TensorCacheManager
from .phase2_tensor_cache import Phase2TensorCacheManager
from .utils import normalize_qp, temporal_layer_onehot

_PAIR_FALLBACK_WARNED = False
_SELF_FALLBACK_WARNED: set[str] = set()
_PHASE2_TENSOR_FALLBACK_WARNED = False
_PHASE1_TENSOR_FALLBACK_WARNED = False


def _warn_pair_fallback_once() -> None:
    global _PAIR_FALLBACK_WARNED
    if not _PAIR_FALLBACK_WARNED:
        warnings.warn(
            "pair cache 未命中，回退在线 extract_pair_features（较慢）。"
            " 请运行 python -m qp_predictor.preprocess_pair_cache --config ...",
            UserWarning,
            stacklevel=2,
        )
        _PAIR_FALLBACK_WARNED = True


def _warn_self_fallback_once(profile: str) -> None:
    if profile not in _SELF_FALLBACK_WARNED:
        warnings.warn(
            f"cache 缺少 {self_feature_storage_key(profile)}，回退在线 extract_self_features（较慢）。"
            " 请重新运行 python -m qp_predictor.preprocess_cache --config ...",
            UserWarning,
            stacklevel=2,
        )
        _SELF_FALLBACK_WARNED.add(profile)


def _warn_phase2_tensor_fallback_once() -> None:
    global _PHASE2_TENSOR_FALLBACK_WARNED
    if not _PHASE2_TENSOR_FALLBACK_WARNED:
        warnings.warn(
            "phase2 tensor cache 未命中，回退普通逐样本取数。"
            " 可运行 python -m qp_predictor.preprocess_phase2_tensor_cache --config ... 预打包缓存。",
            UserWarning,
            stacklevel=2,
        )
        _PHASE2_TENSOR_FALLBACK_WARNED = True


def _warn_phase1_tensor_fallback_once() -> None:
    global _PHASE1_TENSOR_FALLBACK_WARNED
    if not _PHASE1_TENSOR_FALLBACK_WARNED:
        warnings.warn(
            "phase1 tensor cache 未命中，回退 mmap 逐样本取数。"
            " 可运行 python -m qp_predictor.preprocess_phase1_tensor_cache --config ... 预打包缓存。",
            UserWarning,
            stacklevel=2,
        )
        _PHASE1_TENSOR_FALLBACK_WARNED = True


class CacheManager:
    """按序列 mmap 打开主 cache；多进程 DataLoader 下无界 dict 会撑爆 worker RAM，故用 LRU。"""

    def __init__(self, cache_dir: str, max_open_sequences: int = 8):
        self.cache_dir = Path(cache_dir)
        self._max = max(1, int(max_open_sequences))
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def load(self, sequence: str):
        if sequence in self._cache:
            self._cache.move_to_end(sequence)
            return self._cache[sequence]
        path = self.cache_dir / f"{sequence}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")
        while len(self._cache) >= self._max:
            _k, old = self._cache.popitem(last=False)
            if hasattr(old, "close"):
                old.close()
        data = np.load(path, allow_pickle=False, mmap_mode="r")
        self._cache[sequence] = data
        return data


class YLowresManager:
    """按 sequence mmap 打开拆分后的 y_lowres `.npy`，仅在回退路径使用。"""

    def __init__(self, cache_dir: str, suffix: str = ".ylow.npy", max_open_sequences: int = 2):
        self.cache_dir = Path(cache_dir)
        self.suffix = str(suffix)
        self._max = max(1, int(max_open_sequences))
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def load(self, sequence: str):
        if sequence in self._cache:
            self._cache.move_to_end(sequence)
            return self._cache[sequence]
        path = self.cache_dir / f"{sequence}{self.suffix}"
        if not path.exists():
            raise FileNotFoundError(f"y_lowres cache file not found: {path}")
        while len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        data = np.load(path, allow_pickle=False, mmap_mode="r")
        self._cache[sequence] = data
        return data


def _cache_manager_from_cfg(data_cfg: dict) -> CacheManager:
    return CacheManager(
        data_cfg["cache_dir"],
        max_open_sequences=int(data_cfg.get("cache_max_open_sequences", 8)),
    )


def _y_lowres_manager_from_cfg(data_cfg: dict) -> YLowresManager:
    return YLowresManager(
        data_cfg["cache_dir"],
        suffix=str(data_cfg.get("y_lowres_cache_suffix", ".ylow.npy")),
        max_open_sequences=int(data_cfg.get("y_lowres_cache_max_open_sequences", 2)),
    )


def build_meta_vector(row: pd.Series, i_interval: int, feature_profile: str = "legacy") -> np.ndarray:
    segment_span = int(row["segment_span"]) if "segment_span" in row and int(row["segment_span"]) > 0 else int(i_interval)
    tl_oh = temporal_layer_onehot(int(row["temporal_layer"]))
    if "intra_period_pos" in row and pd.notna(row["intra_period_pos"]):
        intra_period_pos = float(row["intra_period_pos"])
    else:
        intra_period_pos = float(row["local_poc"]) / max(segment_span - 1, 1)
    intra_period_pos = float(np.clip(intra_period_pos, 0.0, 1.0))
    ref_d1 = float(row["ref_distance_1"]) / max(segment_span, 1) if int(row["ref_distance_1"]) >= 0 else -1.0
    ref_d2 = float(row["ref_distance_2"]) / max(segment_span, 1) if int(row["ref_distance_2"]) >= 0 else -1.0
    _ = str(feature_profile).lower().strip()
    vec = np.concatenate([
        tl_oh,
        np.asarray([
            intra_period_pos,
            ref_d1,
            ref_d2,
        ], dtype=np.float32),
    ], axis=0)
    return vec.astype(np.float32)


def build_pass1_vector(row: pd.Series, cfg: dict) -> np.ndarray:
    """从 manifest 行构建 pass1 先验：double+bits 为 3 维（无失真）；vmaf 为 4 维含 pass1_vmaf；否则含 pass1_log_mse。"""
    data_cfg = cfg["data"]
    if is_double_bits_cfg(cfg):
        qp_n = normalize_qp(float(row["pass1_qp"]), data_cfg)
        log_bits = float(row["pass1_log_bits"])
        dqp = float(row["pass1_delta_qp"])
        return np.asarray([qp_n, log_bits, dqp], dtype=np.float32)
    mse_term = str(cfg.get("loss", {}).get("mse_term", "log_mse")).lower().strip()
    qp_n = normalize_qp(float(row["pass1_qp"]), data_cfg)
    log_bits = float(row["pass1_log_bits"])
    dqp = float(row["pass1_delta_qp"])
    if mse_term == "vmaf":
        div = float(data_cfg.get("pass1_vmaf_norm_div", 100.0))
        v = float(row["pass1_vmaf"]) / max(div, 1e-6)
        return np.asarray([qp_n, log_bits, v, dqp], dtype=np.float32)
    return np.asarray([qp_n, log_bits, float(row["pass1_log_mse"]), dqp], dtype=np.float32)


def _build_meta_vector_fast(
    *,
    temporal_layer: int,
    intra_period_pos: float,
    ref_distance_1: int,
    ref_distance_2: int,
    segment_span: int,
) -> np.ndarray:
    tl_oh = temporal_layer_onehot(int(temporal_layer))
    span = max(int(segment_span), 1)
    intra = float(np.clip(float(intra_period_pos), 0.0, 1.0))
    ref_d1 = float(ref_distance_1) / span if int(ref_distance_1) >= 0 else -1.0
    ref_d2 = float(ref_distance_2) / span if int(ref_distance_2) >= 0 else -1.0
    return np.concatenate([tl_oh, np.asarray([intra, ref_d1, ref_d2], dtype=np.float32)], axis=0).astype(np.float32)


def _build_pass1_vector_fast(
    *,
    pass1_qp: float,
    pass1_log_bits: float,
    pass1_delta_qp: float,
    pass1_vmaf: float,
    pass1_log_mse: float,
    cfg: dict,
) -> np.ndarray:
    data_cfg = cfg["data"]
    if is_double_bits_cfg(cfg):
        qp_n = normalize_qp(float(pass1_qp), data_cfg)
        log_bits = float(pass1_log_bits)
        dqp = float(pass1_delta_qp)
        return np.asarray([qp_n, log_bits, dqp], dtype=np.float32)
    mse_term = str(cfg.get("loss", {}).get("mse_term", "log_mse")).lower().strip()
    qp_n = normalize_qp(float(pass1_qp), data_cfg)
    log_bits = float(pass1_log_bits)
    dqp = float(pass1_delta_qp)
    if mse_term == "vmaf":
        div = float(data_cfg.get("pass1_vmaf_norm_div", 100.0))
        v = float(pass1_vmaf) / max(div, 1e-6)
        return np.asarray([qp_n, log_bits, v, dqp], dtype=np.float32)
    return np.asarray([qp_n, log_bits, float(pass1_log_mse), dqp], dtype=np.float32)


class FrameDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: dict, split_df: pd.DataFrame, phase: int):
        self.cfg = cfg
        self.phase = int(phase)
        self.feature_profile = resolve_feature_profile(cfg, self.phase)
        self.i_interval = int(cfg["data"]["i_interval"])
        self.self_block_size = int(cfg["features"]["block_size"])
        self.entropy_bins = int(cfg["features"]["entropy_bins"])
        self.edge_threshold = float(cfg["features"]["edge_threshold"])
        self.block_size = int(cfg["features"]["pair_block_size"])
        self.changed_threshold = float(cfg["features"]["changed_threshold"])
        self.cache_manager = _cache_manager_from_cfg(cfg["data"])
        self.y_lowres_manager = _y_lowres_manager_from_cfg(cfg["data"])
        if self.phase not in (1, 2):
            raise ValueError("FrameDataset only supports phase 1/2")

        df = split_df.copy()
        df = df[df["valid_train"] == 1].reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self._self_key = self_feature_storage_key(self.feature_profile)
        self._pair_dim = len(pair_feature_names(self.feature_profile))
        self._zero_ref_qp = np.zeros((1,), dtype=np.float32)
        self._zero_ref_valid = np.zeros((), dtype=np.float32)
        self._mse_term = str(self.cfg["loss"].get("mse_term", "log_mse")).lower().strip()

        # 热路径改为 NumPy 数组索引，避免每个样本都构造 pandas.Series。
        self._yuv_sequence_arr = self.df["yuv_sequence"].astype(str).to_numpy(dtype=object)
        self._sequence_arr = self.df["sequence"].astype(str).to_numpy(dtype=object)
        self._segment_uid_arr = self.df["segment_uid"].astype(str).to_numpy(dtype=object)
        self._poc_arr = self.df["poc"].to_numpy(dtype=np.int64, copy=True)
        self._qp_arr = self.df["qp"].to_numpy(dtype=np.float32, copy=True)
        self._bits_arr = self.df["bits"].to_numpy(dtype=np.float32, copy=True)
        self._mse_arr = self.df["mse"].to_numpy(dtype=np.float32, copy=True)
        self._temporal_layer_arr = self.df["temporal_layer"].to_numpy(dtype=np.int64, copy=True)
        self._valid_train_arr = self.df["valid_train"].to_numpy(dtype=np.float32, copy=True)
        self._segment_span_arr = self.df["segment_span"].to_numpy(dtype=np.int64, copy=True)
        self._intra_period_pos_arr = self.df["intra_period_pos"].to_numpy(dtype=np.float32, copy=True)
        self._ref_distance_1_arr = self.df["ref_distance_1"].to_numpy(dtype=np.int64, copy=True)
        self._ref_distance_2_arr = self.df["ref_distance_2"].to_numpy(dtype=np.int64, copy=True)
        self._ref_poc_1_arr = self.df["ref_poc_1"].to_numpy(dtype=np.int64, copy=True)
        self._ref_poc_2_arr = self.df["ref_poc_2"].to_numpy(dtype=np.int64, copy=True)
        self._ref_qp_1_arr = self.df["ref_qp_1"].to_numpy(dtype=np.float32, copy=True)
        self._ref_qp_2_arr = self.df["ref_qp_2"].to_numpy(dtype=np.float32, copy=True)
        self._pass1_qp_arr = self.df["pass1_qp"].to_numpy(dtype=np.float32, copy=True)
        self._pass1_log_bits_arr = self.df["pass1_log_bits"].to_numpy(dtype=np.float32, copy=True)
        self._pass1_delta_qp_arr = self.df["pass1_delta_qp"].to_numpy(dtype=np.float32, copy=True)
        self._pass1_log_mse_arr = self.df["pass1_log_mse"].to_numpy(dtype=np.float32, copy=True)
        if "pass1_vmaf" in self.df.columns:
            self._pass1_vmaf_arr = self.df["pass1_vmaf"].to_numpy(dtype=np.float32, copy=True)
        else:
            self._pass1_vmaf_arr = np.zeros(len(self.df), dtype=np.float32)
        if "vmaf" in self.df.columns:
            self._vmaf_arr = self.df["vmaf"].to_numpy(dtype=np.float32, copy=True)
        else:
            self._vmaf_arr = np.zeros(len(self.df), dtype=np.float32)

        self._qp_lookup = {}
        for seg_uid, poc, qp in zip(self._segment_uid_arr, self._poc_arr, self._qp_arr):
            self._qp_lookup[(str(seg_uid), int(poc))] = float(qp)

        self._pass1_dim = len(pass1_feature_names(cfg))

        self._row_lookup = {}
        if self.phase == 2:
            for row_idx, (seg_uid, poc) in enumerate(zip(self._segment_uid_arr, self._poc_arr)):
                self._row_lookup[(str(seg_uid), int(poc))] = row_idx

        feat_cfg = cfg["features"]
        self._use_pair_cache = self.phase == 2 and bool(feat_cfg.get("use_pair_cache", False))
        self._pair_fallback = bool(feat_cfg.get("pair_cache_fallback_online", True))
        self._pair_cache = PairCacheManager(cfg, feature_profile=self.feature_profile) if self._use_pair_cache else None
        self._use_phase1_tensor_cache = self.phase == 1 and bool(feat_cfg.get("use_phase1_tensor_cache", False))
        self._phase1_tensor_cache = (
            Phase1TensorCacheManager(cfg, self.feature_profile) if self._use_phase1_tensor_cache else None
        )
        self._use_phase2_tensor_cache = self.phase == 2 and bool(feat_cfg.get("use_phase2_tensor_cache", False))
        self._phase2_tensor_cache = (
            Phase2TensorCacheManager(cfg, self.feature_profile) if self._use_phase2_tensor_cache else None
        )

    def __len__(self):
        return len(self.df)

    def _get_cache_feats(self, sequence: str, poc: int, *, need_y_low: bool):
        """need_y_low=False 时只读 self_features；回退时才读拆分后的 y_lowres。"""
        cache = self.cache_manager.load(sequence)
        y_low = None
        if self._self_key in cache:
            self_feats = cache[self._self_key][poc].astype(np.float32)
            if not need_y_low:
                return self_feats, None
            y_low = self.y_lowres_manager.load(sequence)[poc].astype(np.uint8)
            return self_feats, y_low

        y_low = self.y_lowres_manager.load(sequence)[poc].astype(np.uint8)
        _warn_self_fallback_once(self.feature_profile)
        self_feats = extract_self_features(
            y_low,
            block_size=self.self_block_size,
            entropy_bins=self.entropy_bins,
            edge_threshold=self.edge_threshold,
            feature_profile=self.feature_profile,
        ).astype(np.float32)
        if not need_y_low:
            return self_feats, None
        return self_feats, y_low

    def _phase2_need_cur_y_low(self, seq: str, poc: int, ref_poc_1: int, ref_poc_2: int) -> bool:
        """若 pair 全在 sidecar 命中则无需读当前帧 y_lowres。"""
        if not self._pair_cache:
            return True
        for rp in (int(ref_poc_1), int(ref_poc_2)):
            if rp < 0:
                continue
            hit = self._pair_cache.get_pair_feats(seq, poc, rp) is not None
            if not hit:
                if self._pair_fallback:
                    return True
                raise RuntimeError(
                    f"pair cache 缺少边且 pair_cache_fallback_online=false: sequence={seq} cur={poc} ref={rp}"
                )
        return False

    def _phase1_tensor_cached_item(self, idx: int, seq: str, poc: int):
        if self._phase1_tensor_cache is None:
            return None
        payload = self._phase1_tensor_cache.load(seq)
        if payload is None:
            _warn_phase1_tensor_fallback_once()
            return None
        if poc >= int(payload["present_mask"].shape[0]) or not bool(payload["present_mask"][poc]):
            return None
        return {
            "self_feats": torch.from_numpy(payload["self_feats"][poc]),
            "meta_feats": torch.from_numpy(payload["meta_feats"][poc]),
            "qp": torch.from_numpy(payload["qp"][poc]),
            "target": torch.from_numpy(payload["target"][poc]),
            "temporal_layer": torch.tensor(int(payload["temporal_layer"][poc]), dtype=torch.long),
            "valid_mask": torch.tensor(float(payload["valid_mask"][poc]), dtype=torch.float32),
            "sequence": str(self._sequence_arr[idx]),
            "pass1_feats": torch.from_numpy(payload["pass1_feats"][poc]),
        }

    def _phase2_tensor_cached_item(self, idx: int, seq: str, poc: int):
        if self._phase2_tensor_cache is None:
            return None
        payload = self._phase2_tensor_cache.load(seq)
        if payload is None:
            _warn_phase2_tensor_fallback_once()
            return None
        if poc >= int(payload["present_mask"].shape[0]) or not bool(payload["present_mask"][poc]):
            return None
        out = {
            "self_feats": torch.from_numpy(payload["self_feats"][poc]),
            "meta_feats": torch.from_numpy(payload["meta_feats"][poc]),
            "qp": torch.from_numpy(payload["qp"][poc]),
            "target": torch.from_numpy(payload["target"][poc]),
            "temporal_layer": torch.tensor(int(payload["temporal_layer"][poc]), dtype=torch.long),
            "valid_mask": torch.tensor(float(payload["valid_mask"][poc]), dtype=torch.float32),
            "sequence": str(self._sequence_arr[idx]),
            "pass1_feats": torch.from_numpy(payload["pass1_feats"][poc]),
            "ref_feats": torch.from_numpy(payload["ref_feats"][poc]),
            "pair_feats": torch.from_numpy(payload["pair_feats"][poc]),
            "ref_qps": torch.from_numpy(payload["ref_qps"][poc]),
            "ref_valid_mask": torch.from_numpy(payload["ref_valid_mask"][poc]),
            "ref_pass1_feats": torch.from_numpy(payload["ref_pass1_feats"][poc]),
        }
        if is_double_bits_cfg(self.cfg):
            if self._mse_term == "vmaf":
                aux_dist_target = np.asarray([float(self._vmaf_arr[idx])], dtype=np.float32)
            else:
                aux_dist_target = np.asarray([np.log(float(self._mse_arr[idx]) + 1e-6)], dtype=np.float32)
            out["aux_dist_target"] = torch.from_numpy(aux_dist_target)
        return out

    def __getitem__(self, idx: int):
        seq = str(self._yuv_sequence_arr[idx])
        poc = int(self._poc_arr[idx])
        ref_poc_1 = int(self._ref_poc_1_arr[idx])
        ref_poc_2 = int(self._ref_poc_2_arr[idx])
        if self.phase == 1:
            packed = self._phase1_tensor_cached_item(idx, seq, poc)
            if packed is not None:
                return packed
        if self.phase == 2:
            packed = self._phase2_tensor_cached_item(idx, seq, poc)
            if packed is not None:
                return packed
        if self.phase == 2:
            need_y = self._phase2_need_cur_y_low(seq, poc, ref_poc_1, ref_poc_2)
        else:
            need_y = False
        self_feats, y_low = self._get_cache_feats(seq, poc, need_y_low=need_y)
        meta = _build_meta_vector_fast(
            temporal_layer=int(self._temporal_layer_arr[idx]),
            intra_period_pos=float(self._intra_period_pos_arr[idx]),
            ref_distance_1=int(self._ref_distance_1_arr[idx]),
            ref_distance_2=int(self._ref_distance_2_arr[idx]),
            segment_span=int(self._segment_span_arr[idx]),
        )
        qp = np.asarray([normalize_qp(float(self._qp_arr[idx]), self.cfg["data"])], dtype=np.float32)

        target_bits = np.log1p(float(self._bits_arr[idx]))
        if is_double_bits_cfg(self.cfg):
            target = np.asarray([target_bits], dtype=np.float32)
        elif self._mse_term == "vmaf":
            target_mse = float(self._vmaf_arr[idx])
            target = np.asarray([target_bits, target_mse], dtype=np.float32)
        else:
            target_mse = np.log(float(self._mse_arr[idx]) + 1e-6)
            target = np.asarray([target_bits, target_mse], dtype=np.float32)

        out = {
            "self_feats": torch.from_numpy(self_feats),
            "meta_feats": torch.from_numpy(meta),
            "qp": torch.from_numpy(qp),
            "target": torch.from_numpy(target),
            "temporal_layer": torch.tensor(int(self._temporal_layer_arr[idx]), dtype=torch.long),
            "valid_mask": torch.tensor(float(self._valid_train_arr[idx]), dtype=torch.float32),
            "sequence": str(self._sequence_arr[idx]),
        }
        if self.phase == 2 and is_double_bits_cfg(self.cfg):
            if self._mse_term == "vmaf":
                aux_dist_target = np.asarray([float(self._vmaf_arr[idx])], dtype=np.float32)
            else:
                aux_dist_target = np.asarray([np.log(float(self._mse_arr[idx]) + 1e-6)], dtype=np.float32)
            out["aux_dist_target"] = torch.from_numpy(aux_dist_target)

        out["pass1_feats"] = torch.from_numpy(
            _build_pass1_vector_fast(
                pass1_qp=float(self._pass1_qp_arr[idx]),
                pass1_log_bits=float(self._pass1_log_bits_arr[idx]),
                pass1_delta_qp=float(self._pass1_delta_qp_arr[idx]),
                pass1_vmaf=float(self._pass1_vmaf_arr[idx]),
                pass1_log_mse=float(self._pass1_log_mse_arr[idx]),
                cfg=self.cfg,
            )
        )

        if self.phase == 2:
            pair_feats_all = []
            ref_feats_all = []
            ref_qps_all = []
            ref_valid_all = []
            ref_pass1_all = []
            seg_uid = str(self._segment_uid_arr[idx])

            for ref_poc, ref_qp_raw in ((ref_poc_1, float(self._ref_qp_1_arr[idx])), (ref_poc_2, float(self._ref_qp_2_arr[idx]))):
                if ref_poc >= 0:
                    pair_feats: np.ndarray | None = None
                    if self._pair_cache:
                        pair_feats = self._pair_cache.get_pair_feats(seq, poc, ref_poc)
                    if pair_feats is not None:
                        ref_self_feats, _ = self._get_cache_feats(seq, ref_poc, need_y_low=False)
                    elif self._pair_fallback:
                        _warn_pair_fallback_once()
                        if y_low is None:
                            _, y_low = self._get_cache_feats(seq, poc, need_y_low=True)
                        ref_self_feats, ref_y = self._get_cache_feats(seq, ref_poc, need_y_low=True)
                        pair_feats = extract_pair_features(
                            y_low,
                            ref_y,
                            block_size=self.block_size,
                            changed_threshold=self.changed_threshold,
                            feature_profile=self.feature_profile,
                        )
                    else:
                        raise RuntimeError(
                            f"pair cache 未命中且不允许回退: {seq} cur={poc} ref={ref_poc}"
                        )
                    if np.isfinite(ref_qp_raw):
                        ref_qp = ref_qp_raw
                    else:
                        ref_qp = self._qp_lookup.get((seg_uid, ref_poc), float(self._qp_arr[idx]))
                    ref_valid = 1.0

                    ref_row_idx = self._row_lookup.get((seg_uid, ref_poc))
                    ref_p1 = (
                        _build_pass1_vector_fast(
                            pass1_qp=float(self._pass1_qp_arr[ref_row_idx]),
                            pass1_log_bits=float(self._pass1_log_bits_arr[ref_row_idx]),
                            pass1_delta_qp=float(self._pass1_delta_qp_arr[ref_row_idx]),
                            pass1_vmaf=float(self._pass1_vmaf_arr[ref_row_idx]),
                            pass1_log_mse=float(self._pass1_log_mse_arr[ref_row_idx]),
                            cfg=self.cfg,
                        )
                        if ref_row_idx is not None
                        else np.zeros(self._pass1_dim, dtype=np.float32)
                    )
                else:
                    ref_self_feats = np.zeros_like(self_feats, dtype=np.float32)
                    pair_feats = np.zeros(self._pair_dim, dtype=np.float32)
                    ref_qp = 0.0
                    ref_valid = 0.0
                    ref_p1 = np.zeros(self._pass1_dim, dtype=np.float32)

                pair_feats_all.append(pair_feats.astype(np.float32))
                ref_feats_all.append(ref_self_feats.astype(np.float32))
                if ref_poc >= 0:
                    ref_qps_all.append([normalize_qp(ref_qp, self.cfg["data"])])
                else:
                    ref_qps_all.append([0.0])
                ref_valid_all.append(ref_valid)
                ref_pass1_all.append(ref_p1)

            out.update({
                "ref_feats": torch.from_numpy(np.stack(ref_feats_all, axis=0)),
                "pair_feats": torch.from_numpy(np.stack(pair_feats_all, axis=0)),
                "ref_qps": torch.from_numpy(np.asarray(ref_qps_all, dtype=np.float32)),
                "ref_valid_mask": torch.from_numpy(np.asarray(ref_valid_all, dtype=np.float32)),
            })
            out["ref_pass1_feats"] = torch.from_numpy(np.stack(ref_pass1_all, axis=0))

        return out


class SegmentDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: dict, split_df: pd.DataFrame):
        self.cfg = cfg
        self.i_interval = int(cfg["data"]["i_interval"])
        self.block_size = int(cfg["features"]["pair_block_size"])
        self.changed_threshold = float(cfg["features"]["changed_threshold"])
        self.cache_manager = _cache_manager_from_cfg(cfg["data"])
        self.y_lowres_manager = _y_lowres_manager_from_cfg(cfg["data"])
        self._pair_feature_profile = resolve_phase3_pair_profile(cfg)
        self._pair_dim = len(pair_feature_names(self._pair_feature_profile))

        self._pass1_dim = len(pass1_feature_names(cfg))

        feat_cfg = cfg["features"]
        self._use_pair_cache = bool(feat_cfg.get("use_pair_cache", False))
        self._pair_fallback = bool(feat_cfg.get("pair_cache_fallback_online", True))
        self._pair_cache = (
            PairCacheManager(cfg, feature_profile=self._pair_feature_profile) if self._use_pair_cache else None
        )

        seg_groups = []
        for seg_uid, g in split_df.groupby("segment_uid"):
            local_pocs = set(g["local_poc"].tolist())
            if 0 not in local_pocs:
                continue
            seg_groups.append(g.sort_values("local_poc").reset_index(drop=True))
        self.groups = seg_groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx: int):
        g = self.groups[idx]
        seq = str(g.iloc[0]["yuv_sequence"])
        T = self.i_interval
        topo_order = np.asarray(build_segment_topo_order(g, i_interval=T), dtype=np.int64)

        cache = self.cache_manager.load(seq)
        feat_dim = cache["self_features"].shape[1]
        meta_dim = len(meta_feature_names())

        self_feats = np.zeros((T, feat_dim), dtype=np.float32)
        meta_feats = np.zeros((T, meta_dim), dtype=np.float32)
        qps = np.zeros((T, 1), dtype=np.float32)
        targets = np.zeros((T, 2), dtype=np.float32)
        valid_loss_mask = np.zeros((T,), dtype=np.float32)
        ref_idx = -np.ones((T, 2), dtype=np.int64)
        pair_feats = np.zeros((T, 2, self._pair_dim), dtype=np.float32)
        temporal_layers = -np.ones((T,), dtype=np.int64)
        pass1_feats = np.zeros((T, self._pass1_dim), dtype=np.float32)

        row_by_local = {int(r["local_poc"]): r for _, r in g.iterrows()}

        for t in range(T):
            if t not in row_by_local:
                continue
            row = row_by_local[t]
            poc = int(row["poc"])
            self_feats[t] = cache["self_features"][poc].astype(np.float32)
            meta_feats[t] = build_meta_vector(row, self.i_interval)
            qps[t, 0] = normalize_qp(float(row["qp"]), self.cfg["data"])
            targets[t, 0] = np.log1p(float(row["bits"]))
            mse_term = str(self.cfg["loss"].get("mse_term", "log_mse")).lower().strip()
            if mse_term == "vmaf":
                targets[t, 1] = float(row["vmaf"])
            else:
                targets[t, 1] = np.log(float(row["mse"]) + 1e-6)
            valid_loss_mask[t] = float(row["valid_train"])
            temporal_layers[t] = int(row["temporal_layer"])
            pass1_feats[t] = build_pass1_vector(row, self.cfg)

        local_to_poc = {int(r["local_poc"]): int(r["poc"]) for _, r in g.iterrows()}
        poc_to_local = {v: k for k, v in local_to_poc.items()}

        for t in range(T):
            if t not in row_by_local:
                continue
            row = row_by_local[t]
            cur_poc = int(row["poc"])
            cur_y = None
            for k, ref_col in enumerate(["ref_poc_1", "ref_poc_2"]):
                rpoc = int(row[ref_col])
                if rpoc >= 0 and rpoc in poc_to_local:
                    ref_local = poc_to_local[rpoc]
                    ref_idx[t, k] = ref_local
                    pv = None
                    if self._pair_cache:
                        pv = self._pair_cache.get_pair_feats(seq, cur_poc, rpoc)
                    if pv is not None:
                        pair_feats[t, k] = pv
                    elif self._pair_fallback:
                        _warn_pair_fallback_once()
                        if cur_y is None:
                            cur_y = self.y_lowres_manager.load(seq)[cur_poc].astype(np.uint8)
                        ref_y = self.y_lowres_manager.load(seq)[rpoc].astype(np.uint8)
                        pair_feats[t, k] = extract_pair_features(
                            cur_y,
                            ref_y,
                            block_size=self.block_size,
                            changed_threshold=self.changed_threshold,
                            feature_profile=self._pair_feature_profile,
                        )
                    else:
                        raise RuntimeError(
                            f"pair cache 未命中且不允许回退: {seq} cur={cur_poc} ref={rpoc}"
                        )

        result = {
            "self_feats": torch.from_numpy(self_feats),
            "meta_feats": torch.from_numpy(meta_feats),
            "qps": torch.from_numpy(qps),
            "targets": torch.from_numpy(targets),
            "valid_loss_mask": torch.from_numpy(valid_loss_mask),
            "ref_idx": torch.from_numpy(ref_idx),
            "pair_feats": torch.from_numpy(pair_feats),
            "temporal_layers": torch.from_numpy(temporal_layers),
            "topo_order": torch.from_numpy(topo_order),
            "segment_uid": str(g.iloc[0]["segment_uid"]),
            "sequence": str(g.iloc[0]["sequence"]),
        }
        result["pass1_feats"] = torch.from_numpy(pass1_feats)
        return result
