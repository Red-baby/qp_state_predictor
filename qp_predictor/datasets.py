from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import extract_pair_features, pair_feature_names, pass1_feature_names
from .graph import build_segment_topo_order
from .pair_cache import PairCacheManager
from .utils import frame_type_onehot, normalize_qp

_PAIR_FALLBACK_WARNED = False


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


class CacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self._cache = {}

    def load(self, sequence: str):
        if sequence not in self._cache:
            path = self.cache_dir / f"{sequence}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Cache file not found: {path}")
            self._cache[sequence] = np.load(path, allow_pickle=False, mmap_mode="r")
        return self._cache[sequence]


def build_meta_vector(row: pd.Series, i_interval: int) -> np.ndarray:
    frame_oh = frame_type_onehot(str(row["frame_type"]))
    tl = float(row["temporal_layer"]) if int(row["temporal_layer"]) >= 0 else -1.0
    num_refs = float(row["num_refs"])
    local_poc = float(row["local_poc"]) / max(i_interval - 1, 1)
    d_prev_i = float(row["distance_to_prev_I"]) / max(i_interval, 1)
    d_next_i = float(row["distance_to_next_I"]) / max(i_interval, 1)
    ref_d1 = float(row["ref_distance_1"]) / max(i_interval, 1) if int(row["ref_distance_1"]) >= 0 else -1.0
    ref_d2 = float(row["ref_distance_2"]) / max(i_interval, 1) if int(row["ref_distance_2"]) >= 0 else -1.0
    is_first = float(row["is_first_after_I"])
    vec = np.concatenate([
        frame_oh,
        np.asarray([
            tl / 8.0,
            num_refs / 2.0,
            local_poc,
            d_prev_i,
            d_next_i,
            ref_d1,
            ref_d2,
            is_first,
        ], dtype=np.float32),
    ], axis=0)
    return vec.astype(np.float32)


def build_pass1_vector(row: pd.Series, cfg: dict) -> np.ndarray:
    """从 manifest 行构建 4 维 pass1 先验：mse_term=vmaf 时第 3 维为 pass1_vmaf（除以 pass1_vmaf_norm_div），否则为 pass1_log_mse。"""
    data_cfg = cfg["data"]
    mse_term = str(cfg.get("loss", {}).get("mse_term", "log_mse")).lower().strip()
    qp_n = normalize_qp(float(row["pass1_qp"]), data_cfg)
    log_bits = float(row["pass1_log_bits"])
    dqp = float(row["pass1_delta_qp"])
    if mse_term == "vmaf":
        div = float(data_cfg.get("pass1_vmaf_norm_div", 100.0))
        v = float(row["pass1_vmaf"]) / max(div, 1e-6)
        return np.asarray([qp_n, log_bits, v, dqp], dtype=np.float32)
    return np.asarray([qp_n, log_bits, float(row["pass1_log_mse"]), dqp], dtype=np.float32)


class FrameDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: dict, split_df: pd.DataFrame, phase: int):
        self.cfg = cfg
        self.phase = int(phase)
        self.i_interval = int(cfg["data"]["i_interval"])
        self.block_size = int(cfg["features"]["pair_block_size"])
        self.changed_threshold = float(cfg["features"]["changed_threshold"])
        self.cache_manager = CacheManager(cfg["data"]["cache_dir"])
        if self.phase not in (1, 2):
            raise ValueError("FrameDataset only supports phase 1/2")

        df = split_df.copy()
        df = df[df["valid_train"] == 1].reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self._pair_dim = len(pair_feature_names())

        self._qp_lookup = {}
        for _, r in self.df.iterrows():
            self._qp_lookup[(str(r["segment_uid"]), int(r["poc"]))] = float(r["qp"])

        self._use_pass1 = bool(cfg["data"].get("use_pass1_features", False))
        self._pass1_dim = len(pass1_feature_names(cfg)) if self._use_pass1 else 0

        self._row_lookup = {}
        if self.phase == 2 and self._use_pass1:
            for _, r in self.df.iterrows():
                self._row_lookup[(str(r["segment_uid"]), int(r["poc"]))] = r

        feat_cfg = cfg["features"]
        self._use_pair_cache = self.phase == 2 and bool(feat_cfg.get("use_pair_cache", False))
        self._pair_fallback = bool(feat_cfg.get("pair_cache_fallback_online", True))
        self._pair_cache = PairCacheManager(cfg) if self._use_pair_cache else None

    def __len__(self):
        return len(self.df)

    def _get_cache_feats(self, sequence: str, poc: int, *, need_y_low: bool):
        """need_y_low=False 时只读 self_features（Phase 1），避免多读整帧 y_lowres，减轻远端盘随机读。"""
        cache = self.cache_manager.load(sequence)
        self_feats = cache["self_features"][poc].astype(np.float32)
        if not need_y_low:
            return self_feats, None
        y_low = cache["y_lowres"][poc].astype(np.uint8)
        return self_feats, y_low

    def _phase2_need_cur_y_low(self, seq: str, poc: int, row: pd.Series) -> bool:
        """若 pair 全在 sidecar 命中则无需读当前帧 y_lowres。"""
        if not self._pair_cache:
            return True
        for ref_col in ("ref_poc_1", "ref_poc_2"):
            rp = int(row[ref_col])
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

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = str(row["yuv_sequence"])
        poc = int(row["poc"])
        if self.phase == 2:
            need_y = self._phase2_need_cur_y_low(seq, poc, row)
        else:
            need_y = False
        self_feats, y_low = self._get_cache_feats(seq, poc, need_y_low=need_y)
        meta = build_meta_vector(row, self.i_interval)
        qp = np.asarray([normalize_qp(float(row["qp"]), self.cfg["data"])], dtype=np.float32)

        target_bits = np.log1p(float(row["bits"]))
        mse_term = str(self.cfg["loss"].get("mse_term", "log_mse")).lower().strip()
        if mse_term == "vmaf":
            target_mse = float(row["vmaf"])
        else:
            target_mse = np.log(float(row["mse"]) + 1e-6)
        target = np.asarray([target_bits, target_mse], dtype=np.float32)

        out = {
            "self_feats": torch.from_numpy(self_feats),
            "meta_feats": torch.from_numpy(meta),
            "qp": torch.from_numpy(qp),
            "target": torch.from_numpy(target),
            "frame_type_id": torch.tensor(int(np.argmax(meta[:4])), dtype=torch.long),
            "temporal_layer": torch.tensor(int(row["temporal_layer"]), dtype=torch.long),
            "valid_mask": torch.tensor(float(row["valid_train"]), dtype=torch.float32),
            "base_uid": str(row["base_uid"]),
            "sequence": str(row["sequence"]),
        }

        if self._use_pass1:
            out["pass1_feats"] = torch.from_numpy(build_pass1_vector(row, self.cfg))

        if self.phase == 2:
            pair_feats_all = []
            ref_feats_all = []
            ref_qps_all = []
            ref_valid_all = []
            ref_pass1_all = []
            seg_uid = str(row["segment_uid"])

            for ref_col in ["ref_poc_1", "ref_poc_2"]:
                ref_poc = int(row[ref_col])
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
                        )
                    else:
                        raise RuntimeError(
                            f"pair cache 未命中且不允许回退: {seq} cur={poc} ref={ref_poc}"
                        )
                    ref_qp = self._qp_lookup.get((seg_uid, ref_poc), float(row["qp"]))
                    ref_valid = 1.0

                    if self._use_pass1:
                        ref_row = self._row_lookup.get((seg_uid, ref_poc))
                        ref_p1 = (
                            build_pass1_vector(ref_row, self.cfg)
                            if ref_row is not None
                            else np.zeros(self._pass1_dim, dtype=np.float32)
                        )
                    else:
                        ref_p1 = None
                else:
                    ref_self_feats = np.zeros_like(self_feats, dtype=np.float32)
                    pair_feats = np.zeros(self._pair_dim, dtype=np.float32)
                    ref_qp = 0.0
                    ref_valid = 0.0
                    ref_p1 = np.zeros(self._pass1_dim, dtype=np.float32) if self._use_pass1 else None

                pair_feats_all.append(pair_feats.astype(np.float32))
                ref_feats_all.append(ref_self_feats.astype(np.float32))
                if ref_poc >= 0:
                    ref_qps_all.append([normalize_qp(ref_qp, self.cfg["data"])])
                else:
                    ref_qps_all.append([0.0])
                ref_valid_all.append(ref_valid)
                if ref_p1 is not None:
                    ref_pass1_all.append(ref_p1)

            out.update({
                "ref_feats": torch.from_numpy(np.stack(ref_feats_all, axis=0)),
                "pair_feats": torch.from_numpy(np.stack(pair_feats_all, axis=0)),
                "ref_qps": torch.from_numpy(np.asarray(ref_qps_all, dtype=np.float32)),
                "ref_valid_mask": torch.from_numpy(np.asarray(ref_valid_all, dtype=np.float32)),
            })
            if self._use_pass1 and ref_pass1_all:
                out["ref_pass1_feats"] = torch.from_numpy(np.stack(ref_pass1_all, axis=0))

        return out


class SegmentDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: dict, split_df: pd.DataFrame):
        self.cfg = cfg
        self.i_interval = int(cfg["data"]["i_interval"])
        self.block_size = int(cfg["features"]["pair_block_size"])
        self.changed_threshold = float(cfg["features"]["changed_threshold"])
        self.cache_manager = CacheManager(cfg["data"]["cache_dir"])
        self._pair_dim = len(pair_feature_names())

        self._use_pass1 = bool(cfg["data"].get("use_pass1_features", False))
        self._pass1_dim = len(pass1_feature_names(cfg)) if self._use_pass1 else 0

        feat_cfg = cfg["features"]
        self._use_pair_cache = bool(feat_cfg.get("use_pair_cache", False))
        self._pair_fallback = bool(feat_cfg.get("pair_cache_fallback_online", True))
        self._pair_cache = PairCacheManager(cfg) if self._use_pair_cache else None

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

        self_feats = np.zeros((T, feat_dim), dtype=np.float32)
        meta_feats = np.zeros((T, 12), dtype=np.float32)
        qps = np.zeros((T, 1), dtype=np.float32)
        targets = np.zeros((T, 2), dtype=np.float32)
        valid_loss_mask = np.zeros((T,), dtype=np.float32)
        ref_idx = -np.ones((T, 2), dtype=np.int64)
        pair_feats = np.zeros((T, 2, self._pair_dim), dtype=np.float32)
        frame_type_ids = np.zeros((T,), dtype=np.int64)
        temporal_layers = -np.ones((T,), dtype=np.int64)
        if self._use_pass1:
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
            frame_type_ids[t] = int(np.argmax(meta_feats[t, :4]))
            temporal_layers[t] = int(row["temporal_layer"])
            if self._use_pass1:
                pass1_feats[t] = build_pass1_vector(row, self.cfg)

        local_to_poc = {int(r["local_poc"]): int(r["poc"]) for _, r in g.iterrows()}
        poc_to_local = {v: k for k, v in local_to_poc.items()}

        for t in range(T):
            if t not in row_by_local:
                continue
            row = row_by_local[t]
            cur_poc = int(row["poc"])
            cur_y = cache["y_lowres"][cur_poc].astype(np.uint8)
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
                        ref_y = cache["y_lowres"][rpoc].astype(np.uint8)
                        pair_feats[t, k] = extract_pair_features(
                            cur_y,
                            ref_y,
                            block_size=self.block_size,
                            changed_threshold=self.changed_threshold,
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
            "frame_type_ids": torch.from_numpy(frame_type_ids),
            "temporal_layers": torch.from_numpy(temporal_layers),
            "topo_order": torch.from_numpy(topo_order),
            "segment_uid": str(g.iloc[0]["segment_uid"]),
            "sequence": str(g.iloc[0]["sequence"]),
        }
        if self._use_pass1:
            result["pass1_feats"] = torch.from_numpy(pass1_feats)
        return result
