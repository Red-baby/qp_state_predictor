from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import extract_pair_features, pair_feature_names
from .graph import build_segment_topo_order
from .utils import frame_type_onehot


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

    def __len__(self):
        return len(self.df)

    def _get_cache_feats(self, sequence: str, poc: int):
        cache = self.cache_manager.load(sequence)
        self_feats = cache["self_features"][poc].astype(np.float32)
        y_low = cache["y_lowres"][poc].astype(np.uint8)
        return self_feats, y_low

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = str(row["yuv_sequence"])
        poc = int(row["poc"])
        self_feats, y_low = self._get_cache_feats(seq, poc)
        meta = build_meta_vector(row, self.i_interval)
        qp = np.asarray([float(row["qp"]) / 63.0], dtype=np.float32)

        target_bits = np.log1p(float(row["bits"]))
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

        if self.phase == 2:
            pair_feats_all = []
            ref_feats_all = []
            ref_qps_all = []
            ref_valid_all = []
            seg_df = self.df[self.df["segment_uid"] == row["segment_uid"]]

            for ref_col in ["ref_poc_1", "ref_poc_2"]:
                ref_poc = int(row[ref_col])
                if ref_poc >= 0:
                    ref_self_feats, ref_y = self._get_cache_feats(seq, ref_poc)
                    pair_feats = extract_pair_features(
                        y_low,
                        ref_y,
                        block_size=self.block_size,
                        changed_threshold=self.changed_threshold,
                    )
                    ref_qp_row = seg_df[seg_df["poc"] == ref_poc]
                    ref_qp = float(ref_qp_row.iloc[0]["qp"]) if len(ref_qp_row) > 0 else float(row["qp"])
                    ref_valid = 1.0
                else:
                    ref_self_feats = np.zeros_like(self_feats, dtype=np.float32)
                    pair_feats = np.zeros(self._pair_dim, dtype=np.float32)
                    ref_qp = 0.0
                    ref_valid = 0.0

                pair_feats_all.append(pair_feats.astype(np.float32))
                ref_feats_all.append(ref_self_feats.astype(np.float32))
                ref_qps_all.append([ref_qp / 63.0])
                ref_valid_all.append(ref_valid)

            out.update({
                "ref_feats": torch.from_numpy(np.stack(ref_feats_all, axis=0)),
                "pair_feats": torch.from_numpy(np.stack(pair_feats_all, axis=0)),
                "ref_qps": torch.from_numpy(np.asarray(ref_qps_all, dtype=np.float32)),
                "ref_valid_mask": torch.from_numpy(np.asarray(ref_valid_all, dtype=np.float32)),
            })

        return out


class SegmentDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: dict, split_df: pd.DataFrame):
        self.cfg = cfg
        self.i_interval = int(cfg["data"]["i_interval"])
        self.block_size = int(cfg["features"]["pair_block_size"])
        self.changed_threshold = float(cfg["features"]["changed_threshold"])
        self.cache_manager = CacheManager(cfg["data"]["cache_dir"])
        self._pair_dim = len(pair_feature_names())

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

        row_by_local = {int(r["local_poc"]): r for _, r in g.iterrows()}

        for t in range(T):
            if t not in row_by_local:
                continue
            row = row_by_local[t]
            poc = int(row["poc"])
            self_feats[t] = cache["self_features"][poc].astype(np.float32)
            meta_feats[t] = build_meta_vector(row, self.i_interval)
            qps[t, 0] = float(row["qp"]) / 63.0
            targets[t, 0] = np.log1p(float(row["bits"]))
            targets[t, 1] = np.log(float(row["mse"]) + 1e-6)
            valid_loss_mask[t] = float(row["valid_train"])
            frame_type_ids[t] = int(np.argmax(meta_feats[t, :4]))
            temporal_layers[t] = int(row["temporal_layer"])

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
                    ref_y = cache["y_lowres"][rpoc].astype(np.uint8)
                    pair_feats[t, k] = extract_pair_features(
                        cur_y,
                        ref_y,
                        block_size=self.block_size,
                        changed_threshold=self.changed_threshold,
                    )

        return {
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
