from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .features import pair_feature_names


class PairCacheManager:
    """加载 `<sequence>.pair.npz` sidecar，按 (cur_poc, ref_poc) 查询 pair_feats。"""

    def __init__(self, cfg: dict):
        feat = cfg["features"]
        data = cfg["data"]
        self.cache_dir = Path(data["cache_dir"])
        self.suffix = str(feat.get("pair_cache_suffix", ".pair.npz"))
        self.required = bool(feat.get("pair_cache_required", False))
        self.pair_dim = len(pair_feature_names())
        self._expected = {
            "resize_width": int(data["resize_width"]),
            "resize_height": int(data["resize_height"]),
            "pair_block_size": int(feat["pair_block_size"]),
            "changed_threshold": float(feat["changed_threshold"]),
        }
        self._stores: Dict[str, Any] = {}

    def _read_scalar(self, data: np.lib.npyio.NpzFile, key: str) -> Union[float, int]:
        arr = data[key]
        if arr.shape == ():
            v = arr.item()
        else:
            v = arr.flat[0]
        if isinstance(self._expected[key], float):
            return float(v)
        return int(v)

    def _validate_meta(self, data: np.lib.npyio.NpzFile) -> None:
        for k, exp in self._expected.items():
            got = self._read_scalar(data, k)
            if isinstance(exp, float):
                if not np.isfinite(got) or abs(float(got) - float(exp)) > 1e-5:
                    raise ValueError(f"pair cache meta mismatch {k}: file={got} config={exp}")
            else:
                if int(got) != int(exp):
                    raise ValueError(f"pair cache meta mismatch {k}: file={got} config={exp}")
        pf = data["pair_feats"]
        if int(pf.shape[1]) != self.pair_dim:
            raise ValueError(f"pair_feats dim {pf.shape[1]} != expected {self.pair_dim}")

    def _open(self, sequence: str) -> Optional[dict]:
        if sequence in self._stores:
            return self._stores[sequence]

        path = self.cache_dir / f"{sequence}{self.suffix}"
        if not path.is_file():
            if self.required:
                raise FileNotFoundError(f"pair cache required but missing: {path}")
            return None

        data = np.load(path, allow_pickle=False, mmap_mode="r")
        self._validate_meta(data)
        cur = np.asarray(data["cur_pocs"], dtype=np.int64)
        ref = np.asarray(data["ref_pocs"], dtype=np.int64)
        pair_feats = data["pair_feats"]
        edge_map = {(int(cur[i]), int(ref[i])): int(i) for i in range(int(cur.shape[0]))}
        store = {"pair_feats": pair_feats, "edge_map": edge_map}
        self._stores[sequence] = store
        return store

    def get_pair_feats(self, sequence: str, cur_poc: int, ref_poc: int) -> Optional[np.ndarray]:
        """命中返回 float32 向量；无文件/无边/未启用时返回 None，由调用方决定是否在线计算。"""
        store = self._open(str(sequence))
        if store is None:
            return None
        idx = store["edge_map"].get((int(cur_poc), int(ref_poc)))
        if idx is None:
            return None
        out = np.asarray(store["pair_feats"][idx], dtype=np.float32).copy()
        return out

    def clear_memory_cache(self) -> None:
        self._stores.clear()
