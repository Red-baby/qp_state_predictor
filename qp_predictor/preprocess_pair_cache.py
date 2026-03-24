from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import load_config
from .features import extract_pair_features, pair_feature_names
from .manifest import build_manifest
from .utils import ensure_dir


def _collect_edges_for_sequence(manifest: pd.DataFrame, yuv_sequence: str) -> set[tuple[int, int]]:
    """所有 (cur_poc, ref_poc)，ref>=0，去重。"""
    g = manifest[manifest["yuv_sequence"].astype(str) == str(yuv_sequence)]
    edges: set[tuple[int, int]] = set()
    for _, row in g.iterrows():
        cur = int(row["poc"])
        for col in ("ref_poc_1", "ref_poc_2"):
            r = int(row[col])
            if r >= 0:
                edges.add((cur, r))
    return edges


def main():
    parser = argparse.ArgumentParser(description="基于已有 <seq>.npz 中的 y_lowres 增量生成 pair sidecar 缓存。")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true", help="已存在 .pair.npz 时仍重写。")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = build_manifest(cfg)

    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    cache_dir = Path(data_cfg["cache_dir"])
    ensure_dir(cache_dir)

    suffix = str(feat_cfg.get("pair_cache_suffix", ".pair.npz"))
    block_size = int(feat_cfg["pair_block_size"])
    changed_th = float(feat_cfg["changed_threshold"])
    rw = int(data_cfg["resize_width"])
    rh = int(data_cfg["resize_height"])
    pair_dim = len(pair_feature_names())

    sequences = sorted(manifest["yuv_sequence"].astype(str).unique().tolist())

    for sequence in tqdm(sequences, desc="pair_cache sequences"):
        out_path = cache_dir / f"{sequence}{suffix}"
        if out_path.is_file() and not args.force:
            print(f"[skip] exists: {out_path}")
            continue

        base_path = cache_dir / f"{sequence}.npz"
        if not base_path.is_file():
            print(f"[skip] no base cache: {base_path}")
            continue

        edges = _collect_edges_for_sequence(manifest, sequence)
        if not edges:
            print(f"[skip] no edges: {sequence}")
            continue

        data = np.load(base_path, allow_pickle=False, mmap_mode="r")
        if "y_lowres" not in data:
            raise KeyError(f"{base_path} missing y_lowres")
        y_lowres = data["y_lowres"]

        ordered = sorted(edges)
        cur_list: list[int] = []
        ref_list: list[int] = []
        feats_list: list[np.ndarray] = []

        for cur_poc, ref_poc in ordered:
            if cur_poc >= int(y_lowres.shape[0]) or ref_poc >= int(y_lowres.shape[0]):
                raise IndexError(f"{sequence} poc out of range: cur={cur_poc} ref={ref_poc} n={y_lowres.shape[0]}")
            cur_y = np.asarray(y_lowres[cur_poc], dtype=np.uint8)
            ref_y = np.asarray(y_lowres[ref_poc], dtype=np.uint8)
            pv = extract_pair_features(
                cur_y,
                ref_y,
                block_size=block_size,
                changed_threshold=changed_th,
            )
            if int(pv.shape[0]) != pair_dim:
                raise ValueError(f"pair_dim mismatch {pv.shape[0]} != {pair_dim}")
            cur_list.append(cur_poc)
            ref_list.append(ref_poc)
            feats_list.append(pv.astype(np.float32))

        cur_pocs = np.asarray(cur_list, dtype=np.int32)
        ref_pocs = np.asarray(ref_list, dtype=np.int32)
        pair_feats = np.stack(feats_list, axis=0)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=str(cache_dir))
        os.close(tmp_fd)
        try:
            np.savez_compressed(
                tmp_path,
                cur_pocs=cur_pocs,
                ref_pocs=ref_pocs,
                pair_feats=pair_feats,
                resize_width=np.int32(rw),
                resize_height=np.int32(rh),
                pair_block_size=np.int32(block_size),
                changed_threshold=np.float32(changed_th),
            )
            os.replace(tmp_path, str(out_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        print(f"[saved] {out_path}  edges={len(cur_list)}")


if __name__ == "__main__":
    main()
