from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import load_config
from .features import (
    FEATURE_PROFILE_BITS,
    FEATURE_PROFILE_LEGACY,
    FEATURE_PROFILE_VMAF,
    extract_pair_features,
    pair_feature_names,
    pair_feature_storage_key,
)
from .manifest import build_manifest
from .utils import ensure_dir


PROFILES = (
    FEATURE_PROFILE_LEGACY,
    FEATURE_PROFILE_BITS,
    FEATURE_PROFILE_VMAF,
)


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


def _resolve_workers(requested: int, num_jobs: int) -> int:
    if num_jobs <= 1:
        return 1
    if requested == 0:
        return max(1, min(os.cpu_count() or 1, num_jobs))
    return max(1, min(int(requested), num_jobs))


def _process_sequence_pair_cache(job: dict) -> str:
    sequence = str(job["sequence"])
    out_path = Path(job["out_path"])
    required_keys = set(job["required_keys"])

    if out_path.is_file() and not bool(job["force"]):
        data = np.load(out_path, allow_pickle=False, mmap_mode="r")
        try:
            existing_keys = set(data.files)
        finally:
            data.close()
        if required_keys.issubset(existing_keys):
            return f"[skip] exists: {out_path}"
        print(f"[rebuild] pair cache missing new feature keys: {out_path}")

    base_path = Path(job["base_path"])
    if not base_path.is_file():
        return f"[skip] no base cache: {base_path}"

    ordered = [tuple(edge) for edge in job["edges"]]
    if not ordered:
        return f"[skip] no edges: {sequence}"

    data = np.load(base_path, allow_pickle=False, mmap_mode="r")
    try:
        if "y_lowres" not in data:
            raise KeyError(f"{base_path} missing y_lowres")
        y_lowres = data["y_lowres"]

        pair_dims = {profile: len(pair_feature_names(profile)) for profile in PROFILES}
        cur_list: list[int] = []
        ref_list: list[int] = []
        feats_lists: dict[str, list[np.ndarray]] = {profile: [] for profile in PROFILES}

        for cur_poc, ref_poc in ordered:
            if cur_poc >= int(y_lowres.shape[0]) or ref_poc >= int(y_lowres.shape[0]):
                raise IndexError(f"{sequence} poc out of range: cur={cur_poc} ref={ref_poc} n={y_lowres.shape[0]}")
            cur_y = np.asarray(y_lowres[cur_poc], dtype=np.uint8)
            ref_y = np.asarray(y_lowres[ref_poc], dtype=np.uint8)
            cur_list.append(cur_poc)
            ref_list.append(ref_poc)
            for profile in PROFILES:
                pv = extract_pair_features(
                    cur_y,
                    ref_y,
                    block_size=int(job["pair_block_size"]),
                    changed_threshold=float(job["changed_threshold"]),
                    feature_profile=profile,
                )
                if int(pv.shape[0]) != int(pair_dims[profile]):
                    raise ValueError(f"pair_dim mismatch {pv.shape[0]} != {pair_dims[profile]}")
                feats_lists[profile].append(pv.astype(np.float32))
    finally:
        data.close()

    save_payload = {
        "cur_pocs": np.asarray(cur_list, dtype=np.int32),
        "ref_pocs": np.asarray(ref_list, dtype=np.int32),
        "resize_width": np.int32(job["resize_width"]),
        "resize_height": np.int32(job["resize_height"]),
        "pair_block_size": np.int32(job["pair_block_size"]),
        "changed_threshold": np.float32(job["changed_threshold"]),
    }
    for profile in PROFILES:
        save_payload[pair_feature_storage_key(profile)] = np.stack(feats_lists[profile], axis=0)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=str(out_path.parent))
    os.close(tmp_fd)
    try:
        np.savez_compressed(tmp_path, **save_payload)
        os.replace(tmp_path, str(out_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return f"[saved] {out_path}  edges={len(cur_list)}"


def main():
    parser = argparse.ArgumentParser(description="基于已有 <seq>.npz 中的 y_lowres 增量生成 pair sidecar 缓存。")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true", help="已存在 .pair.npz 时仍重写。")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="按 sequence 并行的进程数；1 为串行，0 为自动使用 CPU 核数。",
    )
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
    required_keys = {"cur_pocs", "ref_pocs"} | {pair_feature_storage_key(profile) for profile in PROFILES}

    sequences = sorted(manifest["yuv_sequence"].astype(str).unique().tolist())
    jobs = []
    for sequence in sequences:
        jobs.append(
            {
                "sequence": sequence,
                "out_path": str(cache_dir / f"{sequence}{suffix}"),
                "base_path": str(cache_dir / f"{sequence}.npz"),
                "edges": sorted(_collect_edges_for_sequence(manifest, sequence)),
                "resize_width": rw,
                "resize_height": rh,
                "pair_block_size": block_size,
                "changed_threshold": changed_th,
                "required_keys": sorted(required_keys),
                "force": bool(args.force),
            }
        )

    workers = _resolve_workers(int(args.workers), len(jobs))
    if workers <= 1:
        for job in tqdm(jobs, desc="pair_cache sequences"):
            print(_process_sequence_pair_cache(job))
        return

    print(f"[parallel] preprocess_pair_cache workers={workers} sequences={len(jobs)}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_process_sequence_pair_cache, job): job["sequence"] for job in jobs}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="pair_cache sequences"):
            sequence = future_map[future]
            try:
                print(future.result())
            except Exception as exc:
                raise RuntimeError(f"preprocess_pair_cache failed for sequence={sequence!r}") from exc


if __name__ == "__main__":
    main()
