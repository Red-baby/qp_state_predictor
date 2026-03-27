from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import load_config
from .features import (
    FEATURE_PROFILE_BITS,
    FEATURE_PROFILE_LEGACY,
    FEATURE_PROFILE_VMAF,
    extract_self_features,
    self_feature_storage_key,
)
from .manifest import build_manifest
from .utils import ensure_dir
from .yuvio import YUVReader420, resize_y


PROFILES = (
    FEATURE_PROFILE_LEGACY,
    FEATURE_PROFILE_BITS,
    FEATURE_PROFILE_VMAF,
)


def _required_keys() -> set[str]:
    return {"y_lowres"} | {self_feature_storage_key(profile) for profile in PROFILES}


def _resolve_workers(requested: int, num_jobs: int) -> int:
    if num_jobs <= 1:
        return 1
    if requested == 0:
        return max(1, min(os.cpu_count() or 1, num_jobs))
    return max(1, min(int(requested), num_jobs))


def _process_sequence_cache(job: dict) -> str:
    sequence = str(job["sequence"])
    out_path = Path(job["out_path"])
    required_keys = set(job["required_keys"])

    if out_path.exists():
        data = np.load(out_path, allow_pickle=False, mmap_mode="r")
        try:
            existing_keys = set(data.files)
        finally:
            data.close()
        if required_keys.issubset(existing_keys):
            return f"[skip] cache exists: {out_path}"
        print(f"[rebuild] cache missing new feature keys: {out_path}")

    frame_ids = [int(p) for p in job["frame_ids"]]
    max_poc = max(frame_ids)
    reader = YUVReader420(
        str(job["yuv_path"]),
        width=int(job["width"]),
        height=int(job["height"]),
        bit_depth=int(job["bit_depth"]),
    )

    out_w = int(job["resize_width"])
    out_h = int(job["resize_height"])
    y_lowres = np.zeros((max_poc + 1, out_h, out_w), dtype=np.uint8)
    self_buffers: dict[str, np.ndarray] = {}
    iterator = frame_ids
    if bool(job.get("show_progress", False)):
        iterator = tqdm(frame_ids, desc=f"cache {sequence}")

    for poc in iterator:
        y = reader.read_y(int(poc))
        y_small = resize_y(y, out_w=out_w, out_h=out_h)

        y_lowres[poc] = y_small
        for profile in PROFILES:
            key = self_feature_storage_key(profile)
            feats = extract_self_features(
                y_small,
                block_size=int(job["block_size"]),
                entropy_bins=int(job["entropy_bins"]),
                edge_threshold=float(job["edge_threshold"]),
                feature_profile=profile,
            )
            if key not in self_buffers:
                self_buffers[key] = np.zeros((max_poc + 1, feats.shape[0]), dtype=np.float32)
            self_buffers[key][poc] = feats.astype(np.float32)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=str(out_path.parent))
    os.close(tmp_fd)
    try:
        np.savez_compressed(tmp_path, y_lowres=y_lowres, **self_buffers)
        os.replace(tmp_path, str(out_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return f"[saved] {out_path}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
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

    width = int(data_cfg["width"])
    height = int(data_cfg["height"])
    bit_depth = int(data_cfg["bit_depth"])
    out_w = int(data_cfg["resize_width"])
    out_h = int(data_cfg["resize_height"])
    filename_tmpl = data_cfg["yuv_filename_template"]
    yuv_root = Path(data_cfg["yuv_root"])
    required_keys = sorted(_required_keys())

    jobs = []
    for sequence, g in manifest.groupby("yuv_sequence"):
        jobs.append(
            {
                "sequence": str(sequence),
                "frame_ids": sorted(set(int(p) for p in g["poc"].tolist())),
                "yuv_path": str(yuv_root / filename_tmpl.format(sequence=sequence)),
                "out_path": str(cache_dir / f"{sequence}.npz"),
                "width": width,
                "height": height,
                "bit_depth": bit_depth,
                "resize_width": out_w,
                "resize_height": out_h,
                "block_size": int(feat_cfg["block_size"]),
                "entropy_bins": int(feat_cfg["entropy_bins"]),
                "edge_threshold": float(feat_cfg["edge_threshold"]),
                "required_keys": required_keys,
            }
        )

    workers = _resolve_workers(int(args.workers), len(jobs))
    if workers <= 1:
        for job in jobs:
            job["show_progress"] = True
            print(_process_sequence_cache(job))
        return

    print(f"[parallel] preprocess_cache workers={workers} sequences={len(jobs)}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_process_sequence_cache, job): job["sequence"] for job in jobs}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="cache sequences"):
            sequence = future_map[future]
            try:
                print(future.result())
            except Exception as exc:
                raise RuntimeError(f"preprocess_cache failed for sequence={sequence!r}") from exc


if __name__ == "__main__":
    main()
