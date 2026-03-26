from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
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

    profiles = (
        FEATURE_PROFILE_LEGACY,
        FEATURE_PROFILE_BITS,
        FEATURE_PROFILE_VMAF,
    )
    required_keys = {"y_lowres"} | {self_feature_storage_key(profile) for profile in profiles}

    for sequence, g in manifest.groupby("yuv_sequence"):
        out_path = cache_dir / f"{sequence}.npz"
        if out_path.exists():
            data = np.load(out_path, allow_pickle=False, mmap_mode="r")
            try:
                existing_keys = set(data.files)
            finally:
                data.close()
            if required_keys.issubset(existing_keys):
                print(f"[skip] cache exists: {out_path}")
                continue
            print(f"[rebuild] cache missing new feature keys: {out_path}")

        frame_ids = sorted(set(g["poc"].tolist()))
        max_poc = max(frame_ids)
        yuv_path = yuv_root / filename_tmpl.format(sequence=sequence)
        reader = YUVReader420(str(yuv_path), width=width, height=height, bit_depth=bit_depth)

        y_lowres = np.zeros((max_poc + 1, out_h, out_w), dtype=np.uint8)
        self_buffers: dict[str, np.ndarray] = {}

        for poc in tqdm(frame_ids, desc=f"cache {sequence}"):
            y = reader.read_y(int(poc))
            y_small = resize_y(y, out_w=out_w, out_h=out_h)

            y_lowres[poc] = y_small
            for profile in profiles:
                key = self_feature_storage_key(profile)
                feats = extract_self_features(
                    y_small,
                    block_size=int(feat_cfg["block_size"]),
                    entropy_bins=int(feat_cfg["entropy_bins"]),
                    edge_threshold=float(feat_cfg["edge_threshold"]),
                    feature_profile=profile,
                )
                if key not in self_buffers:
                    self_buffers[key] = np.zeros((max_poc + 1, feats.shape[0]), dtype=np.float32)
                self_buffers[key][poc] = feats.astype(np.float32)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=str(cache_dir))
        os.close(tmp_fd)
        try:
            np.savez_compressed(tmp_path, y_lowres=y_lowres, **self_buffers)
            os.replace(tmp_path, str(out_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
