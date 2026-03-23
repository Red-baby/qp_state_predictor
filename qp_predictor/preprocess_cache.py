from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import load_config
from .features import extract_self_features
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

    for sequence, g in manifest.groupby("yuv_sequence"):
        out_path = cache_dir / f"{sequence}.npz"
        if out_path.exists():
            print(f"[skip] cache exists: {out_path}")
            continue

        frame_ids = sorted(set(g["poc"].tolist()))
        max_poc = max(frame_ids)
        yuv_path = yuv_root / filename_tmpl.format(sequence=sequence)
        reader = YUVReader420(str(yuv_path), width=width, height=height, bit_depth=bit_depth)

        y_lowres = np.zeros((max_poc + 1, out_h, out_w), dtype=np.uint8)
        self_features = None

        for poc in tqdm(frame_ids, desc=f"cache {sequence}"):
            y = reader.read_y(int(poc))
            y_small = resize_y(y, out_w=out_w, out_h=out_h)
            feats = extract_self_features(
                y_small,
                block_size=int(feat_cfg["block_size"]),
                entropy_bins=int(feat_cfg["entropy_bins"]),
                edge_threshold=float(feat_cfg["edge_threshold"]),
            )
            if self_features is None:
                self_features = np.zeros((max_poc + 1, feats.shape[0]), dtype=np.float32)

            y_lowres[poc] = y_small
            self_features[poc] = feats.astype(np.float32)

        np.savez_compressed(out_path, y_lowres=y_lowres, self_features=self_features)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
