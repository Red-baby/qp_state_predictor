from __future__ import annotations

import copy
import yaml


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


DEFAULT_CONFIG = {
    "seed": 42,
    "data": {
        "labels_csv": "",
        "yuv_root": "",
        "cache_dir": "",
        "output_root": "",
        "sequence_col": "sequence",
        "poc_col": "poc",
        "qp_col": "qp",
        "bits_col": "bits",
        "psnr_col": "psnr",
        "mse_col": "mse",
        "group_cols": ["sequence"],
        "split_by_col": "sequence",
        "yuv_sequence_col": "sequence",
        "yuv_filename_template": "{sequence}.yuv",
        "width": 1280,
        "height": 720,
        "bit_depth": 8,
        "resize_width": 192,
        "resize_height": 108,
        "i_interval": 125,
        "gop_size": 16,
        "infer_refs_if_missing": True,
        "explicit_ref_columns": {
            "frame_type": None,
            "temporal_layer": None,
            "ref_poc_1": None,
            "ref_poc_2": None,
        },
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    },
    "features": {
        "block_size": 8,
        "entropy_bins": 32,
        "edge_threshold": 0.08,
        "changed_threshold": 0.03,
        "pair_block_size": 8,
    },
    "train": {
        "num_workers": 4,
        "epochs": 40,
        "batch_size_phase1": 256,
        "batch_size_phase2": 256,
        "batch_size_phase3": 8,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-5,
        "huber_delta": 0.2,
        "grad_clip": 5.0,
        "amp": True,
        "device": "cuda",
        "save_every": 1,
    },
    "loss": {
        "bits_weight": 1.0,
        "mse_weight": 1.0,
        "aux_weight": 0.3,
    },
    "model": {
        "self_hidden": 128,
        "edge_hidden": 128,
        "state_hidden": 64,
        "head_hidden": 128,
        "dropout": 0.1,
        "state_dim": 32,
    },
    "eval": {
        "max_psnr_value": 255.0,
    },
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f)
    return _deep_update(DEFAULT_CONFIG, user_cfg or {})
