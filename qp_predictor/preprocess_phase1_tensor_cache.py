from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import apply_train_overrides, is_double_bits_cfg, load_config
from .datasets import _build_meta_vector_fast, _build_pass1_vector_fast
from .features import meta_feature_names, pass1_feature_names, resolve_feature_profile, self_feature_names, self_feature_storage_key
from .manifest import build_manifest
from .phase1_tensor_cache import phase1_tensor_cache_filename
from .utils import ensure_dir, normalize_qp


def _resolve_workers(requested: int, num_jobs: int) -> int:
    if num_jobs <= 1:
        return 1
    if requested == 0:
        return max(1, min(os.cpu_count() or 1, num_jobs))
    return max(1, min(int(requested), num_jobs))


def _required_keys() -> set[str]:
    return {
        "present_mask",
        "self_feats",
        "meta_feats",
        "qp",
        "target",
        "temporal_layer",
        "valid_mask",
        "pass1_feats",
        "meta_dim",
        "self_dim",
        "pass1_dim",
    }


def _process_sequence_cache(job: dict) -> str:
    sequence = str(job["sequence"])
    out_path = Path(job["out_path"])
    required_keys = set(job["required_keys"])
    if out_path.is_file() and not bool(job.get("force", False)):
        data = np.load(out_path, allow_pickle=False)
        try:
            existing_keys = set(data.files)
        finally:
            data.close()
        if required_keys.issubset(existing_keys):
            return f"[skip] exists: {out_path}"

    base_path = Path(job["base_path"])
    if not base_path.is_file():
        return f"[skip] no base cache: {base_path}"

    rows: list[dict] = list(job["rows"])
    if not rows:
        return f"[skip] no rows: {sequence}"

    self_dim = int(job["self_dim"])
    meta_dim = int(job["meta_dim"])
    pass1_dim = int(job["pass1_dim"])
    data_cfg = job["data_cfg"]
    cfg_j = job["cfg"]
    pure_bits = is_double_bits_cfg(cfg_j)
    mse_term = str(cfg_j.get("loss", {}).get("mse_term", "log_mse")).lower().strip()
    max_poc = max(int(r["poc"]) for r in rows)
    n = max_poc + 1

    present_mask = np.zeros((n,), dtype=bool)
    self_feats = np.zeros((n, self_dim), dtype=np.float32)
    meta_feats = np.zeros((n, meta_dim), dtype=np.float32)
    qp = np.zeros((n, 1), dtype=np.float32)
    target = np.zeros((n, 1), dtype=np.float32) if pure_bits else np.zeros((n, 2), dtype=np.float32)
    temporal_layer = -np.ones((n,), dtype=np.int64)
    valid_mask = np.zeros((n,), dtype=np.float32)
    pass1_feats = np.zeros((n, pass1_dim), dtype=np.float32)

    base = np.load(base_path, allow_pickle=False)
    try:
        self_arr = np.asarray(base[job["self_key"]], dtype=np.float32)

        iterator = rows
        if bool(job.get("show_progress", False)):
            iterator = tqdm(rows, desc=f"phase1_tensor {sequence}")

        for row in iterator:
            poc = int(row["poc"])
            present_mask[poc] = True
            self_feats[poc] = self_arr[poc]
            meta_feats[poc] = _build_meta_vector_fast(
                temporal_layer=int(row["temporal_layer"]),
                intra_period_pos=float(row["intra_period_pos"]),
                ref_distance_1=int(row["ref_distance_1"]),
                ref_distance_2=int(row["ref_distance_2"]),
                segment_span=int(row["segment_span"]),
            )
            qp[poc, 0] = normalize_qp(float(row["qp"]), data_cfg)
            target[poc, 0] = np.log1p(float(row["bits"]))
            if not pure_bits:
                target[poc, 1] = float(row["vmaf"]) if mse_term == "vmaf" else np.log(float(row["mse"]) + 1e-6)
            temporal_layer[poc] = int(row["temporal_layer"])
            valid_mask[poc] = float(row["valid_train"])
            pass1_feats[poc] = _build_pass1_vector_fast(
                pass1_qp=float(row["pass1_qp"]),
                pass1_log_bits=float(row["pass1_log_bits"]),
                pass1_delta_qp=float(row["pass1_delta_qp"]),
                pass1_vmaf=float(row.get("pass1_vmaf", 0.0)),
                pass1_log_mse=float(row["pass1_log_mse"]),
                cfg=cfg_j,
            )
    finally:
        base.close()

    payload = {
        "present_mask": present_mask,
        "self_feats": self_feats,
        "meta_feats": meta_feats,
        "qp": qp,
        "target": target,
        "temporal_layer": temporal_layer,
        "valid_mask": valid_mask,
        "pass1_feats": pass1_feats,
        "meta_dim": np.int32(meta_dim),
        "self_dim": np.int32(self_dim),
        "pass1_dim": np.int32(pass1_dim),
    }

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=str(out_path.parent))
    os.close(tmp_fd)
    try:
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, str(out_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return f"[saved] {out_path}"


def main():
    parser = argparse.ArgumentParser(description="为 Phase 1 生成按 sequence 预打包的 tensor cache（无 pair/ref）。")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="按 sequence 并行的进程数；1 为串行，0 为自动。")
    parser.add_argument("--only-sequence", type=str, default=None, help="仅处理一个 yuv_sequence，便于调试。")
    parser.add_argument("--model-mode", default=None, choices=["single", "double"])
    parser.add_argument("--double-target", default=None, choices=["bits", "distortion"], dest="double_target")
    parser.add_argument("--mse-term", default=None, dest="mse_term")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_train_overrides(cfg, model_mode=args.model_mode, double_target=args.double_target, mse_term=args.mse_term)
    manifest = build_manifest(cfg)

    feature_profile = resolve_feature_profile(cfg, phase=1)
    cache_dir = Path(cfg["data"]["cache_dir"])
    ensure_dir(cache_dir)
    self_key = self_feature_storage_key(feature_profile)
    required_keys = sorted(_required_keys())
    self_dim = len(self_feature_names(feature_profile))
    meta_dim = len(meta_feature_names(feature_profile))
    pass1_dim = len(pass1_feature_names(cfg))

    jobs = []
    for sequence, g in manifest.groupby("yuv_sequence", sort=False):
        seq = str(sequence)
        if args.only_sequence is not None and seq != str(args.only_sequence):
            continue
        jobs.append(
            {
                "sequence": seq,
                "rows": g.to_dict("records"),
                "out_path": str(cache_dir / phase1_tensor_cache_filename(cfg, seq, feature_profile)),
                "base_path": str(cache_dir / f"{seq}.npz"),
                "required_keys": required_keys,
                "self_key": self_key,
                "self_dim": self_dim,
                "meta_dim": meta_dim,
                "pass1_dim": pass1_dim,
                "cfg": cfg,
                "data_cfg": cfg["data"],
                "force": bool(args.force),
            }
        )

    workers = _resolve_workers(int(args.workers), len(jobs))
    if workers <= 1:
        for job in jobs:
            job["show_progress"] = True
            print(_process_sequence_cache(job))
        return

    print(f"[parallel] preprocess_phase1_tensor_cache workers={workers} sequences={len(jobs)}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_process_sequence_cache, job): job["sequence"] for job in jobs}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="phase1_tensor sequences"):
            sequence = future_map[future]
            try:
                print(future.result())
            except Exception as exc:
                raise RuntimeError(f"preprocess_phase1_tensor_cache failed for sequence={sequence!r}") from exc


if __name__ == "__main__":
    main()
