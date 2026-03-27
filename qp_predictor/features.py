from __future__ import annotations

from typing import List, Optional

import numpy as np

FEATURE_PROFILE_LEGACY = "legacy"
FEATURE_PROFILE_BITS = "bits"
FEATURE_PROFILE_VMAF = "vmaf"
_VALID_FEATURE_PROFILES = {
    FEATURE_PROFILE_LEGACY,
    FEATURE_PROFILE_BITS,
    FEATURE_PROFILE_VMAF,
}


def normalize_feature_profile(feature_profile: str | None) -> str:
    profile = str(feature_profile or FEATURE_PROFILE_LEGACY).lower().strip()
    if profile not in _VALID_FEATURE_PROFILES:
        raise ValueError(f"unsupported feature profile: {feature_profile!r}")
    return profile


def resolve_feature_profile(cfg: Optional[dict], phase: Optional[int] = None) -> str:
    if cfg is None or int(phase or -1) != 2:
        return FEATURE_PROFILE_LEGACY

    model_cfg = cfg.get("model", {})
    mode = str(model_cfg.get("mode", "single")).lower().strip()
    if mode != "double":
        return FEATURE_PROFILE_LEGACY

    double_target = str(model_cfg.get("double_target", "bits")).lower().strip()
    if double_target == "bits":
        return FEATURE_PROFILE_BITS

    mse_term = str(cfg.get("loss", {}).get("mse_term", "log_mse")).lower().strip()
    if double_target == "distortion" and mse_term == "vmaf":
        return FEATURE_PROFILE_VMAF
    return FEATURE_PROFILE_LEGACY


def self_feature_storage_key(feature_profile: str | None = None) -> str:
    profile = normalize_feature_profile(feature_profile)
    if profile == FEATURE_PROFILE_BITS:
        return "self_features_bits"
    if profile == FEATURE_PROFILE_VMAF:
        return "self_features_vmaf"
    return "self_features"


def pair_feature_storage_key(feature_profile: str | None = None) -> str:
    profile = normalize_feature_profile(feature_profile)
    if profile == FEATURE_PROFILE_BITS:
        return "pair_feats_bits"
    if profile == FEATURE_PROFILE_VMAF:
        return "pair_feats_vmaf"
    return "pair_feats"


def _safe_hist_entropy(img: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(img, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _laplacian(img: np.ndarray) -> np.ndarray:
    center = img
    up = np.pad(img[:-1, :], ((1, 0), (0, 0)), mode="edge")
    down = np.pad(img[1:, :], ((0, 1), (0, 0)), mode="edge")
    left = np.pad(img[:, :-1], ((0, 0), (1, 0)), mode="edge")
    right = np.pad(img[:, 1:], ((0, 0), (0, 1)), mode="edge")
    return (-4.0 * center + up + down + left + right)


def _grad_mag(img: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img)
    return np.sqrt(gx * gx + gy * gy)


def _block_view(img: np.ndarray, block: int) -> np.ndarray:
    h, w = img.shape
    h2 = h - (h % block)
    w2 = w - (w % block)
    if h2 <= 0 or w2 <= 0:
        return img[np.newaxis, np.newaxis, :, :]
    img = img[:h2, :w2]
    return img.reshape(h2 // block, block, w2 // block, block).transpose(0, 2, 1, 3)


def _hf_ratio_fft(img: np.ndarray) -> float:
    spec = np.fft.rfft2(img)
    power = (spec.real ** 2 + spec.imag ** 2)
    h, _ = img.shape
    h_cut = max(1, h // 4)
    w_cut = max(1, power.shape[1] // 4)
    low = power[:h_cut, :w_cut].sum()
    total = power.sum() + 1e-8
    high = total - low
    return float(high / total)


def _avg_pool2(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    if h2 < 2 or w2 < 2:
        return img.copy()
    pooled = img[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))
    return pooled.astype(np.float32, copy=False)


def _tail_mean(x: np.ndarray, ratio: float) -> float:
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    keep = max(1, int(np.ceil(flat.size * float(ratio))))
    idx = np.argpartition(flat, -keep)[-keep:]
    return float(flat[idx].mean())


def _global_ssim_norm(x: np.ndarray, y: np.ndarray) -> float:
    ux = x.mean()
    uy = y.mean()
    vx = x.var()
    vy = y.var()
    cxy = ((x - ux) * (y - uy)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2 * ux * uy + c1) * (2 * cxy + c2)
    den = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
    return float(num / (den + 1e-8))


def _collect_self_stats(
    img: np.ndarray,
    *,
    block_size: int,
    entropy_bins: int,
    edge_threshold: float,
) -> dict[str, float]:
    grad = _grad_mag(img)
    lap = _laplacian(img)
    blocks = _block_view(img, block_size)
    bvar = blocks.var(axis=(2, 3))
    gblocks = _block_view(grad, block_size)
    bg = gblocks.mean(axis=(2, 3))

    return {
        "mean": float(img.mean()),
        "std": float(img.std()),
        "min": float(img.min()),
        "max": float(img.max()),
        "p10": float(np.percentile(img, 10)),
        "p50": float(np.percentile(img, 50)),
        "p90": float(np.percentile(img, 90)),
        "grad_mean": float(grad.mean()),
        "grad_std": float(grad.std()),
        "grad_p90": float(np.percentile(grad, 90)),
        "grad_p99": float(np.percentile(grad, 99)),
        "lap_var": float(lap.var()),
        "edge_density": float((grad > edge_threshold).mean()),
        "entropy": _safe_hist_entropy(img, bins=entropy_bins),
        "block_var_mean": float(bvar.mean()),
        "block_var_std": float(bvar.std()),
        "block_var_p90": float(np.percentile(bvar, 90)),
        "block_var_p99": float(np.percentile(bvar, 99)),
        "block_grad_mean": float(bg.mean()),
        "block_grad_std": float(bg.std()),
        "flat_block_ratio": float((bvar < 0.002).mean()),
        "textured_block_ratio": float((bvar > 0.02).mean()),
        "hf_ratio": _hf_ratio_fft(img),
        "dark_ratio": float((img < 0.15).mean()),
        "bright_ratio": float((img > 0.85).mean()),
    }


def _collect_pair_stats(
    cur: np.ndarray,
    ref: np.ndarray,
    *,
    block_size: int,
    changed_threshold: float,
) -> dict[str, float]:
    diff = cur - ref
    adiff = np.abs(diff)

    gcur = _grad_mag(cur)
    gref = _grad_mag(ref)
    gdiff = np.abs(gcur - gref)

    blocks = _block_view(adiff, block_size)
    bmad = blocks.mean(axis=(2, 3))

    cur_flat = cur.reshape(-1)
    ref_flat = ref.reshape(-1)
    cur_z = cur_flat - cur_flat.mean()
    ref_z = ref_flat - ref_flat.mean()
    denom = np.sqrt((cur_z ** 2).sum() * (ref_z ** 2).sum()) + 1e-8
    ncc = float((cur_z * ref_z).sum() / denom)

    return {
        "absdiff_mean": float(adiff.mean()),
        "absdiff_std": float(adiff.std()),
        "absdiff_p90": float(np.percentile(adiff, 90)),
        "absdiff_p99": float(np.percentile(adiff, 99)),
        "diff_mse": float((diff ** 2).mean()),
        "changed_ratio": float((adiff > changed_threshold).mean()),
        "grad_diff_mean": float(gdiff.mean()),
        "grad_diff_std": float(gdiff.std()),
        "grad_diff_p90": float(np.percentile(gdiff, 90)),
        "grad_diff_p99": float(np.percentile(gdiff, 99)),
        "block_mad_mean": float(bmad.mean()),
        "block_mad_std": float(bmad.std()),
        "block_mad_p90": float(np.percentile(bmad, 90)),
        "block_mad_p99": float(np.percentile(bmad, 99)),
        "worst5_block_mad_mean": _tail_mean(bmad, ratio=0.05),
        "diff_hf_ratio": float(abs(_hf_ratio_fft(cur) - _hf_ratio_fft(ref))),
        "ncc": ncc,
        "g_ssim": _global_ssim_norm(cur, ref),
    }


def self_feature_names(feature_profile: str | None = None) -> List[str]:
    profile = normalize_feature_profile(feature_profile)
    if profile == FEATURE_PROFILE_BITS:
        return [
            "std",
            "grad_mean",
            "grad_p90",
            "grad_p99",
            "lap_var",
            "edge_density",
            "entropy",
            "block_var_mean",
            "block_var_std",
            "block_var_p90",
            "block_var_p99",
            "block_grad_mean",
            "block_grad_std",
            "flat_block_ratio",
            "textured_block_ratio",
            "hf_ratio",
        ]
    if profile == FEATURE_PROFILE_VMAF:
        return [
            "mean",
            "std",
            "dark_ratio",
            "bright_ratio",
            "grad_mean_s1",
            "grad_p90_s1",
            "lap_var_s1",
            "edge_density_s1",
            "entropy_s1",
            "block_var_p90_s1",
            "block_var_p99_s1",
            "block_grad_mean_s1",
            "flat_block_ratio_s1",
            "textured_block_ratio_s1",
            "hf_ratio_s1",
            "grad_mean_s2",
            "grad_p90_s2",
            "lap_var_s2",
            "edge_density_s2",
            "entropy_s2",
            "block_var_p90_s2",
            "block_grad_mean_s2",
            "flat_block_ratio_s2",
            "textured_block_ratio_s2",
            "hf_ratio_s2",
        ]
    return [
        "mean",
        "std",
        "min",
        "max",
        "p10",
        "p50",
        "p90",
        "grad_mean",
        "grad_std",
        "grad_p90",
        "lap_var",
        "edge_density",
        "entropy",
        "block_var_mean",
        "block_var_std",
        "block_var_p90",
        "block_grad_mean",
        "block_grad_std",
        "flat_block_ratio",
        "textured_block_ratio",
        "hf_ratio",
    ]


def meta_feature_names(feature_profile: str | None = None) -> List[str]:
    normalize_feature_profile(feature_profile)
    return [
        "tl_0",
        "tl_1",
        "tl_2",
        "tl_3",
        "tl_4",
        "tl_6",
        "intra_period_pos",
        "ref_distance_1",
        "ref_distance_2",
    ]


def extract_self_features(
    img_u8: np.ndarray,
    block_size: int = 8,
    entropy_bins: int = 32,
    edge_threshold: float = 0.08,
    *,
    feature_profile: str | None = None,
) -> np.ndarray:
    profile = normalize_feature_profile(feature_profile)
    img = img_u8.astype(np.float32) / 255.0

    if profile == FEATURE_PROFILE_BITS:
        stats = _collect_self_stats(
            img,
            block_size=block_size,
            entropy_bins=entropy_bins,
            edge_threshold=edge_threshold,
        )
        feats = [stats[name] for name in self_feature_names(profile)]
        return np.asarray(feats, dtype=np.float32)

    if profile == FEATURE_PROFILE_VMAF:
        s1 = _collect_self_stats(
            img,
            block_size=block_size,
            entropy_bins=entropy_bins,
            edge_threshold=edge_threshold,
        )
        img_s2 = _avg_pool2(img)
        s2 = _collect_self_stats(
            img_s2,
            block_size=max(4, block_size // 2),
            entropy_bins=entropy_bins,
            edge_threshold=edge_threshold,
        )
        feats = [
            s1["mean"],
            s1["std"],
            s1["dark_ratio"],
            s1["bright_ratio"],
            s1["grad_mean"],
            s1["grad_p90"],
            s1["lap_var"],
            s1["edge_density"],
            s1["entropy"],
            s1["block_var_p90"],
            s1["block_var_p99"],
            s1["block_grad_mean"],
            s1["flat_block_ratio"],
            s1["textured_block_ratio"],
            s1["hf_ratio"],
            s2["grad_mean"],
            s2["grad_p90"],
            s2["lap_var"],
            s2["edge_density"],
            s2["entropy"],
            s2["block_var_p90"],
            s2["block_grad_mean"],
            s2["flat_block_ratio"],
            s2["textured_block_ratio"],
            s2["hf_ratio"],
        ]
        return np.asarray(feats, dtype=np.float32)

    stats = _collect_self_stats(
        img,
        block_size=block_size,
        entropy_bins=entropy_bins,
        edge_threshold=edge_threshold,
    )
    feats = [stats[name] for name in self_feature_names(profile)]
    return np.asarray(feats, dtype=np.float32)


def pass1_feature_names(cfg: Optional[dict] = None) -> List[str]:
    if cfg is not None:
        term = str(cfg.get("loss", {}).get("mse_term", "log_mse")).lower().strip()
        if term == "vmaf":
            return [
                "pass1_qp_norm",
                "pass1_log_bits",
                "pass1_vmaf_norm",
                "delta_qp_from_pass1",
            ]
    return [
        "pass1_qp_norm",
        "pass1_log_bits",
        "pass1_log_mse",
        "delta_qp_from_pass1",
    ]


def pair_feature_names(feature_profile: str | None = None) -> List[str]:
    profile = normalize_feature_profile(feature_profile)
    if profile == FEATURE_PROFILE_BITS:
        return [
            "absdiff_mean",
            "absdiff_std",
            "absdiff_p90",
            "absdiff_p99",
            "diff_mse",
            "changed_ratio",
            "grad_diff_mean",
            "grad_diff_std",
            "grad_diff_p90",
            "block_mad_mean",
            "block_mad_std",
            "block_mad_p90",
            "block_mad_p99",
            "diff_hf_ratio",
            "ncc",
            "g_ssim",
        ]
    if profile == FEATURE_PROFILE_VMAF:
        return [
            "absdiff_mean_s1",
            "absdiff_p90_s1",
            "absdiff_p99_s1",
            "grad_diff_mean_s1",
            "grad_diff_p90_s1",
            "block_mad_p90_s1",
            "block_mad_p99_s1",
            "worst5_block_mad_mean_s1",
            "changed_ratio_s1",
            "g_ssim_s1",
            "absdiff_mean_s2",
            "absdiff_p90_s2",
            "grad_diff_mean_s2",
            "grad_diff_p90_s2",
            "block_mad_p90_s2",
            "changed_ratio_s2",
            "g_ssim_s2",
            "ncc",
            "diff_hf_ratio_s1",
        ]
    return [
        "absdiff_mean",
        "absdiff_std",
        "absdiff_p90",
        "diff_mse",
        "changed_ratio",
        "grad_diff_mean",
        "grad_diff_std",
        "block_mad_mean",
        "block_mad_std",
        "block_mad_p90",
        "ncc",
        "g_ssim",
    ]


def global_ssim(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float32) / 255.0
    y = y.astype(np.float32) / 255.0
    return _global_ssim_norm(x, y)


def extract_pair_features(
    cur_u8: np.ndarray,
    ref_u8: np.ndarray,
    block_size: int = 8,
    changed_threshold: float = 0.03,
    *,
    feature_profile: str | None = None,
) -> np.ndarray:
    profile = normalize_feature_profile(feature_profile)
    cur = cur_u8.astype(np.float32) / 255.0
    ref = ref_u8.astype(np.float32) / 255.0

    if profile == FEATURE_PROFILE_BITS:
        stats = _collect_pair_stats(
            cur,
            ref,
            block_size=block_size,
            changed_threshold=changed_threshold,
        )
        feats = [stats[name] for name in pair_feature_names(profile)]
        return np.asarray(feats, dtype=np.float32)

    if profile == FEATURE_PROFILE_VMAF:
        s1 = _collect_pair_stats(
            cur,
            ref,
            block_size=block_size,
            changed_threshold=changed_threshold,
        )
        cur_s2 = _avg_pool2(cur)
        ref_s2 = _avg_pool2(ref)
        s2 = _collect_pair_stats(
            cur_s2,
            ref_s2,
            block_size=max(4, block_size // 2),
            changed_threshold=changed_threshold,
        )
        feats = [
            s1["absdiff_mean"],
            s1["absdiff_p90"],
            s1["absdiff_p99"],
            s1["grad_diff_mean"],
            s1["grad_diff_p90"],
            s1["block_mad_p90"],
            s1["block_mad_p99"],
            s1["worst5_block_mad_mean"],
            s1["changed_ratio"],
            s1["g_ssim"],
            s2["absdiff_mean"],
            s2["absdiff_p90"],
            s2["grad_diff_mean"],
            s2["grad_diff_p90"],
            s2["block_mad_p90"],
            s2["changed_ratio"],
            s2["g_ssim"],
            s1["ncc"],
            s1["diff_hf_ratio"],
        ]
        return np.asarray(feats, dtype=np.float32)

    stats = _collect_pair_stats(
        cur,
        ref,
        block_size=block_size,
        changed_threshold=changed_threshold,
    )
    feats = [stats[name] for name in pair_feature_names(profile)]
    return np.asarray(feats, dtype=np.float32)
