from __future__ import annotations

from typing import List

import numpy as np


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


def self_feature_names() -> List[str]:
    return [
        "mean", "std", "min", "max", "p10", "p50", "p90",
        "grad_mean", "grad_std", "grad_p90",
        "lap_var", "edge_density", "entropy",
        "block_var_mean", "block_var_std", "block_var_p90",
        "block_grad_mean", "block_grad_std",
        "flat_block_ratio", "textured_block_ratio",
        "hf_ratio",
    ]


def extract_self_features(img_u8: np.ndarray, block_size: int = 8, entropy_bins: int = 32, edge_threshold: float = 0.08) -> np.ndarray:
    img = img_u8.astype(np.float32) / 255.0
    grad = _grad_mag(img)
    lap = _laplacian(img)
    blocks = _block_view(img, block_size)
    bvar = blocks.var(axis=(2, 3))
    gblocks = _block_view(grad, block_size)
    bg = gblocks.mean(axis=(2, 3))

    feats = [
        float(img.mean()),
        float(img.std()),
        float(img.min()),
        float(img.max()),
        float(np.percentile(img, 10)),
        float(np.percentile(img, 50)),
        float(np.percentile(img, 90)),
        float(grad.mean()),
        float(grad.std()),
        float(np.percentile(grad, 90)),
        float(lap.var()),
        float((grad > edge_threshold).mean()),
        _safe_hist_entropy(img, bins=entropy_bins),
        float(bvar.mean()),
        float(bvar.std()),
        float(np.percentile(bvar, 90)),
        float(bg.mean()),
        float(bg.std()),
        float((bvar < 0.002).mean()),
        float((bvar > 0.02).mean()),
        _hf_ratio_fft(img),
    ]
    return np.asarray(feats, dtype=np.float32)


def pass1_feature_names() -> List[str]:
    return [
        "pass1_qp_norm",
        "pass1_log_bits",
        "pass1_log_mse",
        "delta_qp_from_pass1",
    ]


def pair_feature_names() -> List[str]:
    return [
        "absdiff_mean", "absdiff_std", "absdiff_p90",
        "diff_mse", "changed_ratio",
        "grad_diff_mean", "grad_diff_std",
        "block_mad_mean", "block_mad_std", "block_mad_p90",
        "ncc", "g_ssim",
    ]


def global_ssim(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float32) / 255.0
    y = y.astype(np.float32) / 255.0
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


def extract_pair_features(cur_u8: np.ndarray, ref_u8: np.ndarray, block_size: int = 8, changed_threshold: float = 0.03) -> np.ndarray:
    cur = cur_u8.astype(np.float32) / 255.0
    ref = ref_u8.astype(np.float32) / 255.0
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

    feats = [
        float(adiff.mean()),
        float(adiff.std()),
        float(np.percentile(adiff, 90)),
        float((diff ** 2).mean()),
        float((adiff > changed_threshold).mean()),
        float(gdiff.mean()),
        float(gdiff.std()),
        float(bmad.mean()),
        float(bmad.std()),
        float(np.percentile(bmad, 90)),
        ncc,
        global_ssim(cur_u8, ref_u8),
    ]
    return np.asarray(feats, dtype=np.float32)
