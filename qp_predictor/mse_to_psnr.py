"""
从 MSE 计算 PSNR，与 ``qp_predictor.utils.compute_psnr_from_mse`` 一致::

    PSNR = 10 * log10(max_value^2 / mse)

其中 ``mse`` 会先被 clip 到不小于 1e-8（与训练/评估代码相同）。

用法::

    cd /path/to/qp_state_predictor
    python -m qp_predictor.mse_to_psnr 16.67
    python -m qp_predictor.mse_to_psnr 16.67 0.1 --max-value 255
"""

from __future__ import annotations

import argparse

import numpy as np

from .utils import compute_psnr_from_mse


def main() -> None:
    parser = argparse.ArgumentParser(description="MSE -> PSNR（与项目内 compute_psnr_from_mse 一致）")
    parser.add_argument(
        "mse",
        type=float,
        nargs="+",
        help="一个或多个 MSE 值（正数）",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=255.0,
        help="像素动态范围上限（与 eval.max_psnr_value / 默认一致），默认 255",
    )
    args = parser.parse_args()

    out = compute_psnr_from_mse(np.asarray(args.mse, dtype=np.float64), max_value=float(args.max_value))
    arr = np.atleast_1d(out)

    for m, p in zip(args.mse, arr.flatten()):
        print(f"mse={m:g}  ->  psnr={float(p):.6f} dB  (max_value={args.max_value:g})")


if __name__ == "__main__":
    main()
