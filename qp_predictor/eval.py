from __future__ import annotations

import argparse
import json

import torch

from .config import (
    ENV_TRAIN_DOUBLE_TARGET,
    ENV_TRAIN_MODEL_MODE,
    ENV_TRAIN_MSE_TERM,
    apply_train_overrides,
    load_config,
    resolve_train_override_cli_env,
)
from .manifest import build_manifest
from .train import build_model, make_dataloaders, evaluate_loader


def main():
    parser = argparse.ArgumentParser(
        description="QP 状态预测评估（须与训练 checkpoint 的 model 结构一致）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
与 train 共用覆盖项，便于与训练脚本相同参数：--model-mode --double-target --mse-term
或环境变量 QP_TRAIN_MODEL_MODE, QP_TRAIN_DOUBLE_TARGET, QP_TRAIN_MSE_TERM
""",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model-mode",
        default=None,
        choices=["single", "double"],
        help="覆盖 model.mode（须与 checkpoint 一致）。",
    )
    parser.add_argument(
        "--double-target",
        default=None,
        choices=["bits", "distortion"],
        dest="double_target",
        help="覆盖 model.double_target（须与 checkpoint 一致）。",
    )
    parser.add_argument(
        "--mse-term",
        default=None,
        metavar="TERM",
        dest="mse_term",
        help="覆盖 loss.mse_term（评估指标计算方式须与训练一致）。",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mm, mm_src = resolve_train_override_cli_env(args.model_mode, ENV_TRAIN_MODEL_MODE)
    dt, dt_src = resolve_train_override_cli_env(args.double_target, ENV_TRAIN_DOUBLE_TARGET)
    mt, mt_src = resolve_train_override_cli_env(args.mse_term, ENV_TRAIN_MSE_TERM)
    apply_train_overrides(
        cfg,
        model_mode=mm,
        double_target=dt,
        mse_term=mt,
        model_mode_src=mm_src,
        double_target_src=dt_src,
        mse_term_src=mt_src,
    )
    manifest = build_manifest(cfg)

    model = build_model(cfg, phase=args.phase)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)

    _, _, test_loader = make_dataloaders(manifest, cfg, phase=args.phase)
    metrics = evaluate_loader(model, test_loader, device, cfg, phase=args.phase)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
