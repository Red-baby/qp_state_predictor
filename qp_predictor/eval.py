from __future__ import annotations

import argparse
import json

import torch

from .config import load_config
from .manifest import build_manifest
from .train import build_model, make_dataloaders, evaluate_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
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
