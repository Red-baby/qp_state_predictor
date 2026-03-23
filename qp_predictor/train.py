from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .datasets import FrameDataset, SegmentDataset
from .features import pair_feature_names, self_feature_names
from .manifest import build_manifest
from .models import Phase1Net, Phase2Net, Phase3Net
from .utils import (
    ensure_dir,
    huber_loss_masked,
    inverse_log_bits,
    inverse_log_mse,
    regression_metrics,
    save_json,
    set_seed,
    to_device,
    train_val_test_split,
)


def split_manifest(manifest, cfg):
    split_by_col = cfg["data"]["split_by_col"]
    train_keys, val_keys, test_keys = train_val_test_split(
        manifest[split_by_col].unique().tolist(),
        cfg["data"]["train_ratio"],
        cfg["data"]["val_ratio"],
        cfg["data"]["test_ratio"],
        cfg["seed"],
    )
    train_df = manifest[manifest[split_by_col].isin(train_keys)].reset_index(drop=True)
    val_df = manifest[manifest[split_by_col].isin(val_keys)].reset_index(drop=True)
    test_df = manifest[manifest[split_by_col].isin(test_keys)].reset_index(drop=True)
    return train_df, val_df, test_df


def build_model(cfg: dict, phase: int):
    self_dim = len(self_feature_names())
    pair_dim = len(pair_feature_names())
    meta_dim = 12

    if phase == 1:
        return Phase1Net(self_dim=self_dim, meta_dim=meta_dim, cfg=cfg)
    if phase == 2:
        return Phase2Net(self_dim=self_dim, pair_dim=pair_dim, meta_dim=meta_dim, cfg=cfg)
    if phase == 3:
        return Phase3Net(self_dim=self_dim, pair_dim=pair_dim, meta_dim=meta_dim, cfg=cfg)
    raise ValueError(f"Unsupported phase: {phase}")


def make_dataloaders(manifest, cfg, phase: int):
    train_df, val_df, test_df = split_manifest(manifest, cfg)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    if phase in (1, 2):
        train_ds = FrameDataset(manifest, cfg, train_df, phase=phase)
        val_ds = FrameDataset(manifest, cfg, val_df, phase=phase)
        test_ds = FrameDataset(manifest, cfg, test_df, phase=phase)
        batch_size = int(train_cfg[f"batch_size_phase{phase}"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=int(train_cfg["num_workers"]), pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=int(train_cfg["num_workers"]), pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=int(train_cfg["num_workers"]), pin_memory=True, drop_last=False)
        return train_loader, val_loader, test_loader

    if phase == 3:
        train_ds = SegmentDataset(manifest, cfg, train_df)
        val_ds = SegmentDataset(manifest, cfg, val_df)
        test_ds = SegmentDataset(manifest, cfg, test_df)
        batch_size = int(train_cfg["batch_size_phase3"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=int(train_cfg["num_workers"]), pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=int(train_cfg["num_workers"]), pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=int(train_cfg["num_workers"]), pin_memory=True, drop_last=False)
        return train_loader, val_loader, test_loader

    raise ValueError(f"Unsupported phase: {phase}")


def compute_loss(batch, outputs, cfg, phase: int):
    delta = float(cfg["train"]["huber_delta"])
    bits_w = float(cfg["loss"]["bits_weight"])
    mse_w = float(cfg["loss"]["mse_weight"])
    aux_w = float(cfg["loss"]["aux_weight"])

    if phase in (1, 2):
        pred = outputs["pred"]
        target = batch["target"]
        mask = batch["valid_mask"]
        loss_bits = huber_loss_masked(pred[..., 0:1], target[..., 0:1], mask, delta)
        loss_mse = huber_loss_masked(pred[..., 1:2], target[..., 1:2], mask, delta)
        loss = bits_w * loss_bits + mse_w * loss_mse
        return loss, {
            "loss": float(loss.item()),
            "loss_bits": float(loss_bits.item()),
            "loss_mse": float(loss_mse.item()),
        }

    if phase == 3:
        pred = outputs["pred"]
        aux = outputs["aux_pred"]
        target = batch["targets"]
        mask = batch["valid_loss_mask"]

        loss_bits = huber_loss_masked(pred[..., 0:1], target[..., 0:1], mask, delta)
        loss_mse = huber_loss_masked(pred[..., 1:2], target[..., 1:2], mask, delta)
        loss_aux_bits = huber_loss_masked(aux[..., 0:1], target[..., 0:1], mask, delta)
        loss_aux_mse = huber_loss_masked(aux[..., 1:2], target[..., 1:2], mask, delta)

        loss_main = bits_w * loss_bits + mse_w * loss_mse
        loss_aux = bits_w * loss_aux_bits + mse_w * loss_aux_mse
        loss = loss_main + aux_w * loss_aux
        return loss, {
            "loss": float(loss.item()),
            "loss_bits": float(loss_bits.item()),
            "loss_mse": float(loss_mse.item()),
            "loss_aux_bits": float(loss_aux_bits.item()),
            "loss_aux_mse": float(loss_aux_mse.item()),
        }

    raise ValueError(f"Unsupported phase: {phase}")


def _group_metrics(bits_true, bits_pred, mse_true, mse_pred, psnr_true, psnr_pred, group_ids, label_fn):
    out = {}
    if len(group_ids) == 0:
        return out

    for gid in sorted(set(group_ids.tolist())):
        mask = group_ids == gid
        if not np.any(mask):
            continue
        out[label_fn(int(gid))] = {
            "count": int(mask.sum()),
            "bits": regression_metrics(bits_true[mask], bits_pred[mask]),
            "mse": regression_metrics(mse_true[mask], mse_pred[mask]),
            "psnr": regression_metrics(psnr_true[mask], psnr_pred[mask]),
        }
    return out


@torch.no_grad()
def evaluate_loader(model, loader, device, cfg, phase: int):
    model.eval()
    loss_meter = []
    bits_true_all = []
    bits_pred_all = []
    mse_true_all = []
    mse_pred_all = []
    frame_type_all = []
    temporal_layer_all = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = to_device(batch, device)
        outputs = model(batch)
        _, logs = compute_loss(batch, outputs, cfg, phase)
        loss_meter.append(logs["loss"])

        if phase in (1, 2):
            pred = outputs["pred"]
            target = batch["target"]
            valid_mask = batch["valid_mask"] > 0.5
            frame_type_ids = batch["frame_type_id"]
            temporal_layers = batch["temporal_layer"]
        else:
            pred = outputs["pred"]
            target = batch["targets"]
            valid_mask = batch["valid_loss_mask"] > 0.5
            frame_type_ids = batch["frame_type_ids"]
            temporal_layers = batch["temporal_layers"]

        pred = pred[valid_mask]
        target = target[valid_mask]
        frame_type_ids = frame_type_ids[valid_mask].detach().cpu().numpy()
        temporal_layers = temporal_layers[valid_mask].detach().cpu().numpy()

        bits_pred = inverse_log_bits(pred[:, 0]).detach().cpu().numpy()
        bits_true = inverse_log_bits(target[:, 0]).detach().cpu().numpy()
        mse_pred = inverse_log_mse(pred[:, 1]).detach().cpu().numpy()
        mse_true = inverse_log_mse(target[:, 1]).detach().cpu().numpy()

        bits_true_all.append(bits_true)
        bits_pred_all.append(bits_pred)
        mse_true_all.append(mse_true)
        mse_pred_all.append(mse_pred)
        frame_type_all.append(frame_type_ids)
        temporal_layer_all.append(temporal_layers)

    bits_true_all = np.concatenate(bits_true_all, axis=0) if bits_true_all else np.zeros((0,))
    bits_pred_all = np.concatenate(bits_pred_all, axis=0) if bits_pred_all else np.zeros((0,))
    mse_true_all = np.concatenate(mse_true_all, axis=0) if mse_true_all else np.zeros((0,))
    mse_pred_all = np.concatenate(mse_pred_all, axis=0) if mse_pred_all else np.zeros((0,))
    frame_type_all = np.concatenate(frame_type_all, axis=0) if frame_type_all else np.zeros((0,), dtype=np.int64)
    temporal_layer_all = np.concatenate(temporal_layer_all, axis=0) if temporal_layer_all else np.zeros((0,), dtype=np.int64)

    from .utils import compute_psnr_from_mse
    psnr_true = compute_psnr_from_mse(mse_true_all, max_value=float(cfg["eval"]["max_psnr_value"]))
    psnr_pred = compute_psnr_from_mse(mse_pred_all, max_value=float(cfg["eval"]["max_psnr_value"]))

    metrics = {
        "loss": float(np.mean(loss_meter)) if loss_meter else 0.0,
        "bits": regression_metrics(bits_true_all, bits_pred_all) if len(bits_true_all) > 0 else {},
        "mse": regression_metrics(mse_true_all, mse_pred_all) if len(mse_true_all) > 0 else {},
        "psnr": regression_metrics(psnr_true, psnr_pred) if len(psnr_true) > 0 else {},
        "by_frame_type": _group_metrics(
            bits_true_all,
            bits_pred_all,
            mse_true_all,
            mse_pred_all,
            psnr_true,
            psnr_pred,
            frame_type_all,
            label_fn=lambda x: {0: "I", 1: "P", 2: "B", 3: "UNK"}.get(x, f"UNK_{x}"),
        ),
        "by_temporal_layer": _group_metrics(
            bits_true_all,
            bits_pred_all,
            mse_true_all,
            mse_pred_all,
            psnr_true,
            psnr_pred,
            temporal_layer_all,
            label_fn=lambda x: str(x),
        ),
    }
    return metrics


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, phase: int):
    model.train()
    train_cfg = cfg["train"]
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"

    meter = []
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            outputs = model(batch)
            loss, _ = compute_loss(batch, outputs, cfg, phase)

        if amp_enabled:
            scaler.scale(loss).backward()
            if float(train_cfg["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if float(train_cfg["grad_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["grad_clip"]))
            optimizer.step()

        meter.append(float(loss.item()))
        pbar.set_postfix(loss=f"{np.mean(meter):.4f}")

    return {"loss": float(np.mean(meter)) if meter else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    manifest = build_manifest(cfg)
    model = build_model(cfg, phase=args.phase)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader, test_loader = make_dataloaders(manifest, cfg, phase=args.phase)

    output_dir = Path(cfg["data"]["output_root"]) / f"phase{args.phase}"
    ensure_dir(output_dir)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    best_val = float("inf")
    history = {"train": [], "val": [], "test_best": None}

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        print(f"\n===== Phase {args.phase} | Epoch {epoch}/{cfg['train']['epochs']} =====")
        train_log = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, phase=args.phase)
        val_metrics = evaluate_loader(model, val_loader, device, cfg, phase=args.phase)

        history["train"].append({"epoch": epoch, **train_log})
        history["val"].append({"epoch": epoch, **val_metrics})

        print("train:", json.dumps(train_log, indent=2, ensure_ascii=False))
        print("val:", json.dumps(val_metrics, indent=2, ensure_ascii=False))

        ckpt = {
            "epoch": epoch,
            "phase": args.phase,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(ckpt, output_dir / "best.pt")
            test_metrics = evaluate_loader(model, test_loader, device, cfg, phase=args.phase)
            history["test_best"] = {"epoch": epoch, **test_metrics}
            print("test(best):", json.dumps(test_metrics, indent=2, ensure_ascii=False))

        save_json(history, output_dir / "history.json")

    print(f"\nTraining done. Best val loss = {best_val:.6f}")


if __name__ == "__main__":
    main()
