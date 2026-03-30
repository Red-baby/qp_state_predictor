from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from qp_predictor.config import load_config
    from qp_predictor.features import meta_feature_names, pair_feature_names, self_feature_names
    from qp_predictor.train import build_model, model_head_out_dim
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", None) or str(exc)
    raise SystemExit(
        f"缺少导出脚本依赖: {missing}。请先在训练环境安装 requirements.txt 后再运行。"
    ) from exc


HEADER_GUARD = "QAV1_CHUNK_NN_MODELS_GENERATED_H_"
MODEL_API_VERSION = 1


@dataclass
class LinearLayerExport:
    weight: torch.Tensor
    bias: torch.Tensor


@dataclass
class MLPExport:
    name: str
    layers: list[LinearLayerExport]

    @property
    def in_dim(self) -> int:
        return int(self.layers[0].weight.shape[1])

    @property
    def out_dim(self) -> int:
        return int(self.layers[-1].weight.shape[0])

    @property
    def hidden_dims(self) -> list[int]:
        if len(self.layers) <= 1:
            return []
        return [int(layer.weight.shape[0]) for layer in self.layers[:-1]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出 qav1 chunk predictor 所需的 Phase1/Phase2 权重头文件",
    )
    parser.add_argument("--phase1-config", type=Path, default=REPO_ROOT / "phase1_4_bits_mse.yaml")
    parser.add_argument("--phase1-checkpoint", type=Path, required=True)
    parser.add_argument("--phase2-vmaf-config", type=Path, default=REPO_ROOT / "phase2_2_bits_vmaf.yaml")
    parser.add_argument("--phase2-vmaf-checkpoint", type=Path, required=True)
    parser.add_argument("--phase1-psnr-config", type=Path, default=REPO_ROOT / "phase1_psnr_direct.yaml")
    parser.add_argument("--phase1-psnr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-header", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    return parser.parse_args()


def _load_checkpoint(checkpoint_path: Path) -> dict:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"checkpoint 格式不正确，缺少 model_state: {checkpoint_path}")
    return ckpt


def _build_eval_model(cfg_path: Path, checkpoint_path: Path, phase: int) -> tuple[dict, nn.Module]:
    cfg = load_config(str(cfg_path))
    model = build_model(cfg, phase=phase)
    ckpt = _load_checkpoint(checkpoint_path)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return cfg, model


def _collect_linear_layers(module: nn.Module) -> list[LinearLayerExport]:
    if isinstance(module, nn.Linear):
        return [
            LinearLayerExport(
                weight=module.weight.detach().cpu().to(dtype=torch.float32).contiguous(),
                bias=module.bias.detach().cpu().to(dtype=torch.float32).contiguous(),
            )
        ]

    seq = getattr(module, "net", None)
    if seq is None:
        raise TypeError(f"模块 {module.__class__.__name__} 不是可导出的 MLP/Linear")

    layers: list[LinearLayerExport] = []
    for sub in seq:
        if isinstance(sub, nn.Linear):
            layers.append(
                LinearLayerExport(
                    weight=sub.weight.detach().cpu().to(dtype=torch.float32).contiguous(),
                    bias=sub.bias.detach().cpu().to(dtype=torch.float32).contiguous(),
                )
            )
        elif isinstance(sub, (nn.ReLU, nn.Dropout)):
            continue
        else:
            raise TypeError(f"模块 {module.__class__.__name__} 内含不支持层: {sub.__class__.__name__}")
    if not layers:
        raise ValueError(f"模块 {module.__class__.__name__} 未发现 Linear 层")
    return layers


def _export_mlp(name: str, module: nn.Module) -> MLPExport:
    return MLPExport(name=name, layers=_collect_linear_layers(module))


def _fmt_float(v: float) -> str:
    return format(float(v), ".9g") + "f"


def _flatten_weight(weight: torch.Tensor) -> list[float]:
    return [float(v) for v in weight.reshape(-1).tolist()]


def _flatten_bias(bias: torch.Tensor) -> list[float]:
    return [float(v) for v in bias.reshape(-1).tolist()]


def _emit_float_array(lines: list[str], name: str, values: Iterable[float]) -> None:
    vals = list(values)
    lines.append(f"static const float {name}[{len(vals)}] = {{")
    if vals:
        for start in range(0, len(vals), 8):
            chunk = ", ".join(_fmt_float(v) for v in vals[start:start + 8])
            lines.append(f"    {chunk},")
    lines.append("};")
    lines.append("")


def _emit_nnconfig(lines: list[str], prefix: str, mlp: MLPExport) -> None:
    for layer_idx, layer in enumerate(mlp.layers):
        _emit_float_array(lines, f"{prefix}_w_{layer_idx}", _flatten_weight(layer.weight))
        _emit_float_array(lines, f"{prefix}_b_{layer_idx}", _flatten_bias(layer.bias))

    hidden_dims = mlp.hidden_dims
    hidden_dim_list = ", ".join(str(v) for v in hidden_dims) if hidden_dims else "0"
    lines.append(f"static const nnconfig_t {prefix} = {{")
    lines.append(f"    {mlp.in_dim},")
    lines.append(f"    {mlp.out_dim},")
    lines.append(f"    {len(hidden_dims)},")
    lines.append("    {")
    if hidden_dims:
        lines.append(f"        {hidden_dim_list},")
    else:
        lines.append("        0,")
    lines.append("    },")
    lines.append("    {")
    for layer_idx in range(len(mlp.layers)):
        lines.append(f"        {prefix}_w_{layer_idx},")
    lines.append("    },")
    lines.append("    {")
    for layer_idx in range(len(mlp.layers)):
        lines.append(f"        {prefix}_b_{layer_idx},")
    lines.append("    },")
    lines.append("};")
    lines.append("")


def _phase2_branch_exports(model: nn.Module, prefix: str) -> dict[str, MLPExport]:
    branch = getattr(model, "dist_branch", None)
    if branch is None:
        raise TypeError(f"模型 {model.__class__.__name__} 不含 dist_branch")
    return {
        "current_encoder": _export_mlp(f"{prefix}_current_encoder", branch.current_encoder),
        "ref_encoder": _export_mlp(f"{prefix}_ref_encoder", branch.ref_encoder),
        "edge_encoder": _export_mlp(f"{prefix}_edge_encoder", branch.edge_encoder),
        "edge_gate": _export_mlp(f"{prefix}_edge_gate", branch.edge_gate),
        "context_gate": _export_mlp(f"{prefix}_context_gate", branch.context_gate),
        "trunk": _export_mlp(f"{prefix}_trunk", branch.trunk),
        "head": _export_mlp(f"{prefix}_head", branch.main_head),
    }


def _feature_meta(
    cfg_bits: dict,
    cfg_vmaf: dict,
    cfg_psnr: dict,
    phase1_bits: MLPExport,
    phase2_vmaf: dict[str, MLPExport],
    phase1_psnr: MLPExport,
) -> dict:
    self_dim = len(self_feature_names("legacy"))
    pair_dim = len(pair_feature_names("legacy"))
    meta_dim = len(meta_feature_names("legacy"))
    return {
        "api_version": MODEL_API_VERSION,
        "resize_width": int(cfg_bits["data"]["resize_width"]),
        "resize_height": int(cfg_bits["data"]["resize_height"]),
        "self_dim": self_dim,
        "pair_dim": pair_dim,
        "meta_dim": meta_dim,
        "phase1_pass1_dim": int(phase1_bits.in_dim - self_dim - meta_dim - 1),
        "phase1_psnr_pass1_dim": int(phase1_psnr.in_dim - self_dim - meta_dim - 1),
        "phase2_vmaf_pass1_dim": int(phase2_vmaf["current_encoder"].in_dim - self_dim - meta_dim - 1),
        "qp_norm_min": float(cfg_bits["data"]["qp_norm_min"]),
        "qp_norm_max": float(cfg_bits["data"]["qp_norm_max"]),
        "pass1_psnr_norm_div": float(cfg_psnr["data"].get("pass1_psnr_norm_div", 100.0)),
        "pass1_vmaf_norm_div": float(cfg_vmaf["data"].get("pass1_vmaf_norm_div", 100.0)),
        "head_out_dim_phase1": int(model_head_out_dim(cfg_bits)),
    }


def _validate_cfgs(cfg_bits: dict, cfg_vmaf: dict, cfg_psnr: dict) -> None:
    expected = (
        (cfg_bits["data"]["resize_width"], cfg_bits["data"]["resize_height"]),
        (cfg_vmaf["data"]["resize_width"], cfg_vmaf["data"]["resize_height"]),
        (cfg_psnr["data"]["resize_width"], cfg_psnr["data"]["resize_height"]),
    )
    if len(set(expected)) != 1:
        raise ValueError(f"三套配置的 resize 参数不一致: {expected}")

    expected_qp = (
        (cfg_bits["data"]["qp_norm_min"], cfg_bits["data"]["qp_norm_max"]),
        (cfg_vmaf["data"]["qp_norm_min"], cfg_vmaf["data"]["qp_norm_max"]),
        (cfg_psnr["data"]["qp_norm_min"], cfg_psnr["data"]["qp_norm_max"]),
    )
    if len(set(expected_qp)) != 1:
        raise ValueError(f"三套配置的 qp 归一化区间不一致: {expected_qp}")


def _write_header(
    output_header: Path,
    meta: dict,
    phase1_bits: MLPExport,
    phase2_vmaf: dict[str, MLPExport],
    phase1_psnr: MLPExport,
) -> None:
    lines: list[str] = []
    lines.append("// Auto-generated by scripts/export_qav1_chunk_nn.py. Do not edit.")
    lines.append(f"#ifndef {HEADER_GUARD}")
    lines.append(f"#define {HEADER_GUARD}")
    lines.append("")
    lines.append("// 此头文件要求在已定义 nnconfig_t 之后包含。")
    lines.append("")
    lines.append("typedef struct qav1_chunk_phase2_branch_model_t {")
    lines.append("    const nnconfig_t *current_encoder;")
    lines.append("    const nnconfig_t *ref_encoder;")
    lines.append("    const nnconfig_t *edge_encoder;")
    lines.append("    const nnconfig_t *edge_gate;")
    lines.append("    const nnconfig_t *context_gate;")
    lines.append("    const nnconfig_t *trunk;")
    lines.append("    const nnconfig_t *head;")
    lines.append("} qav1_chunk_phase2_branch_model_t;")
    lines.append("")
    lines.append("typedef struct qav1_chunk_nn_model_bundle_t {")
    lines.append("    int api_version;")
    lines.append("    int has_weights;")
    lines.append("    int resize_width;")
    lines.append("    int resize_height;")
    lines.append("    int self_dim;")
    lines.append("    int pair_dim;")
    lines.append("    int meta_dim;")
    lines.append("    int phase1_pass1_dim;")
    lines.append("    int phase1_psnr_pass1_dim;")
    lines.append("    int phase2_vmaf_pass1_dim;")
    lines.append("    float qp_norm_min;")
    lines.append("    float qp_norm_max;")
    lines.append("    float pass1_psnr_norm_div;")
    lines.append("    float pass1_vmaf_norm_div;")
    lines.append("    const nnconfig_t *phase1_bits;")
    lines.append("    const nnconfig_t *phase1_psnr;")
    lines.append("    qav1_chunk_phase2_branch_model_t phase2_vmaf;")
    lines.append("} qav1_chunk_nn_model_bundle_t;")
    lines.append("")

    _emit_nnconfig(lines, "qav1_chunk_phase1_bits", phase1_bits)
    _emit_nnconfig(lines, "qav1_chunk_phase1_psnr", phase1_psnr)
    for name, export in phase2_vmaf.items():
        _emit_nnconfig(lines, f"qav1_chunk_phase2_vmaf_{name}", export)

    lines.append("static const qav1_chunk_nn_model_bundle_t qav1_chunk_nn_models = {")
    lines.append(f"    {meta['api_version']},")
    lines.append("    1,")
    lines.append(f"    {meta['resize_width']},")
    lines.append(f"    {meta['resize_height']},")
    lines.append(f"    {meta['self_dim']},")
    lines.append(f"    {meta['pair_dim']},")
    lines.append(f"    {meta['meta_dim']},")
    lines.append(f"    {meta['phase1_pass1_dim']},")
    lines.append(f"    {meta['phase1_psnr_pass1_dim']},")
    lines.append(f"    {meta['phase2_vmaf_pass1_dim']},")
    lines.append(f"    {_fmt_float(meta['qp_norm_min'])},")
    lines.append(f"    {_fmt_float(meta['qp_norm_max'])},")
    lines.append(f"    {_fmt_float(meta['pass1_psnr_norm_div'])},")
    lines.append(f"    {_fmt_float(meta['pass1_vmaf_norm_div'])},")
    lines.append("    &qav1_chunk_phase1_bits,")
    lines.append("    &qav1_chunk_phase1_psnr,")
    lines.append("    {")
    lines.append("        &qav1_chunk_phase2_vmaf_current_encoder,")
    lines.append("        &qav1_chunk_phase2_vmaf_ref_encoder,")
    lines.append("        &qav1_chunk_phase2_vmaf_edge_encoder,")
    lines.append("        &qav1_chunk_phase2_vmaf_edge_gate,")
    lines.append("        &qav1_chunk_phase2_vmaf_context_gate,")
    lines.append("        &qav1_chunk_phase2_vmaf_trunk,")
    lines.append("        &qav1_chunk_phase2_vmaf_head,")
    lines.append("    },")
    lines.append("};")
    lines.append("")
    lines.append(f"#endif  // {HEADER_GUARD}")
    lines.append("")

    output_header.parent.mkdir(parents=True, exist_ok=True)
    output_header.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()

    cfg_bits, model_bits = _build_eval_model(args.phase1_config, args.phase1_checkpoint, phase=1)
    cfg_vmaf, model_vmaf = _build_eval_model(args.phase2_vmaf_config, args.phase2_vmaf_checkpoint, phase=2)
    cfg_psnr, model_psnr = _build_eval_model(args.phase1_psnr_config, args.phase1_psnr_checkpoint, phase=1)

    _validate_cfgs(cfg_bits, cfg_vmaf, cfg_psnr)

    phase1_bits = _export_mlp("phase1_bits", model_bits.net)
    phase1_psnr = _export_mlp("phase1_psnr", model_psnr.net)
    phase2_vmaf = _phase2_branch_exports(model_vmaf, "phase2_vmaf")

    meta = _feature_meta(cfg_bits, cfg_vmaf, cfg_psnr, phase1_bits, phase2_vmaf, phase1_psnr)
    meta["phase1"] = {
        "config": str(args.phase1_config),
        "checkpoint": str(args.phase1_checkpoint),
        "in_dim": phase1_bits.in_dim,
        "hidden_dims": phase1_bits.hidden_dims,
        "out_dim": phase1_bits.out_dim,
    }
    meta["phase1_psnr"] = {
        "config": str(args.phase1_psnr_config),
        "checkpoint": str(args.phase1_psnr_checkpoint),
        "in_dim": phase1_psnr.in_dim,
        "hidden_dims": phase1_psnr.hidden_dims,
        "out_dim": phase1_psnr.out_dim,
    }
    meta["phase2_vmaf"] = {
        "config": str(args.phase2_vmaf_config),
        "checkpoint": str(args.phase2_vmaf_checkpoint),
        "modules": {
            name: {
                "in_dim": export.in_dim,
                "hidden_dims": export.hidden_dims,
                "out_dim": export.out_dim,
            }
            for name, export in phase2_vmaf.items()
        },
    }

    _write_header(args.output_header, meta, phase1_bits, phase2_vmaf, phase1_psnr)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] header -> {args.output_header}")
    print(f"[ok] metadata -> {args.output_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
