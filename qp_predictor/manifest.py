from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .graph import build_segment_local_template
from .utils import qp_norm_span


def _validate_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")


def _validate_unique_group_poc(df: pd.DataFrame, group_cols: List[str], poc_col: str) -> None:
    key_cols = list(group_cols) + [poc_col]
    dup_mask = df.duplicated(key_cols, keep=False)
    if not dup_mask.any():
        return

    sample = df.loc[dup_mask, key_cols].head(5).to_dict("records")
    raise ValueError(
        "CSV contains duplicated rows for the same group_cols + poc. "
        "This will corrupt segment topology. "
        f"Please add an encode/run identifier to group_cols. Examples: {sample}"
    )


def _build_segment_templates(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    i_interval = int(data_cfg["i_interval"])
    gop_size = int(data_cfg["gop_size"])
    tail_hier_min = int(data_cfg.get("tail_hier_min", 4))

    tables = []
    for segment_uid, seg_df in df.groupby("segment_uid", sort=False):
        last_local = int(seg_df["local_poc"].max())
        tpl = build_segment_local_template(
            last_local_poc=last_local,
            i_interval=i_interval,
            gop_size=gop_size,
            tail_hier_min=tail_hier_min,
        ).rename(
            columns={
                "frame_type": "template_frame_type",
                "temporal_layer": "template_temporal_layer",
                "ref_local_1": "template_ref_local_1",
                "ref_local_2": "template_ref_local_2",
                "num_refs": "template_num_refs",
                "valid_train": "template_valid_train",
            }
        )
        tpl["segment_uid"] = segment_uid
        tpl["segment_last_local"] = last_local
        tpl["segment_span"] = last_local + 1
        tables.append(tpl)

    if not tables:
        return pd.DataFrame(
            columns=[
                "segment_uid",
                "local_poc",
                "segment_last_local",
                "segment_span",
                "template_frame_type",
                "template_temporal_layer",
                "template_ref_local_1",
                "template_ref_local_2",
                "template_num_refs",
                "template_valid_train",
            ]
        )
    return pd.concat(tables, ignore_index=True)


def build_manifest(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["labels_csv"])

    seq_col = data_cfg["sequence_col"]
    poc_col = data_cfg["poc_col"]
    qp_col = data_cfg["qp_col"]
    bits_col = data_cfg["bits_col"]
    psnr_col = data_cfg["psnr_col"]
    mse_col = data_cfg["mse_col"]
    group_cols = list(data_cfg["group_cols"])
    yuv_sequence_col = data_cfg["yuv_sequence_col"]

    required = list(set(group_cols + [seq_col, yuv_sequence_col, poc_col, qp_col, bits_col, psnr_col, mse_col]))
    _validate_columns(df, required)
    _validate_unique_group_poc(df, group_cols, poc_col)

    df = df.copy()
    df["row_id"] = range(len(df))
    df["segment_id"] = df[poc_col] // int(data_cfg["i_interval"])
    df["local_poc"] = df[poc_col] % int(data_cfg["i_interval"])

    seg_cols = [c for c in group_cols] + ["segment_id"]
    df["segment_uid"] = df[seg_cols].astype(str).agg("|".join, axis=1)
    segment_tpl = _build_segment_templates(df, cfg)
    df = df.merge(segment_tpl, on=["segment_uid", "local_poc"], how="left")

    explicit = data_cfg["explicit_ref_columns"]
    explicit_frame_type = explicit.get("frame_type")
    explicit_layer = explicit.get("temporal_layer")
    explicit_ref1 = explicit.get("ref_poc_1")
    explicit_ref2 = explicit.get("ref_poc_2")
    explicit_ref_qp1 = explicit.get("ref_qp_1")
    explicit_ref_qp2 = explicit.get("ref_qp_2")
    intra_period_pos_col = data_cfg.get("intra_period_pos_col")

    if explicit_frame_type and explicit_frame_type in df.columns:
        df["frame_type"] = df[explicit_frame_type].astype(str)
    if explicit_layer and explicit_layer in df.columns:
        df["temporal_layer"] = df[explicit_layer].astype(int)
    if explicit_ref1 and explicit_ref1 in df.columns:
        df["ref_poc_1"] = df[explicit_ref1].fillna(-1).astype(int)
    if explicit_ref2 and explicit_ref2 in df.columns:
        df["ref_poc_2"] = df[explicit_ref2].fillna(-1).astype(int)
    if explicit_ref_qp1 and explicit_ref_qp1 in df.columns:
        df["ref_qp_1"] = df[explicit_ref_qp1].astype(float)
    if explicit_ref_qp2 and explicit_ref_qp2 in df.columns:
        df["ref_qp_2"] = df[explicit_ref_qp2].astype(float)

    if "frame_type" not in df.columns:
        df["frame_type"] = df["template_frame_type"].astype(str)
    if "temporal_layer" not in df.columns:
        df["temporal_layer"] = df["template_temporal_layer"].fillna(-1).astype(int)

    if "ref_poc_1" not in df.columns or "ref_poc_2" not in df.columns:
        if not data_cfg["infer_refs_if_missing"]:
            raise ValueError("Reference columns missing and infer_refs_if_missing=False")
        segment_start = (df[poc_col] - df["local_poc"]).astype(int)
        if "ref_poc_1" not in df.columns:
            ref_local_1 = df["template_ref_local_1"].fillna(-1).astype(int)
            df["ref_poc_1"] = np.where(ref_local_1 >= 0, segment_start + ref_local_1, -1).astype(int)
        if "ref_poc_2" not in df.columns:
            ref_local_2 = df["template_ref_local_2"].fillna(-1).astype(int)
            df["ref_poc_2"] = np.where(ref_local_2 >= 0, segment_start + ref_local_2, -1).astype(int)

    df["valid_train"] = df["template_valid_train"].fillna(0).astype(int)

    df["ref_distance_1"] = (df[poc_col] - df["ref_poc_1"]).where(df["ref_poc_1"] >= 0, -1).astype(int)
    df["ref_distance_2"] = (df["ref_poc_2"] - df[poc_col]).where(df["ref_poc_2"] >= 0, -1).astype(int)

    base_cols = [c for c in group_cols] + [poc_col]
    df["base_uid"] = df[base_cols].astype(str).agg("|".join, axis=1)

    segment_keys = set(zip(df["segment_uid"], df[poc_col]))
    for ref_col in ("ref_poc_1", "ref_poc_2"):
        has_ref = df[ref_col] >= 0
        ref_ok = pd.Series(True, index=df.index, dtype=bool)
        ref_ok.loc[has_ref] = np.asarray([
            (seg_uid, int(ref_poc)) in segment_keys
            for seg_uid, ref_poc in zip(df.loc[has_ref, "segment_uid"], df.loc[has_ref, ref_col])
        ], dtype=bool)
        df.loc[has_ref & (~ref_ok), "valid_train"] = 0

    df["sequence"] = df[seq_col]
    df["yuv_sequence"] = df[yuv_sequence_col]
    df["poc"] = df[poc_col]
    df["qp"] = df[qp_col]
    df["bits"] = df[bits_col]
    df["psnr"] = df[psnr_col]
    df["mse"] = df[mse_col]
    df["valid_train"] = df["valid_train"].astype(int)
    df["segment_last_local"] = df["segment_last_local"].fillna(df["local_poc"]).astype(int)
    df["segment_span"] = df["segment_span"].fillna(df["segment_last_local"] + 1).astype(int)
    if intra_period_pos_col:
        if intra_period_pos_col not in df.columns:
            raise ValueError(f"CSV 缺少 intra_period_pos 列: {intra_period_pos_col!r}")
        df["intra_period_pos"] = df[intra_period_pos_col].astype(float).clip(0.0, 1.0)
    else:
        denom = np.maximum(df["segment_span"].values.astype(np.float64) - 1.0, 1.0)
        df["intra_period_pos"] = (
            df["local_poc"].values.astype(np.float64) / denom
        ).clip(0.0, 1.0).astype(np.float32)

    mse_term = str(cfg.get("loss", {}).get("mse_term", "log_mse")).lower().strip()
    if mse_term == "vmaf":
        vcol = data_cfg.get("vmaf_col")
        if not vcol:
            raise ValueError('loss.mse_term 为 "vmaf" 时必须在 data.vmaf_col 中指定 CSV 列名（如 pass2_vmaf）。')
        if vcol not in df.columns:
            raise ValueError(f"CSV 缺少 VMAF 列 {vcol!r}（loss.mse_term=vmaf 需要该列作为失真标签）。")
        df["vmaf"] = df[vcol].astype(float)

    p1cols = data_cfg.get("pass1_columns", {})
    p1_qp_col = p1cols.get("qp") or "pass1_qp"
    p1_bits_col = p1cols.get("bits") or "pass1_bits"
    p1_mse_col = p1cols.get("mse") or "pass1_mse"
    p1_psnr_col = p1cols.get("psnr") or "pass1_psnr"
    p1_vmaf_col = p1cols.get("vmaf") or "pass1_vmaf"

    if mse_term == "vmaf":
        needed = [p1_qp_col, p1_bits_col, p1_vmaf_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(
                f"当前实现默认使用 pass1 特征；loss.mse_term=vmaf 时，CSV 必须包含 pass1 列 {missing}。"
            )
    else:
        needed = [p1_qp_col, p1_bits_col, p1_mse_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"当前实现默认使用 pass1 特征；CSV 缺少 pass1 列 {missing}。")

    df["pass1_qp"] = df[p1_qp_col].astype(float)
    df["pass1_bits"] = df[p1_bits_col].astype(float)
    if mse_term == "vmaf":
        df["pass1_vmaf"] = df[p1_vmaf_col].astype(float)
        if p1_mse_col in df.columns:
            df["pass1_mse"] = df[p1_mse_col].astype(float)
        else:
            df["pass1_mse"] = 0.0
    else:
        df["pass1_mse"] = df[p1_mse_col].astype(float)
        if p1_vmaf_col in df.columns:
            df["pass1_vmaf"] = df[p1_vmaf_col].astype(float)
    if p1_psnr_col in df.columns:
        df["pass1_psnr"] = df[p1_psnr_col].astype(float)

    df["pass1_log_bits"] = np.log1p(df["pass1_bits"].values.astype(np.float64)).astype(np.float32)
    df["pass1_log_mse"] = np.log(df["pass1_mse"].values.astype(np.float64) + 1e-6).astype(np.float32)
    df["pass1_delta_qp"] = (
        (df[qp_col].values - df["pass1_qp"].values) / qp_norm_span(data_cfg)
    ).astype(np.float32)

    drop_cols = [
        "template_frame_type",
        "template_temporal_layer",
        "template_ref_local_1",
        "template_ref_local_2",
        "template_num_refs",
        "template_valid_train",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df.sort_values(group_cols + [poc_col, "row_id"]).reset_index(drop=True)
