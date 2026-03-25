from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .graph import build_default_local_template
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
    template = build_default_local_template(
        i_interval=int(data_cfg["i_interval"]),
        gop_size=int(data_cfg["gop_size"]),
    )
    template_valid = template[["local_poc", "valid_train"]].rename(columns={"valid_train": "template_valid_train"})
    df = df.merge(template_valid, on="local_poc", how="left")

    explicit = data_cfg["explicit_ref_columns"]
    explicit_frame_type = explicit.get("frame_type")
    explicit_layer = explicit.get("temporal_layer")
    explicit_ref1 = explicit.get("ref_poc_1")
    explicit_ref2 = explicit.get("ref_poc_2")

    if explicit_frame_type and explicit_frame_type in df.columns:
        df["frame_type"] = df[explicit_frame_type].astype(str)
    if explicit_layer and explicit_layer in df.columns:
        df["temporal_layer"] = df[explicit_layer].astype(int)
    if explicit_ref1 and explicit_ref1 in df.columns:
        df["ref_poc_1"] = df[explicit_ref1].fillna(-1).astype(int)
    if explicit_ref2 and explicit_ref2 in df.columns:
        df["ref_poc_2"] = df[explicit_ref2].fillna(-1).astype(int)

    if not {"frame_type", "temporal_layer", "ref_poc_1", "ref_poc_2"}.issubset(set(df.columns)):
        if not data_cfg["infer_refs_if_missing"]:
            raise ValueError("Reference columns missing and infer_refs_if_missing=False")
        structural_cols = ["local_poc", "frame_type", "temporal_layer", "ref_local_1", "ref_local_2", "num_refs"]
        df = df.merge(template[structural_cols], on="local_poc", how="left")
        df["ref_poc_1"] = df.apply(
            lambda x: -1 if int(x["ref_local_1"]) < 0 else int(x[poc_col] - x["local_poc"] + x["ref_local_1"]),
            axis=1,
        )
        df["ref_poc_2"] = df.apply(
            lambda x: -1 if int(x["ref_local_2"]) < 0 else int(x[poc_col] - x["local_poc"] + x["ref_local_2"]),
            axis=1,
        )
        df.drop(columns=["ref_local_1", "ref_local_2"], inplace=True)
    else:
        df["num_refs"] = ((df["ref_poc_1"] >= 0).astype(int) + (df["ref_poc_2"] >= 0).astype(int))
    df["valid_train"] = df["template_valid_train"].fillna(0).astype(int)

    df["distance_to_prev_I"] = df["local_poc"]
    df["distance_to_next_I"] = int(data_cfg["i_interval"]) - df["local_poc"]
    df["is_first_after_I"] = ((df["local_poc"] > 0) & (df["local_poc"] <= int(data_cfg["gop_size"]))).astype(int)
    df["ref_distance_1"] = (df[poc_col] - df["ref_poc_1"]).where(df["ref_poc_1"] >= 0, -1)
    df["ref_distance_2"] = (df["ref_poc_2"] - df[poc_col]).where(df["ref_poc_2"] >= 0, -1)

    base_cols = [c for c in group_cols] + [poc_col]
    df["base_uid"] = df[base_cols].astype(str).agg("|".join, axis=1)

    seg_cols = [c for c in group_cols] + ["segment_id"]
    df["segment_uid"] = df[seg_cols].astype(str).agg("|".join, axis=1)
    segment_keys = set(zip(df["segment_uid"], df[poc_col]))
    for ref_col in ["ref_poc_1", "ref_poc_2"]:
        has_ref = df[ref_col] >= 0
        ref_ok = pd.Series(True, index=df.index)
        ref_ok.loc[has_ref] = [
            (seg_uid, int(ref_poc)) in segment_keys
            for seg_uid, ref_poc in zip(df.loc[has_ref, "segment_uid"], df.loc[has_ref, ref_col])
        ]
        df.loc[has_ref & (~ref_ok), "valid_train"] = 0
    df.drop(columns=["template_valid_train"], inplace=True)

    df["sequence"] = df[seq_col]
    df["yuv_sequence"] = df[yuv_sequence_col]
    df["poc"] = df[poc_col]
    df["qp"] = df[qp_col]
    df["bits"] = df[bits_col]
    df["psnr"] = df[psnr_col]
    df["mse"] = df[mse_col]
    df["valid_train"] = df["valid_train"].astype(int)

    mse_term = str(cfg.get("loss", {}).get("mse_term", "log_mse")).lower().strip()
    if mse_term == "vmaf":
        vcol = data_cfg.get("vmaf_col")
        if not vcol:
            raise ValueError(
                'loss.mse_term 为 "vmaf" 时必须在 data.vmaf_col 中指定 CSV 列名（如 pass2_vmaf）。'
            )
        if vcol not in df.columns:
            raise ValueError(f"CSV 缺少 VMAF 列 {vcol!r}（loss.mse_term=vmaf 需要该列作为失真标签）。")
        df["vmaf"] = df[vcol].astype(float)

    if data_cfg.get("use_pass1_features", False):
        p1cols = data_cfg.get("pass1_columns", {})
        p1_qp_col = p1cols.get("qp")
        p1_bits_col = p1cols.get("bits")
        p1_mse_col = p1cols.get("mse")
        p1_psnr_col = p1cols.get("psnr")
        p1_vmaf_col = p1cols.get("vmaf")

        if mse_term == "vmaf":
            vmaf_pass1_col = p1_vmaf_col or "pass1_vmaf"
            needed = [c for c in (p1_qp_col, p1_bits_col, vmaf_pass1_col) if c]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                raise ValueError(
                    f"use_pass1_features=true 且 loss.mse_term=vmaf 时，CSV 须含 pass1 列: {missing}。"
                    f" 第 3 维先验为 {vmaf_pass1_col!r}（可在 data.pass1_columns.vmaf 覆盖列名）。"
                )
        else:
            needed = [c for c in (p1_qp_col, p1_bits_col, p1_mse_col) if c]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                raise ValueError(
                    f"use_pass1_features=true 但 CSV 中缺少 pass1 列: {missing}。"
                    " 请检查 data.pass1_columns 配置与 labels_csv 是否一致。"
                )

        if p1_qp_col and p1_qp_col in df.columns:
            df["pass1_qp"] = df[p1_qp_col].astype(float)
        if p1_bits_col and p1_bits_col in df.columns:
            df["pass1_bits"] = df[p1_bits_col].astype(float)
        if mse_term == "vmaf":
            df["pass1_vmaf"] = df[vmaf_pass1_col].astype(float)
            if p1_mse_col and p1_mse_col in df.columns:
                df["pass1_mse"] = df[p1_mse_col].astype(float)
            else:
                df["pass1_mse"] = 0.0
        elif p1_mse_col and p1_mse_col in df.columns:
            df["pass1_mse"] = df[p1_mse_col].astype(float)
        if p1_psnr_col and p1_psnr_col in df.columns:
            df["pass1_psnr"] = df[p1_psnr_col].astype(float)

        df["pass1_log_bits"] = np.log1p(df["pass1_bits"].values.astype(np.float64)).astype(np.float32)
        df["pass1_log_mse"] = np.log(df["pass1_mse"].values.astype(np.float64) + 1e-6).astype(np.float32)
        df["pass1_delta_qp"] = (
            (df[qp_col].values - df["pass1_qp"].values) / qp_norm_span(data_cfg)
        ).astype(np.float32)

    return df.sort_values(group_cols + [poc_col, "row_id"]).reset_index(drop=True)
