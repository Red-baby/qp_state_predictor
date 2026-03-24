#!/usr/bin/env python3
"""
从编码器日志的 pass1 段解析每帧 RC 行（qp/bits/psnr/mse），按 CSV 中已有的 (sequence, poc) 写入新列。
不修改原始 CSV：默认读入后写出到另一路径。

依赖：与 VCA_CHANGE/scripts/build_rc_dataset.py 相同的 RC 行正则与 VFR 清洗逻辑。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

_VCA_SCRIPTS = Path(__file__).resolve().parents[2] / "VCA_CHANGE" / "scripts"
if str(_VCA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_VCA_SCRIPTS))

from build_rc_dataset import RC_RE, VFR_RUNTIME_RE  # noqa: E402

PASS1_HDR_RE = re.compile(r"Encoding at pass 1", re.IGNORECASE)
PASS2_HDR_RE = re.compile(r"Encoding at pass 2", re.IGNORECASE)


def select_pass1_segment(text: str) -> str:
    """
    确定 pass1 的 RC 文本段（再交给 RC_RE 解析）：

    1) x265/部分编码器：存在「Encoding at pass 1」→ 取最后一次该标记之后到其后第一次「Encoding at pass 2」之前（无 pass2 则到 EOF）。
    2) vtcoder 等：无上述标记 → 全文去 vfr 后，按**第一次**与**第二次**出现「RC … 0I …」之间的内容作为 pass1（第二次 0I 起为 pass2）。
       若仅有一次 0I，则从该处到 EOF（单 pass 日志）。
    """
    cleaned = VFR_RUNTIME_RE.sub("", text)

    hdr1 = list(PASS1_HDR_RE.finditer(text))
    if hdr1:
        start = hdr1[-1].start()
        tail = text[start:]
        p2 = PASS2_HDR_RE.search(tail)
        if p2:
            segment = tail[: p2.start()]
        else:
            segment = tail
        return VFR_RUNTIME_RE.sub("", segment)

    # vtcoder：两次完整编码以第二个 RC 0I 为 pass2 起点
    starts: list[int] = []
    for m in RC_RE.finditer(cleaned):
        if int(m.group(1)) != 0:
            continue
        if str(m.group(2)).upper() != "I":
            continue
        starts.append(m.start())
    if not starts:
        raise ValueError("pass1: no RC 0I line found (cannot split pass1/pass2)")
    if len(starts) >= 2:
        return cleaned[starts[0] : starts[1]]
    return cleaned[starts[0] :]


def parse_pass1_frames(log_path: Path) -> Dict[int, Dict[str, float]]:
    """返回 poc -> {pass1_qp, pass1_bits, pass1_psnr, pass1_mse}（仅非 O 帧；与同 POC 重复则报错，与 build_rc_dataset 一致）。"""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    segment = select_pass1_segment(text)

    frames = []
    seen = set()
    for match in RC_RE.finditer(segment):
        frame = {
            "timestamp": int(match.group(1)),
            "type_raw": match.group(2),
            "qp": int(match.group(3)),
            "bits": int(match.group(8)),
            "psnr": float(match.group(9)),
            "mse": float(match.group(10)),
        }
        if str(frame["type_raw"]).upper() == "O":
            continue
        key = (
            frame["timestamp"],
            frame["type_raw"],
            frame["qp"],
            frame["bits"],
            frame["psnr"],
            frame["mse"],
        )
        if key in seen:
            continue
        seen.add(key)
        frames.append(frame)

    if not frames:
        raise ValueError("no non-overlay RC frames parsed in pass1 segment")

    frames.sort(key=lambda item: (item["timestamp"], item["type_raw"]))
    by_poc: Dict[int, Dict[str, float]] = {}
    for frame in frames:
        poc = frame["timestamp"]
        if poc in by_poc:
            raise ValueError(f"duplicate non-overlay RC frame at poc {poc} in pass1")
        by_poc[poc] = {
            "pass1_qp": frame["qp"],
            "pass1_bits": frame["bits"],
            "pass1_psnr": frame["psnr"],
            "pass1_mse": frame["mse"],
        }
    return by_poc


def main() -> None:
    p = argparse.ArgumentParser(description="Enrich merged CSV with pass1 qp/bits/psnr/mse from encoder logs.")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/data/lh/qp_state_predictor/dataset/vca_rc_merged.csv"),
        help="原始合并表（只读）。",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/data/lh/qp_state_predictor/dataset/vca_rc_merged_with_pass1.csv"),
        help="写出路径（含新增列）。",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/mnt/ec-data2/lh_data/logs"),
        help="编码日志目录，文件名为 <sequence>.log。",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="可选：写出解析摘要 JSON。",
    )
    args = p.parse_args()

    if not args.input_csv.is_file():
        raise SystemExit(f"input csv not found: {args.input_csv}")

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "sequence" not in reader.fieldnames or "poc" not in reader.fieldnames:
            raise SystemExit("CSV must contain columns: sequence, poc")
        fieldnames: List[str] = list(reader.fieldnames)
        rows_in = list(reader)

    new_cols = ["pass1_qp", "pass1_bits", "pass1_psnr", "pass1_mse"]
    for c in new_cols:
        if c not in fieldnames:
            fieldnames.append(c)

    sequences: Set[str] = set()
    for row in rows_in:
        sequences.add(str(row.get("sequence", "")).strip())

    cache: Dict[str, Optional[Dict[int, Dict[str, float]]]] = {}
    errors: Dict[str, str] = {}

    for seq in sorted(sequences):
        log_path = args.log_dir / f"{seq}.log"
        if not log_path.is_file():
            errors[seq] = f"missing log: {log_path}"
            cache[seq] = None
            continue
        try:
            cache[seq] = parse_pass1_frames(log_path)
        except Exception as e:
            errors[seq] = str(e)
            cache[seq] = None

    out_rows: List[Dict[str, str]] = []
    missing_poc = 0

    for row in rows_in:
        seq = str(row.get("sequence", "")).strip()
        poc_s = str(row.get("poc", "")).strip()
        try:
            poc = int(poc_s)
        except ValueError:
            poc = -1

        m = cache.get(seq)
        if m is None or poc < 0 or poc not in m:
            if m is not None and poc >= 0:
                missing_poc += 1
            row["pass1_qp"] = ""
            row["pass1_bits"] = ""
            row["pass1_psnr"] = ""
            row["pass1_mse"] = ""
        else:
            row["pass1_qp"] = str(m[poc]["pass1_qp"])
            row["pass1_bits"] = str(m[poc]["pass1_bits"])
            row["pass1_psnr"] = str(m[poc]["pass1_psnr"])
            row["pass1_mse"] = str(m[poc]["pass1_mse"])
        out_rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "log_dir": str(args.log_dir),
        "rows": len(out_rows),
        "sequences_in_csv": len(sequences),
        "sequences_with_pass1_parse_ok": sum(1 for s in sequences if cache.get(s) is not None),
        "sequences_log_missing_or_parse_error": errors,
        "rows_with_missing_pass1_for_poc": missing_poc,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
