#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PASS1_HDR_RE = re.compile(r"Encoding at pass\s+1", re.IGNORECASE)
PASS2_HDR_RE = re.compile(r"Encoding at pass\s+2", re.IGNORECASE)
RC_START_RE = re.compile(r"RC\s+(?P<poc>\d+)(?P<frame_type>[IOPBb])", re.IGNORECASE)
INTERLEAVED_LOG_RE = re.compile(r"\[[^\r\n]*\][^\r\n]*", re.IGNORECASE)

# Match RC records anywhere in the text. Keep optional tags and the extended
# reference/meta fields optional so the parser can handle mixed old/new logs.
RC_RE = re.compile(
    r"RC\s+"
    r"(?P<poc>\d+)"
    r"(?P<frame_type>[IOPBb])"
    r"(?:\s+\([^\r\n]*?\))?"
    r"\s+qp\s+(?P<qp>-?\d+)"
    r"\s+qpa\s+(?P<qpa>-?\d+(?:\.\d+)?)"
    r"\s+level\s+(?P<level>-?\d+)"
    r"\s+enh\s+(?P<enh>-?\d+)"
    r"\s+fg\s+(?P<fg>-?\d+)"
    r"\s+bits\s+(?P<bits>-?\d+)"
    r"\s+psnr\s+(?P<psnr>-?\d+(?:\.\d+)?)"
    r"\s+mse\s+(?P<mse>-?\d+(?:\.\d+)?)"
    r"(?:\s+vmaf\s+(?P<vmaf>-?\d+(?:\.\d+)?))?"
    r"(?:\s+refpoc0\s+(?P<refpoc0>-?\d+))?"
    r"(?:\s+refpoc1\s+(?P<refpoc1>-?\d+))?"
    r"(?:\s+refqp0\s+(?P<refqp0>-?\d+))?"
    r"(?:\s+refqp1\s+(?P<refqp1>-?\d+))?"
    r"(?:\s+intra_period_pos\s+(?P<intra_period_pos>-?\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FrameRecord:
    poc: int
    frame_type: str
    qp: int
    qpa: float
    level: int
    enh: int
    fg: int
    bits: int
    psnr: float
    mse: float
    vmaf: Optional[float]
    refpoc0: Optional[int]
    refpoc1: Optional[int]
    refqp0: Optional[int]
    refqp1: Optional[int]
    intra_period_pos: Optional[float]

    def dedup_key(self) -> tuple:
        return (
            self.poc,
            self.frame_type,
            self.qp,
            self.qpa,
            self.level,
            self.enh,
            self.fg,
            self.bits,
            self.psnr,
            self.mse,
            self.vmaf,
            self.refpoc0,
            self.refpoc1,
            self.refqp0,
            self.refqp1,
            self.intra_period_pos,
        )


def _parse_optional_int(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw == "":
        return None
    return int(raw)


def _parse_optional_float(raw: Optional[str]) -> Optional[float]:
    if raw is None or raw == "":
        return None
    return float(raw)


def _collect_zero_i_starts(text: str) -> List[int]:
    starts: List[int] = []
    for match in RC_START_RE.finditer(text):
        if int(match.group("poc")) != 0:
            continue
        if str(match.group("frame_type")).upper() != "I":
            continue
        starts.append(match.start())
    return starts


def select_pass_segment(text: str, pass_id: int) -> str:
    if pass_id not in (1, 2):
        raise ValueError(f"unsupported pass id: {pass_id}")

    hdr1 = list(PASS1_HDR_RE.finditer(text))
    hdr2 = list(PASS2_HDR_RE.finditer(text))

    if pass_id == 1 and hdr1:
        start = hdr1[-1].start()
        tail = text[start:]
        next_p2 = PASS2_HDR_RE.search(tail)
        return tail[: next_p2.start()] if next_p2 else tail

    if pass_id == 2 and hdr2:
        return text[hdr2[-1].start() :]

    starts = _collect_zero_i_starts(text)
    if len(starts) >= 2:
        if pass_id == 1:
            return text[starts[0] : starts[1]]
        return text[starts[1] :]
    if len(starts) == 1:
        return text[starts[0] :]
    return text


def _iter_rc_candidates(segment_text: str) -> Iterable[str]:
    starts = list(RC_START_RE.finditer(segment_text))
    for idx, match in enumerate(starts):
        start = match.start()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(segment_text)
        yield segment_text[start:end]


def _normalize_rc_candidate(candidate: str) -> str:
    cleaned = INTERLEAVED_LOG_RE.sub(" ", candidate)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    rc_pos = cleaned.find("RC")
    if rc_pos > 0:
        cleaned = cleaned[rc_pos:]
    return cleaned


def _parse_candidate(candidate: str) -> Optional[FrameRecord]:
    normalized = _normalize_rc_candidate(candidate)
    if not normalized:
        return None
    match = RC_RE.search(normalized)
    if match is None:
        raise ValueError(f"unparseable RC candidate: {normalized[:240]}")
    return FrameRecord(
        poc=int(match.group("poc")),
        frame_type=str(match.group("frame_type")),
        qp=int(match.group("qp")),
        qpa=float(match.group("qpa")),
        level=int(match.group("level")),
        enh=int(match.group("enh")),
        fg=int(match.group("fg")),
        bits=int(match.group("bits")),
        psnr=float(match.group("psnr")),
        mse=float(match.group("mse")),
        vmaf=_parse_optional_float(match.group("vmaf")),
        refpoc0=_parse_optional_int(match.group("refpoc0")),
        refpoc1=_parse_optional_int(match.group("refpoc1")),
        refqp0=_parse_optional_int(match.group("refqp0")),
        refqp1=_parse_optional_int(match.group("refqp1")),
        intra_period_pos=_parse_optional_float(match.group("intra_period_pos")),
    )


def parse_rc_frames(segment_text: str, *, include_overlay: bool) -> Dict[int, FrameRecord]:
    raw_records: List[FrameRecord] = []
    seen = set()
    overlays_by_poc: Dict[int, FrameRecord] = {}

    for candidate in _iter_rc_candidates(segment_text):
        record = _parse_candidate(candidate)
        if record is None:
            continue
        key = record.dedup_key()
        if key in seen:
            continue
        seen.add(key)
        raw_records.append(record)

        if record.frame_type.upper() == "O" and record.vmaf is not None:
            prev_overlay = overlays_by_poc.get(record.poc)
            if prev_overlay is not None and prev_overlay.vmaf != record.vmaf:
                raise ValueError(f"conflicting overlay VMAF at poc {record.poc}")
            overlays_by_poc[record.poc] = record

    frames: Dict[int, FrameRecord] = {}
    for record in sorted(raw_records, key=lambda item: (item.poc, item.frame_type)):
        if record.frame_type.upper() != "O" and record.vmaf is None:
            overlay = overlays_by_poc.get(record.poc)
            if overlay is not None and overlay.vmaf is not None:
                record = replace(record, vmaf=overlay.vmaf)

        if not include_overlay and record.frame_type.upper() == "O":
            continue

        prev = frames.get(record.poc)
        if prev is not None and prev.dedup_key() != record.dedup_key():
            raise ValueError(f"duplicate non-identical RC records at poc {record.poc}")
        frames[record.poc] = record

    if not frames:
        raise ValueError("no RC frames parsed from selected segment")
    return frames


def infer_intra_period_pos(poc: int, i_interval: int) -> float:
    if i_interval <= 1:
        return 0.0
    return float((poc % i_interval) / float(i_interval - 1))


def scan_log_files(root: Path, pattern: str, recursive: bool) -> List[Path]:
    files = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(p for p in files if p.is_file())


def build_row(
    sequence: str,
    poc: int,
    pass2_record: FrameRecord,
    pass1_record: Optional[FrameRecord],
    *,
    i_interval: int,
) -> Dict[str, object]:
    intra_period_pos = (
        pass2_record.intra_period_pos
        if pass2_record.intra_period_pos is not None
        else infer_intra_period_pos(poc, i_interval)
    )
    return {
        "sequence": sequence,
        "poc": poc,
        "frame_type": pass2_record.frame_type,
        "level": pass2_record.level,
        "enc_qp": pass2_record.qp,
        "enc_bits": pass2_record.bits,
        "enc_psnr": pass2_record.psnr,
        "enc_mse": pass2_record.mse,
        "pass2_vmaf": pass2_record.vmaf if pass2_record.vmaf is not None else "",
        "refpoc0": pass2_record.refpoc0 if pass2_record.refpoc0 is not None else -1,
        "refpoc1": pass2_record.refpoc1 if pass2_record.refpoc1 is not None else -1,
        "refqp0": pass2_record.refqp0 if pass2_record.refqp0 is not None else -1,
        "refqp1": pass2_record.refqp1 if pass2_record.refqp1 is not None else -1,
        "intra_period_pos": intra_period_pos,
        "pass1_qp": pass1_record.qp if pass1_record is not None else "",
        "pass1_bits": pass1_record.bits if pass1_record is not None else "",
        "pass1_psnr": pass1_record.psnr if pass1_record is not None else "",
        "pass1_mse": pass1_record.mse if pass1_record is not None else "",
        "pass1_vmaf": pass1_record.vmaf if pass1_record is not None and pass1_record.vmaf is not None else "",
    }


def write_csv(rows: Iterable[Dict[str, object]], output_csv: Path) -> None:
    fieldnames = [
        "sequence",
        "poc",
        "frame_type",
        "level",
        "enc_qp",
        "enc_bits",
        "enc_psnr",
        "enc_mse",
        "pass2_vmaf",
        "refpoc0",
        "refpoc1",
        "refqp0",
        "refqp1",
        "intra_period_pos",
        "pass1_qp",
        "pass1_bits",
        "pass1_psnr",
        "pass1_mse",
        "pass1_vmaf",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse encoder logs under a directory and build a training CSV for qp_state_predictor."
    )
    parser.add_argument("--log-dir", type=Path, required=True, help="Root directory containing encoder logs.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--pattern",
        default="*.log",
        help="Log filename glob pattern. Default: *.log",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan log-dir with rglob(pattern).",
    )
    parser.add_argument(
        "--include-overlay",
        action="store_true",
        help="Include O frames. Default is false to match analysis/training defaults.",
    )
    parser.add_argument(
        "--allow-missing-pass1",
        action="store_true",
        help="Do not fail when a pass2 row cannot find the same poc in pass1.",
    )
    parser.add_argument(
        "--allow-missing-vmaf",
        action="store_true",
        help="Do not fail when pass1/pass2 VMAF is missing.",
    )
    parser.add_argument(
        "--i-interval",
        type=int,
        default=125,
        help="Used to infer intra_period_pos when the log record does not contain it. Default: 125",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON path.",
    )
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        raise SystemExit(f"log directory not found: {args.log_dir}")

    log_files = scan_log_files(args.log_dir, args.pattern, args.recursive)
    if not log_files:
        raise SystemExit(f"no log files matched under {args.log_dir} with pattern {args.pattern!r}")

    rows: List[Dict[str, object]] = []
    errors: Dict[str, str] = {}
    missing_pass1_rows = 0
    missing_vmaf_rows = 0
    sequence_seen: Dict[str, Path] = {}

    for log_path in log_files:
        sequence = log_path.stem
        if sequence in sequence_seen:
            prev = sequence_seen[sequence]
            raise SystemExit(
                f"duplicate sequence name derived from filename stem {sequence!r}: {prev} and {log_path}. "
                "Rename files or separate them before building a single CSV."
            )
        sequence_seen[sequence] = log_path

        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            pass1_frames = parse_rc_frames(
                select_pass_segment(text, 1),
                include_overlay=args.include_overlay,
            )
            pass2_frames = parse_rc_frames(
                select_pass_segment(text, 2),
                include_overlay=args.include_overlay,
            )
        except Exception as exc:
            errors[str(log_path)] = str(exc)
            continue

        for poc in sorted(pass2_frames):
            pass2_record = pass2_frames[poc]
            pass1_record = pass1_frames.get(poc)

            if pass1_record is None:
                missing_pass1_rows += 1
                if not args.allow_missing_pass1:
                    raise SystemExit(
                        f"missing same-poc pass1 record for sequence={sequence!r}, poc={poc}. "
                        "Use --allow-missing-pass1 only if you do not need pass1 features."
                    )

            pass2_missing_vmaf = pass2_record.vmaf is None
            pass1_missing_vmaf = pass1_record is None or pass1_record.vmaf is None
            if pass2_missing_vmaf or pass1_missing_vmaf:
                missing_vmaf_rows += 1
                if not args.allow_missing_vmaf:
                    raise SystemExit(
                        f"missing VMAF for sequence={sequence!r}, poc={poc}. "
                        "Use --allow-missing-vmaf only if the target run does not need VMAF."
                    )

            rows.append(
                build_row(
                    sequence=sequence,
                    poc=poc,
                    pass2_record=pass2_record,
                    pass1_record=pass1_record,
                    i_interval=args.i_interval,
                )
            )

    if errors:
        raise SystemExit(
            "failed to parse some logs:\n"
            + "\n".join(f"- {path}: {msg}" for path, msg in sorted(errors.items()))
        )

    rows.sort(key=lambda item: (str(item["sequence"]), int(item["poc"])))
    write_csv(rows, args.output_csv)

    summary = {
        "log_dir": str(args.log_dir),
        "pattern": args.pattern,
        "recursive": bool(args.recursive),
        "files_parsed": len(log_files),
        "rows_written": len(rows),
        "missing_pass1_rows": missing_pass1_rows,
        "missing_vmaf_rows": missing_vmaf_rows,
        "output_csv": str(args.output_csv),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
