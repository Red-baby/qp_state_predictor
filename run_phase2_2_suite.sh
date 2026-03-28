#!/usr/bin/env bash
# 批量运行 4 组 phase2_2 实验，并为每组生成独立日志文件。
# 用法：
#   chmod +x run_phase2_2_suite.sh
#   ./run_phase2_2_suite.sh
#   TRAIN_GPUS=2 ./run_phase2_2_suite.sh
#   ONLY="phase2_2_onlybits,phase2_2_onlyvmaf" ./run_phase2_2_suite.sh

set -euo pipefail
set -o pipefail
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
LOGROOT="${LOGROOT:-$ROOT/logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="$LOGROOT/phase2_2_suite_$STAMP"
mkdir -p "$LOGDIR"

CONFIGS=(
  "phase2_2_onlybits.yaml"
  "phase2_2_onlybits_vmafaux.yaml"
  "phase2_2_onlyvmaf.yaml"
  "phase2_2_bits_vmaf.yaml"
)

if [[ -n "${ONLY:-}" ]]; then
  IFS=',' read -r -a _wanted <<< "$ONLY"
  FILTERED=()
  for cfg in "${CONFIGS[@]}"; do
    base="${cfg%.yaml}"
    for want in "${_wanted[@]}"; do
      want_trim="$(echo "$want" | xargs)"
      if [[ "$base" == "$want_trim" || "$cfg" == "$want_trim" ]]; then
        FILTERED+=("$cfg")
        break
      fi
    done
  done
  CONFIGS=("${FILTERED[@]}")
fi

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "ERROR: no configs selected"
  exit 1
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_train() {
  local config="$1"
  local idx="$2"
  local base="${config%.yaml}"
  local logfile="$LOGDIR/${idx}_${base}.log"

  if [[ ! -f "$ROOT/$config" ]]; then
    log "ERROR: config not found: $config"
    exit 1
  fi

  log "========== START: $base =========="
  log "config=$config"
  log "logfile=$logfile"

  if [[ "$TRAIN_GPUS" -gt 1 ]]; then
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TRAIN_GPUS" \
      -m qp_predictor.train --config "$config" --phase 2 2>&1 | tee "$logfile"
  else
    "$PYTHON" -m qp_predictor.train --config "$config" --phase 2 2>&1 | tee "$logfile"
  fi

  log "========== DONE: $base =========="
}

summary="$LOGDIR/00_suite_summary.txt"
{
  echo "run_phase2_2_suite"
  echo "root=$ROOT"
  echo "python=$PYTHON"
  echo "train_gpus=$TRAIN_GPUS"
  echo "logdir=$LOGDIR"
  echo "configs=${CONFIGS[*]}"
} > "$summary"

for i in "${!CONFIGS[@]}"; do
  idx="$(printf '%02d' "$((i + 1))")"
  run_train "${CONFIGS[$i]}" "$idx"
done

log "All phase2_2 jobs finished. Logs are under: $LOGDIR"
