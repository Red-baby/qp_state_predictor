#!/usr/bin/env bash
# 先跑 preprocess_cache，成功后依次训练 Phase 1 → 2 → 3。
# 用法（在 qp_state_predictor 目录下）:
#   chmod +x run_preprocess_then_train.sh
#   ./run_preprocess_then_train.sh
# 在终端外层用 nohup 后台跑（勿在脚本内写 nohup）:
#   mkdir -p logs
#   nohup ./run_preprocess_then_train.sh > logs/nohup_console.txt 2>&1 &
#   echo $!
#
# 双卡训练示例（同样在外层加 nohup）:
#   nohup env TRAIN_GPUS=2 ./run_preprocess_then_train.sh > logs/nohup_preprocess_train.log 2>&1 &
#
# Phase 2/3 若启用 features.use_pair_cache，请先手动执行:
#   $PYTHON -m qp_predictor.preprocess_pair_cache --config "$CONFIG"
# （依赖已有 cache_dir/<seq>.npz，见 README Step 2b）
#
# 环境变量:
#   CONFIG          配置文件路径，默认 ./my_config.yaml
#   SKIP_PREPROCESS 设为 1 则跳过缓存预处理，直接训练（缓存已就绪时用）
#   TRAIN_GPUS      训练进程数；设为 2 时用 torch.distributed.run 双卡 DDP（默认 1 为单卡）
#   PYTHON          Python 解释器，默认 python；多卡时请与已安装 PyTorch 的版本一致（建议 ≥3.7）

set -euo pipefail
set -o pipefail

# 挂机时日志立即落盘，避免缓冲导致“卡住”假象
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 与 `python` 一致；勿用 PATH 里的 torchrun（其 shebang 常指向系统旧版 python3，会触发 3.6 不支持的语法）。
PYTHON="${PYTHON:-python}"

CONFIG="${CONFIG:-./my_config.yaml}"
LOGDIR="${LOGDIR:-$ROOT/logs}"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/preprocess_train_${STAMP}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_step() {
  log "========== START: $* =========="
  "$@" 2>&1 | tee -a "$LOG"
  log "========== DONE: $* =========="
}

log "Working directory: $ROOT"
log "Config: $CONFIG"
log "Full log file: $LOG"
echo ""

if [[ ! -f "$CONFIG" ]]; then
  log "ERROR: config not found: $CONFIG"
  exit 1
fi

if [[ "${SKIP_PREPROCESS:-0}" != "1" ]]; then
  run_step "$PYTHON" -m qp_predictor.preprocess_cache --config "$CONFIG"
else
  log "SKIP_PREPROCESS=1 — skipping preprocess_cache"
fi

TRAIN_GPUS="${TRAIN_GPUS:-1}"

run_train_phase() {
  local phase="$1"
  log "========== START: train phase ${phase} (GPUs=${TRAIN_GPUS}) =========="
  if [[ "${TRAIN_GPUS}" -gt 1 ]]; then
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${TRAIN_GPUS}" -m qp_predictor.train --config "$CONFIG" --phase "${phase}" 2>&1 | tee -a "$LOG"
  else
    "$PYTHON" -m qp_predictor.train --config "$CONFIG" --phase "${phase}" 2>&1 | tee -a "$LOG"
  fi
  log "========== DONE: train phase ${phase} =========="
}

run_train_phase 1
run_train_phase 2
run_train_phase 3

log "All steps finished successfully."
log "Checkpoints: under data.output_root, e.g. phase1/ or phase1_pass1/ when use_pass1_features=true (see train.phase_output_dirname)."
