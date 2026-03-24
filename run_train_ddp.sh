#!/usr/bin/env bash
# 双卡（或多卡）DDP 训练示例。batch_size_phase* 为每卡 batch，全局有效 batch ≈ 每卡 × 卡数。
#
# 用法:
#   ./run_train_ddp.sh 1          # Phase 1，默认 2 卡
#   NGPUS=4 ./run_train_ddp.sh 3  # Phase 3，4 卡
#   CONFIG=./my_config.yaml ./run_train_ddp.sh 2
#
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

CONFIG="${CONFIG:-./my_config.yaml}"
NGPUS="${NGPUS:-2}"
PHASE="${1:?用法: $0 <phase 1|2|3>}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi

exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NGPUS}" -m qp_predictor.train --config "$CONFIG" --phase "$PHASE"
