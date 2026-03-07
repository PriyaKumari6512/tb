#!/usr/bin/env bash
# =============================================================================
# TB Segmentation Training Script
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

export DATA_ROOT="${DATA_ROOT:-./DDS3}"
export CKPT_DIR="${CKPT_DIR:-./checkpoints}"
export LOG_DIR="${LOG_DIR:-./logs}"

echo "=== TB Segmentation Training ==="
echo "DATA_ROOT: $DATA_ROOT"
echo "CKPT_DIR:  $CKPT_DIR"
echo "Starting at: $(date)"

python -m tb_project.train --config configs/tb_config.yaml "$@"

echo "Completed at: $(date)"

# ---- Background execution ----
# nohup bash scripts/run_tb_train.sh > logs/tb_train.log 2>&1 &
