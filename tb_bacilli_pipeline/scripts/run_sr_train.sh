#!/usr/bin/env bash
# =============================================================================
# SR Training Script
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

export DATA_ROOT="${DATA_ROOT:-./DDS3}"
export CKPT_DIR="${CKPT_DIR:-./checkpoints}"
export LOG_DIR="${LOG_DIR:-./logs}"

echo "=== SR Training ==="
echo "DATA_ROOT: $DATA_ROOT"
echo "CKPT_DIR:  $CKPT_DIR"
echo "Starting at: $(date)"

# Run training
python -m sr_project.train --config configs/sr_config.yaml "$@"

echo "Completed at: $(date)"

# ---- Background execution examples ----
# nohup bash scripts/run_sr_train.sh > logs/sr_train.log 2>&1 &
# tmux new-session -d -s sr_train 'bash scripts/run_sr_train.sh'
