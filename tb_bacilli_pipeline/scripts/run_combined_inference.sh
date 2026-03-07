#!/usr/bin/env bash
# =============================================================================
# Combined SR → TB Inference Pipeline
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

export DATA_ROOT="${DATA_ROOT:-./DDS3}"
export CKPT_DIR="${CKPT_DIR:-./checkpoints}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

echo "=== Combined SR → TB Pipeline ==="
echo "DATA_ROOT: $DATA_ROOT"
echo "Starting at: $(date)"

python -m pipelines.combined_inference --config configs/pipeline_config.yaml "$@"

echo "Completed at: $(date)"
