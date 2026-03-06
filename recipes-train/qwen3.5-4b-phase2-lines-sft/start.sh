#!/usr/bin/env bash
# Line-generator SFT — Qwen3.5-4B-Thinking-2507 + QLoRA (8-bit, r=64)
# DDP: 2× RTX 5090  |  10k train / 500 eval
# Output: checkpoints/chess-lines-4b-sft/
#
# Usage:
#   ./recipes-train/qwen3.5-4b-phase2-lines-sft/start.sh
#   ./recipes-train/qwen3.5-4b-phase2-lines-sft/start.sh --resume
#
# Logs: /tmp/chess-lines-train.log
# Stop: ./recipes-train/qwen3.5-4b-phase2-lines-sft/stop.sh

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
cd "$REPO_ROOT"

LOG_FILE="/tmp/chess-lines-train.log"
PID_FILE="/tmp/chess-lines-train.pid"
CONFIG="$RECIPE_DIR/config.yaml"
NPROC=2
EXTRA_ARGS=()

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID=$(cat "$PID_FILE")
  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Training already running (PID $EXISTING_PID). Run stop.sh first."
    exit 1
  else
    rm -f "$PID_FILE"
  fi
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1"); shift
done

echo "Config : $CONFIG"
echo "Devices: $NPROC GPUs (DDP)"
echo "Log    : $LOG_FILE"
echo ""

TRAIN_CMD="torchrun --nproc_per_node=$NPROC $RECIPE_DIR/train.py --config $CONFIG ${EXTRA_ARGS[*]:-}"

# shellcheck disable=SC2086
nohup bash -c "source $REPO_ROOT/.venv/bin/activate \
  && export PYTORCH_ALLOC_CONF=expandable_segments:True \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
  && export NCCL_TIMEOUT=3600 \
  && export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
  && $TRAIN_CMD" \
  > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "Started (PID $TRAIN_PID)"
echo "Monitor: tail -f $LOG_FILE"
echo "Stop   : $RECIPE_DIR/stop.sh"
