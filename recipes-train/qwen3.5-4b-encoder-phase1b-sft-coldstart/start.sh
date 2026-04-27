#!/usr/bin/env bash
# Phase 1b: SFT Cold-start on puzzle traces
# 2× GPU DDP via torchrun. SGLang not needed.
#
# Usage:
#   ./recipes-train/qwen3.5-4b-encoder-phase1b-sft-coldstart/start.sh
#   ./recipes-train/qwen3.5-4b-encoder-phase1b-sft-coldstart/start.sh --resume

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
cd "$REPO_ROOT"

LOG="/tmp/phase1b-sft-coldstart.log"
PID_FILE="/tmp/phase1b-sft-coldstart.pid"
CONFIG="$RECIPE_DIR/config.yaml"
EXTRA_ARGS=()

if [[ -f "$PID_FILE" ]]; then
  EXISTING=$(cat "$PID_FILE")
  if kill -0 "$EXISTING" 2>/dev/null; then
    echo "Already running (PID $EXISTING). Stop it first."
    exit 1
  fi
  rm -f "$PID_FILE"
fi

source "$REPO_ROOT/.venv/bin/activate"

while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

echo "Starting Phase 1b SFT cold-start → $LOG"

nohup bash -c "
  source $REPO_ROOT/.venv/bin/activate
  export CUDA_VISIBLE_DEVICES=0,1
  export PYTHONPATH=$REPO_ROOT/src:\${PYTHONPATH:-}
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  torchrun --nproc_per_node=2 --master_port=29501 \
    $RECIPE_DIR/train.py --config $CONFIG ${EXTRA_ARGS[*]:-}
" > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "PID $PID — tail -f $LOG"
