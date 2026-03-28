#!/usr/bin/env bash
# GRPO Phase 1 — Board-reading pretraining (both GPUs, no judge server needed)
#
# Usage:
#   ./recipes-train/qwen3.5-4b-grpo-phase1/start.sh
#   ./recipes-train/qwen3.5-4b-grpo-phase1/start.sh --resume
#   ./recipes-train/qwen3.5-4b-grpo-phase1/start.sh --resume checkpoints/qwen3.5-4b-grpo-phase1/checkpoint-100
#
# Logs: /tmp/chess-grpo-phase1.log
# Stop: ./recipes-train/qwen3.5-4b-grpo-phase1/stop.sh

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
cd "$REPO_ROOT"

LOG_FILE="/tmp/chess-grpo-phase1.log"
PID_FILE="/tmp/chess-grpo-phase1.pid"
CONFIG="$RECIPE_DIR/config.yaml"
EXTRA_ARGS=()

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID=$(cat "$PID_FILE")
  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "GRPO Phase-1 training already running (PID $EXISTING_PID). Run stop.sh first."
    exit 1
  else
    rm -f "$PID_FILE"
  fi
fi

source "$REPO_ROOT/.venv/bin/activate"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_VERBOSITY=warning
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export FLASHINFER_CACHE_DIR="/tmp/flashinfer-cache-zheng"
NVIDIA_BASE="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia"
NVIDIA_LIB_PATHS=$(find "$NVIDIA_BASE" -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:-}"
mkdir -p "$FLASHINFER_CACHE_DIR"

while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

echo "Config   : $CONFIG"
echo "Devices  : cuda:0,1 (both GPUs)"
echo "Log      : $LOG_FILE"
echo "Completions: completions_phase1_board_reading.jsonl"
echo ""

TRAIN_CMD="torchrun --nproc_per_node=2 $RECIPE_DIR/train.py --config $CONFIG ${EXTRA_ARGS[*]:-}"

nohup bash -c "source $REPO_ROOT/.venv/bin/activate \
  && export PYTORCH_ALLOC_CONF=expandable_segments:True \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
  && export HF_HUB_VERBOSITY=warning \
  && export CUDA_VISIBLE_DEVICES=0,1 \
  && export STOCKFISH_PATH=${STOCKFISH_PATH} \
  && export PYTHONPATH=$REPO_ROOT/src:${PYTHONPATH:-} \
  && export FLASHINFER_CACHE_DIR=/tmp/flashinfer-cache-zheng \
  && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
  && $TRAIN_CMD" \
  > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Started (PID $TRAIN_PID)"
echo "Monitor: tail -f $LOG_FILE"
echo "Stop   : $RECIPE_DIR/stop.sh"
