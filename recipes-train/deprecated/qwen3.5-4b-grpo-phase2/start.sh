#!/usr/bin/env bash
# GRPO Phase 1 — Qwen3.5-4B + LoRA + LLM Judge (no encoder)
# Single GPU (GPU 0) — GPU 1 reserved for inference (judge model).
#
# Prerequisites:
#   ./recipes-inference/llm-judge/start.sh  (must be running before starting training)
#
# Usage:
#   ./recipes-train/qwen3.5-4b-grpo-phase2/start.sh
#   ./recipes-train/qwen3.5-4b-grpo-phase2/start.sh --resume
#   ./recipes-train/qwen3.5-4b-grpo-phase2/start.sh --resume checkpoints/qwen3.5-4b-grpo-phase2/checkpoint-50
#
# Logs: /tmp/chess-grpo-phase2.log
# Stop: ./recipes-train/qwen3.5-4b-grpo-phase2/stop.sh

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
cd "$REPO_ROOT"

LOG_FILE="/tmp/chess-grpo-phase2.log"
PID_FILE="/tmp/chess-grpo-phase2.pid"
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

JUDGE_PORT=8400
if ! curl -sf "http://localhost:$JUDGE_PORT/health" > /dev/null 2>&1; then
  echo "ERROR: Inference server not running on port $JUDGE_PORT."
  echo "Start it first with: ./recipes-inference/llm-judge/start.sh"
  exit 1
fi
echo "Inference server: running (port $JUDGE_PORT)"

source "$REPO_ROOT/.venv/bin/activate"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export CUDA_VISIBLE_DEVICES=0
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish}"
export SF15_PATH="${SF15_PATH:-$HOME/.local/bin/stockfish-15}"
export JUDGE_SERVER_URL="http://localhost:$JUDGE_PORT"
export JUDGE_TIMEOUT=300.0
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export FLASHINFER_CACHE_DIR="/tmp/flashinfer-cache-zheng"
mkdir -p "$FLASHINFER_CACHE_DIR"

while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

echo "Config    : $CONFIG"
echo "Device    : cuda:0 (single GPU)"
echo "Stockfish : $STOCKFISH_PATH"
echo "SF15      : $SF15_PATH"
echo "Inference : $JUDGE_SERVER_URL"
echo "Log       : $LOG_FILE"
echo "Completions: completions_phase2_grpo.jsonl"
echo ""

TRAIN_CMD="python $RECIPE_DIR/train.py --config $CONFIG ${EXTRA_ARGS[*]:-}"

nohup bash -c "source $REPO_ROOT/.venv/bin/activate \
  && export PYTORCH_ALLOC_CONF=expandable_segments:True \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
  && export CUDA_VISIBLE_DEVICES=0 \
  && export STOCKFISH_PATH=${STOCKFISH_PATH} \
  && export SF15_PATH=${SF15_PATH} \
  && export JUDGE_SERVER_URL=http://localhost:${JUDGE_PORT} \
  && export PYTHONPATH=$REPO_ROOT/src:${PYTHONPATH:-} \
  && export FLASHINFER_CACHE_DIR=/tmp/flashinfer-cache-zheng \
  && $TRAIN_CMD" \
  > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Started (PID $TRAIN_PID)"
echo "Monitor: tail -f $LOG_FILE"
echo "Stop   : $RECIPE_DIR/stop.sh"
