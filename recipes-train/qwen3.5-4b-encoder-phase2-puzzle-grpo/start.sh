#!/usr/bin/env bash
# Puzzle GRPO training — Lichess tactical puzzles + SGLang rollouts
# GPU 0: TRL GRPOTrainer (training actor)
# GPU 1: SGLang server (rollout engine)
#
# Usage:
#   ./recipes-train/qwen3.5-4b-encoder-phase2-puzzle-grpo/start.sh
#   ./recipes-train/qwen3.5-4b-encoder-phase2-puzzle-grpo/start.sh --resume           # resume last
#   ./recipes-train/qwen3.5-4b-encoder-phase2-puzzle-grpo/start.sh --max-steps 2      # smoke test
#
# Logs: /tmp/puzzle-grpo.log  /tmp/puzzle-sglang.log
# Stop: kill $(cat /tmp/puzzle-grpo.pid /tmp/puzzle-sglang.pid 2>/dev/null)

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
cd "$REPO_ROOT"

TRAIN_LOG="/tmp/puzzle-grpo.log"
SGLANG_LOG="/tmp/puzzle-sglang.log"
TRAIN_PID_FILE="/tmp/puzzle-grpo.pid"
SGLANG_PID_FILE="/tmp/puzzle-sglang.pid"
CONFIG="$RECIPE_DIR/config.yaml"
MODEL_PATH="${CHESS_MODEL_PATH:-/tmp/chess-merged-board-reading}"
EXTRA_ARGS=()

# Guard against double-start
if [[ -f "$TRAIN_PID_FILE" ]]; then
  EXISTING=$(cat "$TRAIN_PID_FILE")
  if kill -0 "$EXISTING" 2>/dev/null; then
    echo "Puzzle GRPO trainer already running (PID $EXISTING)."
    exit 1
  else
    rm -f "$TRAIN_PID_FILE"
  fi
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

# Forward remaining args to train.py
while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 1. Start SGLang on GPU 1
# ---------------------------------------------------------------------------
echo "Starting SGLang server on GPU 1 → $SGLANG_LOG"
nohup bash -c "
  source $REPO_ROOT/.venv/bin/activate
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTHONPATH=$REPO_ROOT/src:\${PYTHONPATH:-}
  export CUDA_VISIBLE_DEVICES=1
  export CHESS_DP=1
  export CHESS_DEBUG_MM=1
  bash $REPO_ROOT/sglang/serve.sh $MODEL_PATH
" > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
echo "$SGLANG_PID" > "$SGLANG_PID_FILE"
echo "SGLang PID $SGLANG_PID"

# ---------------------------------------------------------------------------
# 2. Wait for SGLang health
# ---------------------------------------------------------------------------
SGLANG_URL="http://localhost:8300"
echo -n "Waiting for SGLang..."
MAX_WAIT=120
WAITED=0
until curl -sf "$SGLANG_URL/health" > /dev/null 2>&1; do
  sleep 2
  WAITED=$((WAITED + 2))
  if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo ""
    echo "ERROR: SGLang did not start within ${MAX_WAIT}s. Check $SGLANG_LOG"
    kill "$SGLANG_PID" 2>/dev/null || true
    exit 1
  fi
  echo -n "."
done
echo " ready (${WAITED}s)"

# ---------------------------------------------------------------------------
# 3. Start GRPOTrainer on GPU 0
# ---------------------------------------------------------------------------
echo "Starting GRPOTrainer on GPU 0 → $TRAIN_LOG"
TRAIN_CMD="CUDA_VISIBLE_DEVICES=0 python $RECIPE_DIR/train.py --config $CONFIG ${EXTRA_ARGS[*]:-}"

nohup bash -c "
  source $REPO_ROOT/.venv/bin/activate
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH=$REPO_ROOT/src:\${PYTHONPATH:-}
  $TRAIN_CMD
" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$TRAIN_PID_FILE"

echo ""
echo "Puzzle GRPO training started:"
echo "  SGLang  PID $SGLANG_PID  → tail -f $SGLANG_LOG"
echo "  Trainer PID $TRAIN_PID   → tail -f $TRAIN_LOG"
echo ""
echo "Monitor R_correct: grep 'rewards/mean' $TRAIN_LOG"
echo "Stop: kill \$(cat $TRAIN_PID_FILE $SGLANG_PID_FILE)"
