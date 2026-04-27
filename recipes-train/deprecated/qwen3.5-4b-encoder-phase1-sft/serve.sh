#!/usr/bin/env bash
# Launch the encoder inference server (ChessLMWithEncoder on GPU 0).
#
# Usage:
#   ./recipes-train/qwen3.5-4b-encoder-phase1-sft/serve.sh
#
# Env overrides:
#   ENCODER_CHECKPOINT  — path to checkpoint dir (default: checkpoints/qwen3.5-4b-encoder-phase1-sft)
#   ENCODER_PORT        — port to listen on (default: 8200)
#   STOCKFISH_DEPTH     — Stockfish analysis depth (default: 18)
#
# Logs: /tmp/encoder-server.log

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
cd "$REPO_ROOT"

LOG_FILE="/tmp/encoder-server.log"
PID_FILE="/tmp/encoder-server.pid"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID=$(cat "$PID_FILE")
  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Encoder server already running (PID $EXISTING_PID)."
    echo "Stop it first: kill $EXISTING_PID"
    exit 1
  else
    rm -f "$PID_FILE"
  fi
fi

source "$REPO_ROOT/.venv/bin/activate"

export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish}"
export SF15_PATH="${SF15_PATH:-$HOME/.local/bin/stockfish-15}"
# checkpoint-840 has best eval_loss (0.3412); final checkpoint is checkpoint-890
export ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-checkpoints/qwen3.5-4b-encoder-phase1-sft/checkpoint-840}"
export ENCODER_CONFIG="${ENCODER_CONFIG:-recipes-train/qwen3.5-4b-encoder-phase1-sft/config.yaml}"
export ENCODER_PORT="${ENCODER_PORT:-8200}"
export STOCKFISH_DEPTH="${STOCKFISH_DEPTH:-18}"

echo "Checkpoint : $ENCODER_CHECKPOINT"
echo "Port       : $ENCODER_PORT"
echo "Stockfish  : $STOCKFISH_PATH (depth $STOCKFISH_DEPTH)"
echo "SF15       : $SF15_PATH"
echo "Log        : $LOG_FILE"
echo ""

nohup bash -c "
  source '$REPO_ROOT/.venv/bin/activate'
  export PYTHONPATH='$REPO_ROOT'
  export STOCKFISH_PATH='$STOCKFISH_PATH'
  export SF15_PATH='$SF15_PATH'
  export ENCODER_CHECKPOINT='$ENCODER_CHECKPOINT'
  export ENCODER_CONFIG='$ENCODER_CONFIG'
  export ENCODER_PORT='$ENCODER_PORT'
  export STOCKFISH_DEPTH='$STOCKFISH_DEPTH'
  python src/tutor/encoder_server.py
" > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo "Started encoder server (PID $SERVER_PID)"
echo "Monitor: tail -f $LOG_FILE"
echo "Health:  curl http://localhost:$ENCODER_PORT/health"
echo "Stop:    kill \$(cat $PID_FILE)"
