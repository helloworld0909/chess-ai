#!/usr/bin/env bash
# LLM Judge — Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 on GPU 1 (SGLang) + FastAPI judge on port 8400
#
# Two processes:
#   1. SGLang OpenAI-compatible server (port 8300, GPU 1)
#   2. FastAPI judge server (port 8400, CPU)
#
# Usage:
#   ./recipes-inference/llm-judge/start.sh
#   ./recipes-inference/llm-judge/stop.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BACKEND_LOG="/tmp/chess-sglang.log"
JUDGE_LOG="/tmp/chess-judge-api.log"
BACKEND_PID_FILE="/tmp/chess-sglang.pid"
JUDGE_PID_FILE="/tmp/chess-judge-api.pid"

MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}"
BACKEND_PORT=8300
JUDGE_PORT=8400

STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish}"
SF15_PATH="${SF15_PATH:-$HOME/.local/bin/stockfish-15}"

# ---------------------------------------------------------------------------
# Check nothing already running
# ---------------------------------------------------------------------------

for PID_FILE in "$BACKEND_PID_FILE" "$JUDGE_PID_FILE"; do
  if [[ -f "$PID_FILE" ]]; then
    EXISTING=$(cat "$PID_FILE")
    if kill -0 "$EXISTING" 2>/dev/null; then
      echo "Already running (PID $EXISTING). Run stop.sh first."
      exit 1
    else
      rm -f "$PID_FILE"
    fi
  fi
done

# Also check for lingering Docker container from previous vLLM runs
if docker ps --format '{{.Names}}' | grep -q chess-sglang 2>/dev/null; then
  echo "Stopping leftover chess-sglang Docker container..."
  docker stop chess-sglang 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 1. Start SGLang on GPU 1
# ---------------------------------------------------------------------------

echo "Starting SGLang judge model: $MODEL"
echo "  GPU      : cuda:1"
echo "  Port     : $BACKEND_PORT"
echo "  Log      : $BACKEND_LOG"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest-cu130-runtime}"

nohup docker run --rm \
  --name chess-sglang \
  --gpus '"device=1"' \
  --shm-size 16g \
  --ipc host \
  -p "$BACKEND_PORT:8300" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  "$SGLANG_IMAGE" \
    python3 -m sglang.launch_server \
      --model-path "$MODEL" \
      --host 0.0.0.0 \
      --port 8300 \
      --quantization moe_wna16 \
      --context-length 32768 \
      --mem-fraction-static 0.90 \
      --attention-backend triton \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --tp 1 \
  > "$BACKEND_LOG" 2>&1 &

BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
echo "  PID      : $BACKEND_PID"

# ---------------------------------------------------------------------------
# 2. Wait for SGLang to be ready
# ---------------------------------------------------------------------------

echo ""
echo "Waiting for SGLang to be ready (timeout 600s)..."
for i in $(seq 1 600); do
  if curl -sf "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
    echo "SGLang ready after ${i}s"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null && ! docker ps --format '{{.Names}}' | grep -q chess-sglang; then
    echo "SGLang process died. Check $BACKEND_LOG"
    exit 1
  fi
  sleep 1
done

# ---------------------------------------------------------------------------
# 3. Start FastAPI judge server
# ---------------------------------------------------------------------------

echo ""
echo "Starting FastAPI judge server"
echo "  Port     : $JUDGE_PORT"
echo "  Log      : $JUDGE_LOG"

source "$REPO_ROOT/.venv/bin/activate"

nohup env \
  JUDGE_BASE_URL="http://localhost:$BACKEND_PORT/v1" \
  JUDGE_MODEL="$MODEL" \
  STOCKFISH_PATH="$STOCKFISH_PATH" \
  SF15_PATH="$SF15_PATH" \
  PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" \
  uvicorn src.judge.server:app \
    --host 0.0.0.0 \
    --port "$JUDGE_PORT" \
    --workers 1 \
  > "$JUDGE_LOG" 2>&1 &

JUDGE_PID=$!
echo "$JUDGE_PID" > "$JUDGE_PID_FILE"
echo "  PID      : $JUDGE_PID"

# ---------------------------------------------------------------------------
# 4. Wait for judge API to be ready
# ---------------------------------------------------------------------------

echo ""
echo "Waiting for judge API to be ready (timeout 30s)..."
for i in $(seq 1 30); do
  if curl -sf "http://localhost:$JUDGE_PORT/health" > /dev/null 2>&1; then
    echo "Judge API ready after ${i}s"
    break
  fi
  if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
    echo "Judge API process died. Check $JUDGE_LOG"
    exit 1
  fi
  sleep 1
done

echo ""
echo "Judge stack running:"
echo "  SGLang (GPU 1, port $BACKEND_PORT) PID $BACKEND_PID — tail -f $BACKEND_LOG"
echo "  Judge  (CPU,   port $JUDGE_PORT)   PID $JUDGE_PID  — tail -f $JUDGE_LOG"
echo ""
echo "Stop: ./recipes-inference/llm-judge/stop.sh"
