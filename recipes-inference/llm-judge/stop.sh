#!/usr/bin/env bash
# Stop judge stack (SGLang + FastAPI judge server)

SGLANG_PID_FILE="/tmp/chess-sglang.pid"
JUDGE_PID_FILE="/tmp/chess-judge-api.pid"

stopped=0

# Stop FastAPI judge
if [[ -f "$JUDGE_PID_FILE" ]]; then
  PID=$(cat "$JUDGE_PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped chess-judge-api (PID $PID)"
    stopped=$((stopped + 1))
  else
    echo "chess-judge-api not running (stale PID $PID)"
  fi
  rm -f "$JUDGE_PID_FILE"
else
  echo "No PID file for chess-judge-api"
fi

# Stop vLLM Docker container (preferred) or the nohup process
if docker ps --format '{{.Names}}' | grep -q chess-sglang 2>/dev/null; then
  docker stop chess-sglang
  echo "Stopped chess-sglang Docker container"
  stopped=$((stopped + 1))
elif [[ -f "$SGLANG_PID_FILE" ]]; then
  PID=$(cat "$SGLANG_PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped chess-sglang (PID $PID)"
    stopped=$((stopped + 1))
  else
    echo "chess-sglang not running (stale PID $PID)"
  fi
fi
rm -f "$SGLANG_PID_FILE"

[[ $stopped -gt 0 ]] && echo "Done." || echo "Nothing was running."
