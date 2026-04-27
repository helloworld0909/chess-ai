#!/usr/bin/env bash
# Stop GRPO Phase 1 training

set -euo pipefail

PID_FILE="/tmp/chess-grpo-phase1.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found ($PID_FILE). Nothing to stop."
  exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping GRPO Phase-1 training (PID $PID)..."
  kill "$PID"
  sleep 2
  if kill -0 "$PID" 2>/dev/null; then
    kill -9 "$PID"
  fi
  echo "Stopped."
else
  echo "Process $PID not running."
fi

rm -f "$PID_FILE"
