#!/usr/bin/env bash
# Stop GRPO Phase-1 training

PID_FILE="/tmp/chess-grpo-phase1.pid"

if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped GRPO Phase-1 training (PID $PID)"
  else
    echo "Not running (stale PID $PID)"
  fi
  rm -f "$PID_FILE"
else
  echo "No PID file found — nothing to stop"
fi
