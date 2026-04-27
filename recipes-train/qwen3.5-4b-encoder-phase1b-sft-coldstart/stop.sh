#!/usr/bin/env bash
PID_FILE="/tmp/phase1b-sft-coldstart.pid"
if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping phase1b SFT (PID $PID)..."
    kill "$PID"
  else
    echo "Not running (stale PID $PID)"
  fi
  rm -f "$PID_FILE"
else
  echo "No PID file found."
fi
