#!/usr/bin/env bash
# Stop Qwen3.5-4B Encoder Phase 1 SFT training

set -euo pipefail

PID_FILE="/tmp/encoder-phase1-sft.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE. Is training running?"
  exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping training (PID $PID)..."
  kill "$PID"
  rm -f "$PID_FILE"
  echo "Stopped."
else
  echo "Process $PID is not running."
  rm -f "$PID_FILE"
fi
