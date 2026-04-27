#!/usr/bin/env bash
# Stop GRPO training started by start.sh

set -euo pipefail

PID_FILE="/tmp/chess-grpo.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found ($PID_FILE). Is training running?"
  exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping GRPO training (PID $PID)..."
  kill "$PID"
  # Wait up to 30s for clean shutdown
  for i in $(seq 1 30); do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "Process still alive after 30s — sending SIGKILL"
    kill -9 "$PID"
  fi
  echo "Stopped."
else
  echo "Process $PID is not running."
fi

rm -f "$PID_FILE"
