#!/usr/bin/env bash
# Stop phase0 alignment training started by start.sh

set -euo pipefail

PID_FILE="/tmp/phase1-alignment.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found ($PID_FILE). Is training running?"
  exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping phase0 alignment training (PID $PID)..."
  # Kill the whole process group (torchrun + all worker children)
  kill -- -"$PID" 2>/dev/null || kill "$PID"
  for i in $(seq 1 30); do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "Process still alive after 30s — sending SIGKILL"
    kill -9 -- -"$PID" 2>/dev/null || kill -9 "$PID"
  fi
  echo "Stopped."
else
  echo "Process $PID is not running."
fi

# Also kill any remaining torchrun/train.py workers
pkill -f "torchrun.*phase0" 2>/dev/null || true
pkill -f "train.py.*phase0" 2>/dev/null || true
fuser -k 29500/tcp 2>/dev/null || true

rm -f "$PID_FILE"
