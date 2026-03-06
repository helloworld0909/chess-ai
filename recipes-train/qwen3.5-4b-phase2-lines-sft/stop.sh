#!/usr/bin/env bash
set -euo pipefail
PID_FILE="/tmp/chess-lines-train.pid"
if [[ ! -f "$PID_FILE" ]]; then echo "No PID file ($PID_FILE). Not running?"; exit 0; fi
PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping (PID $PID)..."
  kill "$PID"
  for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -0 "$PID" 2>/dev/null && kill -9 "$PID"
  echo "Stopped."
else
  echo "Process $PID not running."
fi
rm -f "$PID_FILE"
