#!/usr/bin/env bash
# Stop SFT training — Qwen3.5-4B-Thinking-2507

PID_FILE="/tmp/chess-train.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found ($PID_FILE). Is training running?"
  STRAY=$(pgrep -f "train\.py|torchrun" 2>/dev/null || true)
  if [[ -n "$STRAY" ]]; then
    echo "Found stray processes: $STRAY — killing them."
    pkill -SIGTERM -f "train\.py|torchrun" 2>/dev/null || true
  fi
  exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID is not running (already finished or crashed)."
  rm -f "$PID_FILE"
  exit 0
fi

echo "Sending SIGTERM to process group of PID $PID..."
kill -SIGTERM -- "-$PID" 2>/dev/null || kill -SIGTERM "$PID" 2>/dev/null

for i in $(seq 1 30); do
  if ! kill -0 "$PID" 2>/dev/null; then break; fi
  sleep 1
done

if kill -0 "$PID" 2>/dev/null; then
  echo "Process still alive after 30s — sending SIGKILL."
  kill -SIGKILL -- "-$PID" 2>/dev/null || kill -SIGKILL "$PID" 2>/dev/null
fi

rm -f "$PID_FILE"
echo "Done."
