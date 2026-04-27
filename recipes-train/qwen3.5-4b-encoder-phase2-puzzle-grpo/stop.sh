#!/usr/bin/env bash
# Stop puzzle GRPO training (trainer + SGLang server)

TRAIN_PID_FILE="/tmp/puzzle-grpo.pid"
SGLANG_PID_FILE="/tmp/puzzle-sglang.pid"

stop_pid() {
  local pidfile="$1"
  local label="$2"
  if [[ -f "$pidfile" ]]; then
    PID=$(cat "$pidfile")
    if kill -0 "$PID" 2>/dev/null; then
      echo "Stopping $label (PID $PID)..."
      kill "$PID"
    else
      echo "$label not running (stale PID $PID)"
    fi
    rm -f "$pidfile"
  else
    echo "No PID file for $label ($pidfile)"
  fi
}

stop_pid "$TRAIN_PID_FILE" "GRPOTrainer"
stop_pid "$SGLANG_PID_FILE" "SGLang"

# Kill any remaining sglang.launch_server processes (child processes outlive parent PID)
SGLANG_PIDS=$(pgrep -f "sglang.launch_server" 2>/dev/null)
if [[ -n "$SGLANG_PIDS" ]]; then
  echo "Killing remaining SGLang processes: $SGLANG_PIDS"
  kill $SGLANG_PIDS 2>/dev/null || true
fi
