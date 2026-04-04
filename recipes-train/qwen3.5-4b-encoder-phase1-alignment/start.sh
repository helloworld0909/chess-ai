#!/usr/bin/env bash
# Phase 0: MLP Projector Alignment
# Trains only cnn.proj (2-layer MLP) — CNN trunk and LLM fully frozen.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish}"
export SF15_PATH="${SF15_PATH:-$HOME/.local/bin/stockfish-15}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# cuDNN from venv
VENV_CUDNN="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
if [ -d "$VENV_CUDNN" ]; then
    export LD_LIBRARY_PATH="$VENV_CUDNN:${LD_LIBRARY_PATH:-}"
fi

LOG_FILE="/tmp/phase1-alignment.log"
PID_FILE="/tmp/phase1-alignment.pid"
CONFIG="recipes-train/qwen3.5-4b-encoder-phase1-alignment/config.yaml"

RESUME_ARG=""
if [ "${1:-}" = "--resume" ]; then
    # Pass the checkpoint path if provided, else let Trainer auto-find last checkpoint.
    if [ -n "${2:-}" ]; then
        RESUME_ARG="--resume ${2}"
    else
        RESUME_ARG="--resume"
    fi
fi

TRAIN_CMD="torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    recipes-train/qwen3.5-4b-encoder-phase1-alignment/train.py \
    --config $CONFIG \
    $RESUME_ARG"

echo "Starting phase0 alignment training (log: $LOG_FILE)"

nohup bash -c "source $REPO_ROOT/.venv/bin/activate \
  && export PYTHONPATH=$REPO_ROOT/src:$REPO_ROOT \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
  && export HF_HUB_OFFLINE=1 \
  && export NCCL_TIMEOUT=3600 \
  && export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
  && $TRAIN_CMD" \
  > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Training PID: $TRAIN_PID (saved to $PID_FILE)"
echo "Monitor: tail -f $LOG_FILE"
echo "Stop   : recipes-train/qwen3.5-4b-encoder-phase1-alignment/stop.sh"
