#!/usr/bin/env bash
# Launch ChessQwen3ForCausalLM via SGLang on GPU 1.
# Uses the sglang-serve venv which has the correct SGLang + CUDA version.
#
# Prerequisites:
#   1. Run merge_adapter.py to produce a merged model directory
#   2. Set MODEL_PATH below (or pass as first argument)
#
# Usage:
#   ./sglang/serve.sh [/path/to/chess-merged]
#
# To create the merged model first (use chess-ai venv, not sglang-serve):
#   python sglang/merge_adapter.py --checkpoint <ckpt> --output /tmp/chess-merged

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHESS_AI_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${1:-/tmp/chess-merged}"
shift 2>/dev/null || true  # consume $1 so $@ only has extra flags
LOG_PATH="${CHESS_LOG_PATH:-/tmp/chess-sglang.log}"
DP="${CHESS_DP:-2}"  # data parallelism replicas; 2 = use both GPUs

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    echo "Run: python sglang/merge_adapter.py --checkpoint <ckpt> --output $MODEL_PATH"
    exit 1
fi

# Activate sglang-serve venv
SGLANG_VENV="/home/zheng/workspace/sglang-serve/.venv"
PYTHON="$SGLANG_VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: sglang-serve venv not found at $SGLANG_VENV"
    exit 1
fi

# Export model path so ChessQwen3ForCausalLM._load_cnn_weights() can find encoder_weights.pt
export SGLANG_MODEL_PATH="$MODEL_PATH"

# Add chess_sglang package to Python path
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "Starting ChessQwen3ForCausalLM SGLang server..."
echo "  Model:  $MODEL_PATH"
echo "  GPUs:   ${CUDA_VISIBLE_DEVICES:-0,1} (dp=$DP)"
echo "  Port:   8300"
echo "  Log:    $LOG_PATH"

mkdir -p "$(dirname "$LOG_PATH")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONUNBUFFERED=1 \
CHESS_DEBUG_MM="${CHESS_DEBUG_MM:-0}" \
SGLANG_EXTERNAL_MODEL_PACKAGE=chess_sglang \
SGLANG_EXTERNAL_MM_MODEL_ARCH=ChessQwen3ForCausalLM \
SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=chess_sglang \
SGLANG_DISABLE_CUDNN_CHECK=1 \
SGLANG_MODEL_PATH="$MODEL_PATH" \
"$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8300 \
    --skip-server-warmup \
    --tp 1 \
    --dp "$DP" \
    --mem-fraction-static 0.90 \
    --max-running-requests 128 \
    --tool-call-parser qwen \
    --attention-backend triton \
    --trust-remote-code \
    "$@" 2>&1 | tee -a "$LOG_PATH"
