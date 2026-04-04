#!/usr/bin/env bash
# Run encoder alignment probes on a checkpoint (CPU-only, GPU untouched).
#
# Usage:
#   ./recipes-train/encoder-phase0/probe.sh checkpoints/encoder-clip/checkpoint-4200/checkpoint.pt
#   ./recipes-train/encoder-phase0/probe.sh checkpoints/encoder-clip/checkpoint-4200/checkpoint.pt --n-positions 2000

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${1:?Usage: $0 <checkpoint.pt> [extra args]}"
shift

CUDNN="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
NCCL="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
CUSPARSE="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib"

CUDA_VISIBLE_DEVICES="" \
LD_LIBRARY_PATH="$CUDNN:$NCCL:$CUSPARSE:${LD_LIBRARY_PATH:-}" \
uv run python recipes-train/encoder-phase0/probe_checkpoint.py \
  --checkpoint "$CHECKPOINT" "$@"
