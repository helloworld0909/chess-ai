#!/bin/bash
# Start vLLM server — Qwen3.5-4B-Thinking-2507 + chess-tutor LoRA
# Port: 8100  |  Tensor parallel: 2  |  Max context: 8192
#
# API model names:
#   "chess-tutor"                   → fine-tuned (with LoRA adapter)
#   "Qwen/Qwen3.5-4B-Instruct"  → base model (for comparison)

set -e
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="$(dirname "$(dirname "$RECIPE_DIR")")"

ADAPTER_DIR="$PROJECT_DIR/checkpoints/qwen3.5-4b-phase2-lines-sft/checkpoint-350"
if [[ ! -d "$ADAPTER_DIR" ]]; then
    echo "Error: LoRA adapter not found at $ADAPTER_DIR"
    echo "Expected Phase 2 line-generator SFT checkpoint at: $ADAPTER_DIR"
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -q '^chess-ai-vllm-4b-lora$'; then
    echo "Already running. Logs: docker logs -f chess-ai-vllm-4b-lora"
    exit 0
fi

echo "Starting Qwen3.5-4B-Thinking-2507 + LoRA on port 8100..."
echo "Adapter: $ADAPTER_DIR"
docker compose -f "$RECIPE_DIR/docker-compose.yml" up -d

echo ""
echo "Health check : curl http://localhost:8100/health"
echo "View logs    : docker logs -f chess-ai-vllm-4b-lora"
echo ""
echo "Fine-tuned : curl http://localhost:8100/v1/chat/completions -d '{\"model\":\"chess-tutor\",...}'"
echo "Base model : curl http://localhost:8100/v1/chat/completions -d '{\"model\":\"Qwen/Qwen3.5-4B-Instruct\",...}'"
