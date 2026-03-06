#!/bin/bash
# Stop vLLM server — Qwen3.5-4B-Thinking-2507 + LoRA

set -e
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="$(dirname "$(dirname "$RECIPE_DIR")")"

echo "Stopping Qwen3.5-4B-Thinking-2507 + LoRA vLLM server..."
docker compose -f "$RECIPE_DIR/docker-compose.yml" down
echo "Done."
