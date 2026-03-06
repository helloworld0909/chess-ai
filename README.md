# Chess AI Tutor

> **Work in progress.** The web UI and coaching pipeline are functional. The two-stage training pipeline (Phase 2 line-gen SFT → Phase 3 GRPO) is actively being developed.

A pedagogical chess analysis system combining Stockfish 17 NNUE evaluations with fine-tuned LLM coaching via a two-stage training pipeline (SFT → GRPO).

## Features

- **Move Analysis**: Classify moves as Best/Great/Good/Inaccuracy/Mistake/Blunder
- **Natural Language Coaching**: Human-like explanations of chess concepts
- **Web UI**: Browser-based game review with chess.com integration
- **MCP Integration**: Stockfish tools accessible via Model Context Protocol
- **Two-Stage Training**: Line-generator SFT cold-start → GRPO with verifiable chess rewards

## Installation

```bash
# Install dependencies
uv sync

# Install Stockfish 17
./scripts/install_stockfish.sh

# Run tests
./scripts/test.sh -v
```

## Quick Start

```bash
# Game review (fetches chess.com games, opens browser UI)
uv run chess-review <username>

# CLI tutor
STOCKFISH_PATH="$HOME/.local/bin/stockfish" uv run python -c \
  "import sys; sys.path.insert(0,'src'); from tutor.cli import main; main()"
```

---

## Project Structure

### `src/chess_mcp/` — MCP Server & Stockfish Integration ✅

- `stockfish.py` — Async Stockfish UCI wrapper
- `tools.py` — 6 MCP tools: `get_best_move`, `get_eval`, `get_threats`, `compare_moves`, `get_legal_moves`, `validate_move`
- `server.py` — MCP server entry point
- `representations.py` — FEN / ASCII / SVG / PNG converters

### `src/verification/` — Move Validation & GRPO Rewards ✅

- `legality.py` — Move legality validation (UCI, SAN, LAN)
- `tactical_loop.py` — LLM output verification against engine
- `rewards.py` — GRPO reward functions: R1 legality, R3 eval accuracy, R4a annotations, R5 relevance

### `src/tutor/` — Web UI & Coaching ✅

- `web.py` — FastAPI server with model toggle and compare mode
- `prompts.py` — Shared prompts: `SYSTEM_PROMPT`, `LINE_GENERATOR_SYSTEM_PROMPT`, formatting helpers
- `review.py` — CLI entry: fetches chess.com games → starts web server → opens browser
- `chesscom.py` — chess.com public API client + PGN parser

### `data/pipeline/` — Training Data Pipeline ✅

- `prepare_datasets.py` — Main pipeline: Stockfish analysis + LLM coaching annotations
- `convert_lines_to_sft.py` — Converts `lines_30k.jsonl` to `<line>` tag SFT format
- Outputs: `data/processed/lines_sft.jsonl` (28k train), `data/processed/lines_sft_eval.jsonl` (1.4k eval)

### `training/` — Shared Training Utilities

- `train.py` — Base SFT script (QLoRA, DDP, response-only masking)
- `lib.py` — Dataset helpers: `load_jsonl`, `load_jsonl_lines`, `format_dataset`, `strip_think_from_target`

---

## Training Pipeline

Three phases, all using `Qwen/Qwen3.5-4B-Instruct` + QLoRA 8-bit on 2× RTX 5090:

### Phase 1 — Coach SFT (`recipes-train/qwen3.5-4b-phase1-coach-sft/`)

Teaches the model to explain chess moves in natural language.

```bash
./recipes-train/qwen3.5-4b-phase1-coach-sft/start.sh
```

- Data: `data/processed/train.jsonl` (coach annotations)
- Output: `checkpoints/qwen3.5-4b-phase1-coach-sft/`

### Phase 2 — Line Generator SFT (`recipes-train/qwen3.5-4b-phase2-lines-sft/`)

Cold-starts the model on the `<line>` output format before RL.

```bash
./recipes-train/qwen3.5-4b-phase2-lines-sft/start.sh
./recipes-train/qwen3.5-4b-phase2-lines-sft/stop.sh
# Logs: /tmp/chess-lines-train.log
```

- Data: `data/processed/lines_sft.jsonl` (28k samples)
- Output: `checkpoints/qwen3.5-4b-phase2-lines-sft/`
- Format: `<line>LINE N: move (purpose) → move (purpose) | eval: <label></line>`

### Phase 3 — GRPO (`recipes-train/qwen3.5-4b-phase3-grpo/`)

Reinforcement learning with verifiable chess rewards. Starts from Phase 2 checkpoint.

```bash
./recipes-train/qwen3.5-4b-phase3-grpo/start.sh
```

- Rewards: R1 legality (0.25), R3 eval accuracy (0.30), R4a annotations (0.15), R5 relevance (0.10)
- Output: `checkpoints/qwen3.5-4b-phase3-grpo/`

---

## Tests

```bash
./scripts/test.sh -v        # runs all 99 tests with STOCKFISH_PATH set
```

| File | Tests |
|------|-------|
| `test_stockfish.py` | 16 |
| `test_mcp_tools.py` | 17 |
| `test_verification.py` | 40 |
| `test_representations.py` | 25 |
| `test_chesscom.py` | 1 |

---

## Configuration

```bash
STOCKFISH_PATH=/home/zheng/.local/bin/stockfish
STOCKFISH_DEPTH=20
STOCKFISH_THREADS=4
STOCKFISH_HASH_MB=256
```

vLLM server (Qwen3.5-35B-A3B, port 8100) is used for coaching data generation and the web UI's LLM backend. See `docker-compose.yml`.
