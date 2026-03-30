# Claude Code Guidelines for Chess AI Tutor

## Quality Rules

- **Never lower LLM quality parameters** (`max_tokens`, `thinking_budget`, `temperature`) to save compute. Quality is paramount. The current values (`max_tokens=8192`, `thinking_budget=2048`) must not be reduced without explicit user instruction.

## Development Workflow

### Git Commits
- **Commit frequently**: Each small, complete feature gets its own commit
- **Only commit tested code**: All new code must have passing unit tests before committing
- **Commit message format**:
  ```
  <type>: <short description>

  <optional body explaining why, not what>
  ```
- **Types**: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`
- **Examples**:
  - `feat: add is_check to is_game_over return value`
  - `fix: correct stalemate FEN in verification tests`
  - `test: add unit tests for check detection`

### Testing Requirements
- Every new function needs unit tests
- Run tests using the test script:
  ```bash
  ./scripts/test.sh -v
  ```
- Test files mirror source structure:
  - `src/chess_mcp/stockfish.py` → `tests/test_stockfish.py`
  - `src/verification/legality.py` → `tests/test_verification.py`
- Async tests use `pytest-asyncio` with `@pytest.mark.asyncio` decorator

### Claude Code Hooks
- **Auto-formatting**: `scripts/hooks/format.sh` runs after Edit/Write on `.py` files
- Hook runs `ruff format` and `ruff check --fix --select I` (import sorting)
- Configured in `.claude/settings.local.json`

### IDE Diagnostics
- **Always fix IDE warnings** unless 100% certain they are false positives
- Check diagnostics with `mcp__ide__getDiagnostics` tool
- Common fixes:
  - Unused imports → remove them
  - `str | None` passed where `str` expected → add `assert x is not None` after validation
  - Type mismatches → fix the types or add proper narrowing
- Run type checker: `uv run mypy src/`

### Code Style
- Use type hints for all function signatures
- Docstrings for public functions (Google style)
- Imports: stdlib → third-party → local (separated by blank lines)

## Project Structure

```
chess-ai/
├── src/
│   ├── chess_mcp/        # MCP Server & Stockfish (✅ complete)
│   ├── verification/     # Move validation + GRPO rewards (✅ complete)
│   ├── encoder/          # Board tensor builder, MOVE_TOKEN constants (✅ complete)
│   └── tutor/            # Web UI, chess.com client, prompts (partial)
├── data/
│   ├── pipeline/         # Dataset generation scripts (✅ complete)
│   ├── raw/textbooks/    # Gutenberg + Lichess study PGNs (1,058 positions)
│   └── processed/        # Generated datasets (JSONL + textbook TXT)
├── recipes-train/
│   ├── encoder-pretrain/ # CNN encoder pretrain on SF15 (v2, training)
│   ├── qwen3.5-4b-encoder-phase1-sft/  # SFT: joint task (✅ checkpoint-890)
│   └── qwen3.5-4b-grpo-phase1/         # GRPO: board reading (in progress)
├── training/             # Shared model/collator for encoder+LLM (partial)
├── tests/                # Unit tests (99 tests)
└── scripts/              # Utility scripts
```

## Running the Project

Chess.com username is set in `.env` as `CHESS_COM_USERNAME`.

```bash
# Install dependencies (always use uv sync, never uv pip install)
uv sync --extra training

# Set Stockfish paths
export STOCKFISH_PATH="/home/zheng/.local/bin/stockfish"
export SF15_PATH="/home/zheng/.local/bin/stockfish-15"

# Run tests
./scripts/test.sh -v

# Launch encoder inference server (checkpoint-890, port 8200)
./recipes-train/qwen3.5-4b-encoder-phase1-sft/serve.sh

# Launch web UI (port 8080)
STOCKFISH_PATH=/home/zheng/.local/bin/stockfish ENCODER_SERVER_URL=http://localhost:8200 PYTHONPATH=src uv run uvicorn tutor.web:app --host 0.0.0.0 --port 8080

# Game review (fetches chess.com games, opens browser UI)
./scripts/start-review.sh MiracleRoguee
```

## Training Scripts

```bash
# Encoder pretrain (v2 — 2× RTX 5090, DDP)
./recipes-train/encoder-pretrain/start.sh
./recipes-train/encoder-pretrain/start.sh --resume checkpoints/encoder-pretrain/checkpoint-NNNN/checkpoint.pt
./recipes-train/encoder-pretrain/stop.sh

# Phase1 SFT (encoder + Qwen3.5-4B)
./recipes-train/qwen3.5-4b-encoder-phase1-sft/start.sh

# Phase1 GRPO (board reading, base Qwen3.5-4B + LoRA)
./recipes-train/qwen3.5-4b-grpo-phase1/start.sh
```

Logs: `/tmp/encoder-pretrain.log`, `/tmp/phase1-sft.log`, etc.

## Current TODO Items

### Training (Active)
- [ ] Encoder pretrain v2 completing (~9h remaining) → new `encoder_weights.pt`
- [ ] Update encoder-phase1-sft to use v2 encoder weights + retrain
- [ ] Build SFT dataset from textbooks: `data/processed/textbooks/*.txt` → `messages` format

### src/tutor/ (High Priority)
- [ ] `classifier.py` - Move classification with fine-tuned model
- [ ] `explainer.py` - Natural language explanation generator
- [ ] `session.py` - Game state & conversation history

### training/ (Medium Priority)
- [ ] `eval_chess.py` - Elo evaluation harness
- [ ] `merge_adapter.py` - Merge LoRA adapter with base model

### data/pipeline/ (Low Priority)
- [ ] `generate_textbook_sft.py` - Convert textbook TXTs to SFT messages format
- [ ] `validate_dataset.py` - Dataset validation & statistics

## Testing Checklist Before Commit

1. [ ] New feature has unit tests
2. [ ] All tests pass: `uv run pytest tests/ -v`
3. [ ] No import errors in new code
4. [ ] Stockfish tests require `PATH="$HOME/.local/bin:$PATH"`
