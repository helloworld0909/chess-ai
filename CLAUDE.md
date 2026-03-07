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
│   ├── verification/     # Move validation (✅ complete)
│   └── tutor/            # CLI interface (partial)
├── data/pipeline/        # Training data scripts (✅ complete)
├── training/             # Fine-tuning setup (partial)
├── tests/                # Unit tests (99 tests)
└── scripts/              # Utility scripts
```

## Running the Project

Chess.com username is set in `.env` as `CHESS_COM_USERNAME`.

```bash
# Install dependencies
uv sync

# Set Stockfish path (required for engine tests)
export STOCKFISH_PATH="/home/zheng/.local/bin/stockfish"

# Run tests
STOCKFISH_PATH="/home/zheng/.local/bin/stockfish" uv run pytest tests/ -v

# Run CLI tutor
STOCKFISH_PATH="/home/zheng/.local/bin/stockfish" uv run python -c "import sys; sys.path.insert(0,'src'); from tutor.cli import main; main()"

# Launch encoder inference server (checkpoint-840, port 8200)
./recipes-train/qwen3.5-4b-encoder-phase1-sft/serve.sh

# Launch web UI (port 8080)
STOCKFISH_PATH=/home/zheng/.local/bin/stockfish ENCODER_SERVER_URL=http://localhost:8200 PYTHONPATH=src uv run uvicorn tutor.web:app --host 0.0.0.0 --port 8080

# Or via chess-review (fetches chess.com games for CHESS_COM_USERNAME)
./scripts/start-review.sh MiracleRoguee
```

## Current TODO Items

### src/tutor/ (High Priority)
- [ ] `classifier.py` - Move classification with fine-tuned model
- [ ] `explainer.py` - Natural language explanation generator
- [ ] `session.py` - Game state & conversation history

### training/ (Medium Priority)
- [ ] `eval_chess.py` - Elo evaluation harness
- [ ] `merge_adapter.py` - Merge LoRA adapter with base model

### data/pipeline/ (Low Priority)
- [ ] `validate_dataset.py` - Dataset validation & statistics

## Testing Checklist Before Commit

1. [ ] New feature has unit tests
2. [ ] All tests pass: `uv run pytest tests/ -v`
3. [ ] No import errors in new code
4. [ ] Stockfish tests require `PATH="$HOME/.local/bin:$PATH"`
