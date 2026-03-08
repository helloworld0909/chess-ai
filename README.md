# Chess AI Tutor

A chess analysis system combining Stockfish evaluations with a fine-tuned encoder+LLM model for move coaching. The model uses a CNN board encoder to inject positional embeddings into the LLM, then learns to annotate engine key lines and produce coaching comments via SFT + GRPO.

## Features

- **Move Analysis**: Classify moves via Stockfish (Best/Inaccuracy/Mistake/Blunder)
- **Encoder-augmented LLM**: ResNet CNN encodes board states → embeddings injected at `<|vision_pad|>` tokens in the prompt
- **SF15 Term Annotations**: Each engine key line move annotated with classical eval term diffs (mobility, king safety, threats, etc.)
- **Natural Language Coaching**: Model produces structured line analysis + coaching comment grounded in SF15 terms
- **Web UI**: Browser-based game review with chess.com integration
- **MCP Integration**: Stockfish tools via Model Context Protocol

## Installation

```bash
uv sync
./scripts/test.sh -v
```

Set Stockfish path:
```bash
export STOCKFISH_PATH="$HOME/.local/bin/stockfish"
export SF15_PATH="$HOME/.local/bin/stockfish-15"
```

## Quick Start

```bash
# Game review (fetches chess.com games, opens browser UI)
uv run chess-review <username>

# Launch encoder inference server (phase1-sft checkpoint, port 8200)
./recipes-train/qwen3.5-4b-encoder-phase1-sft/serve.sh

# Launch web UI (port 8080)
STOCKFISH_PATH=/home/zheng/.local/bin/stockfish \
ENCODER_SERVER_URL=http://localhost:8200 \
PYTHONPATH=src uv run uvicorn tutor.web:app --host 0.0.0.0 --port 8080
```

---

## Project Structure

### `src/`

- `chess_mcp/` — MCP server + async Stockfish wrapper (6 tools: get_best_move, get_eval, compare_moves, ...)
- `verification/` — Move legality validation + GRPO reward functions (`rewards.py`)
- `tutor/` — FastAPI web server, chess.com client, prompt templates
- `encoder/` — Board tensor builder (`MOVE_TOKEN`, `MOVE_TOKEN_ID` constants)

### `training/`

- `encoder_model.py` — `ChessLMWithEncoder`: Qwen3.5-4B + ResNet CNN (72M params), PEFT LoRA wrapper
- `encoder_collator.py` — Builds board tensors from FEN + SAN sequences; injects at `<|vision_pad|>` positions
- `lib.py` — Shared SFT utilities: `load_jsonl_lines`, `strip_think_from_target`, `make_training_args`

### `recipes-train/`

| Recipe | Purpose | Status |
|--------|---------|--------|
| `encoder-pretrain/` | CNN encoder regression on SF15 evals | ✅ Done (`checkpoint-580083`) |
| `qwen3.5-4b-encoder-phase1-sft/` | SFT: joint task with engine key lines + coaching comment | ✅ Done (`checkpoint-890`) |
| `qwen3.5-4b-encoder-phase2-grpo/` | GRPO: SF15 annotation accuracy + comment quality rewards | 🔄 Active |

### `data/pipeline/`

- `generate_phase2_data.py` — Textbook positions + Stockfish depth-18 → `lines_joint_sft.jsonl`
- `generate_grpo_joint_prompts.py` — Lichess `lines_30k.jsonl` + SF15 annotations → `grpo_joint_prompts_sf15.jsonl`
- `generate_encoder_data.py` — Board positions for encoder pretraining
- `sf15_eval.py` — Stockfish 15 classical eval term wrapper (`get_sf15_eval(fen)`)

### `data/processed/` (active datasets)

| File | Used by |
|------|---------|
| `encoder_pretrain_1b.jsonl` | encoder-pretrain |
| `lines_joint_sft.jsonl` | phase1-sft |
| `lines_joint_sft_eval.jsonl` | phase1-sft eval |
| `grpo_joint_prompts_sf15.jsonl` | phase2-grpo (14,950 train) |
| `grpo_joint_prompts_sf15_eval50.jsonl` | phase2-grpo eval (50 samples) |
| `lines_30k.jsonl` | source for grpo prompts |

---

## Training Pipeline

### Architecture

```
User prompt:
  ## Engine Key Lines
  PLAYED LINE: <|vision_pad|>Ne5 [mobility +0.32; threats +0.18] → <|vision_pad|>d4 [...] | eval: equal
  Line 1: <|vision_pad|>Nd5 [king safety +0.21] → ...

  ↑ CNN injects board embeddings at each <|vision_pad|> token

Assistant output:
  <think>
  PLAYED: Ne5 → d4 | eval: equal
    Ne5: mobility +0.32, threats +0.18 → centralises knight, creates tactical pressure
    ...
  </think>
  <line>LINE 1: Ne5 (centralise knight) → d4 (gain space) | eval: equal</line>
  [coaching comment]
```

### Phase 1 — SFT (`qwen3.5-4b-encoder-phase1-sft`)

Cold-starts the joint task: given board + student move + engine key lines with SF15 annotations, output annotated lines + coaching comment.

```bash
./recipes-train/qwen3.5-4b-encoder-phase1-sft/start.sh
```

### Phase 2 — GRPO (`qwen3.5-4b-encoder-phase2-grpo`)

Reinforcement learning from verifiable rewards. Starts from phase1 checkpoint-890.

```bash
./recipes-train/qwen3.5-4b-encoder-phase2-grpo/start.sh
./recipes-train/qwen3.5-4b-encoder-phase2-grpo/start.sh --resume   # resume last checkpoint
```

**Rewards** (6 total, all free — no LLM judge):

| Reward | Weight | Signal |
|--------|--------|--------|
| R0 format | 0.10 | `<line>` tags present |
| R1 legality | hard gate | All line moves legal (python-chess) |
| R_think | 0.15 | Non-empty think block with ≥2 moves |
| R3b sf15 | 0.35 | Think-block SF15 term interpretations match prompt diffs |
| RC_tone | 0.20 | Chess concept vocabulary in coaching comment |
| RC_educ | 0.20 | Grounded moves + causal reasoning in comment |

Key config: `beta=0.0` (no reference model, pure REINFORCE), 2-GPU DDP, 50-sample eval set.

---

## GRPO Reward Details

`R3b sf15_annotation` is the primary learning signal. For each move in the prompt's Engine Key Lines that has notable SF15 term diffs (|delta| ≥ 0.10), it checks whether the model's `<think>` block interprets the top term in the correct direction using a vocabulary mapping (`_SF15_TERM_VOCAB` in `rewards.py`).

Example: prompt shows `Ne5 [mobility +0.32]` → model says "improves piece activity" → +1.0. Model says "restricts pieces" → −1.0. Model ignores the term → 0.0.

---

## Tests

```bash
./scripts/test.sh -v   # 288 tests
```

| File | Coverage |
|------|---------|
| `test_stockfish.py` | Stockfish wrapper |
| `test_mcp_tools.py` | MCP tools |
| `test_verification.py` | Move legality |
| `test_representations.py` | Board display |
| `test_rewards.py` | All 6 GRPO reward functions |
| `test_encoder.py` | Board tensors, CNN forward pass |
| `test_chesscom.py` | chess.com API client |

---

## Experiments Log

See [EXPERIMENTS.md](EXPERIMENTS.md) for a history of tried approaches and why each was abandoned or superseded.
