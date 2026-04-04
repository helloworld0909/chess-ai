# Chess AI Tutor

A chess coaching system built on a multimodal encoder+LLM architecture. A ResNet CNN board encoder is CLIP-aligned to Qwen3.5-4B's embedding space, then fine-tuned jointly to annotate engine key lines and produce coaching comments.

## Current Training Pipeline

```
encoder-phase0 (CLIP alignment, InfoNCE) ✅ checkpoint-9000
        ↓
phase1-alignment (proj+LoRA, 20 board-reading tasks) ← ACTIVE (step ~9100/32079)
        ↓
phase2-sft  (joint task: annotated lines + coaching comment)
        ↓
phase3-grpo (RL on SF15 annotation accuracy + comment quality)
```

### Architecture

The model injects a flat block of 65 `<|vision_pad|>` sentinel tokens (64 per-square + 1 global) into the LLM prompt. The CNN encoder maps the board tensor to `(65, 2560)` embeddings which replace the sentinel token embeddings before the LLM forward pass.

```
Board (FEN) ──→ ChessEncoder (ResNet, 10 blocks, 256 filters)
                    ├─ 64 per-square tokens ──→ proj MLP (256→2560) ──→ grid embeddings
                    └─ 1 global token ─────→ cross-attn + MLP ────→ global embedding
                                                        ↓
[system] [65 × <|vision_pad|>] [question]  ←── injected into LLM prompt
                                                        ↓
                                              Qwen3.5-4B + LoRA r=64
```

**Why CLIP first**: The CNN was initially pretrained on SF15 regression (no language supervision), leaving its features in an arbitrary space unrelated to the LLM's embedding manifold. Direct projector alignment failed — the LLM ignores CNN tokens and solves tasks from text instead (bootstrap deadlock). encoder-phase0 fixes this by contrastively aligning CNN features to Qwen3.5-4B text embeddings via InfoNCE before any projector training.

---

## Installation

```bash
uv sync --extra training
export STOCKFISH_PATH="$HOME/.local/bin/stockfish"
export SF15_PATH="$HOME/.local/bin/stockfish-15"
./scripts/test.sh -v
```

## Training

```bash
# Phase 1 alignment (active)
./recipes-train/qwen3.5-4b-encoder-phase1-alignment/start.sh
./recipes-train/qwen3.5-4b-encoder-phase1-alignment/start.sh --resume   # resume last checkpoint
./recipes-train/qwen3.5-4b-encoder-phase1-alignment/start.sh --resume checkpoints/.../checkpoint-N

# Monitor
tail -f /tmp/phase1-alignment.log

# encoder-phase0 CLIP training (complete)
./recipes-train/encoder-phase0/start.sh
```

---

## Project Structure

### `src/`

- `encoder/` — Board tensor builder, `BOARD_TOKEN` / `BOARD_TOKEN_ID` constants
- `verification/` — Move legality validation + GRPO reward functions (`rewards.py`)
- `chess_mcp/` — MCP server + async Stockfish wrapper
- `tutor/` — FastAPI web server, chess.com client, prompt templates

### `training/`

- `encoder_model.py` — `ChessLMWithEncoder`: wraps Qwen3.5-4B + ChessEncoder CNN; handles sentinel injection
- `encoder_collator.py` — Builds board tensors from FEN; injects at `<|vision_pad|>` positions
- `lib.py` — Shared training utilities

### `recipes-train/`

| Recipe | Purpose | Status |
|--------|---------|--------|
| `encoder-phase0/` | CLIP-style InfoNCE alignment of CNN to Qwen3.5-4B | ✅ checkpoint-9000 |
| `qwen3.5-4b-encoder-phase1-alignment/` | proj+LoRA alignment, 20 board-reading tasks | 🔄 step ~9100/32079 |
| `qwen3.5-4b-encoder-phase2-sft/` | SFT: annotated key lines + coaching comment | ⏳ pending |
| `qwen3.5-4b-encoder-phase3-grpo/` | GRPO: SF15 annotation + comment quality rewards | ⏳ pending |

### `data/pipeline/`

- `generate_alignment_board_description.py` — 5M board positions × 20 tasks → `alignment_board_description.jsonl`
- `generate_encoder_pretrain_sf15.py` — 656M positions with SF15 term labels → `encoder_pretrain_sf15.jsonl`
- `generate_phase2_data.py` — Textbook positions + Stockfish depth-18 → `lines_joint_sft.jsonl`
- `generate_grpo_joint_prompts.py` — Lichess positions + SF15 annotations → `grpo_joint_prompts_sf15.jsonl`
- `sf15_eval.py` — Stockfish 15 classical eval term wrapper

### `data/processed/` (active)

| File | Records | Used by |
|------|---------|---------|
| `alignment_board_description.jsonl` | 4.58M | phase1-alignment train |
| `alignment_board_description_eval.jsonl` | 103k | phase1-alignment eval |
| `encoder_pretrain_sf15.jsonl` | 656M | encoder-phase0 CLIP training, phase1-alignment SF15 tasks |
| `encoder_pretrain_1b.jsonl` | ~1B | phase1-alignment main tasks (source) |

---

## Phase 1 Alignment — 20 Tasks

The alignment dataset trains the model to answer direct board-reading questions using only the 65 CNN sentinel tokens — no text anchors. Tasks are split by what information they probe:

**Per-square** (probe grid tokens): `piece_abbr_at`, `piece_name_at`, `rank_contents`, `file_contents`, `count_piece_type`, `count_total_material`, `find_piece_type`, `is_square_occupied`, `attackers_at`, `is_pinned`, `mobility_at`

**Global** (probe summary token): `side_to_move`, `castling_rights`, `en_passant`, `move_number`, `is_check`, `material_balance`, `who_is_better`

**SF15-dependent** (from `encoder_pretrain_sf15.jsonl`): `eval_tier`, `sf15_dominant`

Current eval accuracy: ~85–90% at step 9100/32079.

---

## Phase 3 GRPO Rewards

Reinforcement learning from verifiable rewards — no LLM judge needed:

| Reward | Weight | Signal |
|--------|--------|--------|
| R0 format | 0.10 | `<line>` tags present |
| R1 legality | hard gate | All line moves legal (python-chess) |
| R_think | 0.15 | Non-empty think block with ≥2 moves |
| R3b sf15 | 0.35 | Think-block SF15 term interpretations match prompt diffs |
| RC_tone | 0.20 | Chess concept vocabulary in coaching comment |
| RC_educ | 0.20 | Grounded moves + causal reasoning in comment |

---

## Tests

```bash
./scripts/test.sh -v
```

---

## Experiments Log

See [EXPERIMENTS.md](EXPERIMENTS.md) for a full history of tried approaches, what failed, and why.
