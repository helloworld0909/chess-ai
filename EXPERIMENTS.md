# Training Experiments Log

Documents completed and abandoned training experiments.

---

## Active Training Pipeline

```
encoder-pretrain → qwen3.5-4b-encoder-phase1-sft → qwen3.5-4b-encoder-phase2-grpo
                                                  ↗
qwen3.5-4b-grpo-phase1 (board-reading pretraining, current)
```

### encoder-pretrain
- **Goal**: Pretrain ResNet CNN board encoder via regression on Stockfish classical evals
- **Data**: `encoder_pretrain_1b.jsonl` — ~100M board positions with SF15 eval targets (dataset named 1b, but current run used ~100M; scaling to 1B is a future step)
- **Result**: `checkpoints/encoder-pretrain/checkpoint-580083/encoder_weights.pt` ✅ (used by all downstream recipes)

### qwen3.5-4b-encoder-phase1-sft
- **Goal**: SFT — joint task; CNN board embeddings in user prompt for each move in Engine Key Lines; assistant outputs annotated lines + coaching comment
- **Base model**: `Qwen/Qwen3.5-4B` + pretrained encoder
- **Data**: `lines_joint_sft.jsonl` (28k textbook positions with SF15 annotations, Stockfish depth-18 lines)
- **Result**: `checkpoints/qwen3.5-4b-encoder-phase1-sft/checkpoint-890` ✅ (used by phase2 GRPO)

### qwen3.5-4b-grpo-phase1 (current)
- **Goal**: GRPO board-reading pretraining — teach the model to read chess positions from board images before tackling coaching tasks
- **Base model**: `Qwen/Qwen3.5-4B` base (not SFT checkpoint — encoder-phase1-sft was broken)
- **Data**: `grpo_phase1c_board_reading.jsonl` — 100k samples, 16,750 unique positions × 6 tasks
- **Tasks** (all rule-based, no LLM judge):
  - `piece_at`: what piece is on square X? → "white rook" / "empty"
  - `count_pieces`: how many white/black pieces total? → integer
  - `material_count`: count both sides → "white: N\nblack: N"
  - `piece_positions`: list squares of white/black knights (etc.) → square list or "none"
  - `in_check`: which squares give check? → square list or "none"
  - `attacked_by`: which pieces attack square X? → attacker square list or "none"
- **Rewards**: `0.01 * format + 0.99 * exact_match - len_penalty(max=0.5)`
  - Jaccard similarity for square-list tasks; exact match for counts/pieces
  - Format reward kept tiny (0.01) to prevent gaming
- **Config**: beta=0.05 (KL from base), lr=1e-5, warmup=50, t=1.0, r=64 LoRA
- **Key lessons learned**:
  - `beta=0.0` (pure REINFORCE) → entropy collapse by step ~735 despite t=1.3
  - `beta=0.01` → mode collapse by step ~700 (outputs `</think><answer>no</answer>`)
  - `beta=0.05` → stable so far; KL stays bounded
  - TRL PEFT resume bug: adapter weights reset on `resume_from_checkpoint`; fixed by explicit `set_peft_model_state_dict` before `trainer.train()`
  - Yes/no tasks (`in_check`, `attacked_by`) gameable — replaced with square-list answers
  - Eval OOM: TRL eval computes ref logprobs + Accelerate casts logits to fp32 → 15GB spike; disabled eval entirely
  - Checkpoint collapse: save every 50 steps, limit 50, never delete old ones
- **Status**: In progress (phase1c, step ~100)

### qwen3.5-4b-encoder-phase2-grpo
- **Goal**: GRPO RL on the joint coaching task; rewards for SF15 annotation accuracy, comment quality, move legality
- **Base model**: phase1-sft checkpoint-890
- **Data**: `grpo_joint_prompts_sf15.jsonl` (14,950 Lichess positions with SF15 term annotations)
- **Rewards**: R0 format, R1 legality, R3b SF15 annotation, RC_tone, RC_educ
- **Status**: Paused (blocked on board-reading pretraining)

---

## Abandoned Experiments

### qwen3.5-4b-phase1-coach-sft (abandoned)
- **Goal**: First approach — SFT directly on coaching comments (no line generation, no encoder)
- **Base model**: `Qwen/Qwen3.5-4B-Instruct`
- **Data**: `train.jsonl` / `eval.jsonl` — textbook coaching data
- **Why abandoned**: Moved to encoder architecture + structured line output for verifiability
- **Checkpoints deleted**: `checkpoints/qwen3.5-4b-phase1-coach-sft/`

### qwen3.5-4b-phase2-lines-sft (abandoned)
- **Goal**: Second approach — SFT on line generation (no encoder, parenthetical annotations)
- **Base model**: phase1-coach-sft checkpoint
- **Data**: `lines_sft_thinking.jsonl` — lines with `(purpose)` parenthetical annotations
- **Why abandoned**: Parenthetical format was replaced by SF15 term annotation format;
  encoder architecture made sentinel-based embedding injection possible
- **Checkpoints deleted**: `checkpoints/qwen3.5-4b-phase2-lines-sft/`

### qwen3.5-4b-phase3-grpo (abandoned)
- **Goal**: GRPO on the old line-generator format (no encoder, no SF15 terms)
- **Base model**: phase2-lines-sft checkpoint
- **Why abandoned**: Superseded by encoder-phase2-grpo which has SF15 annotation rewards
- **Checkpoints deleted**: `checkpoints/qwen3.5-4b-phase3-grpo/`

### qwen3.5-35b-sft (abandoned)
- **Goal**: Scale to 35B model for better coaching quality
- **Base model**: `Qwen/Qwen3.5-35B-A3B`
- **Why abandoned**: 35B model with encoder architecture too expensive to iterate; 4B
  with encoder+RL shows better ROI; 35B checkpoint never trained
- **Recipe deleted**: `recipes-train/qwen3.5-35b-sft/`

### qwen3.5-4b-encoder-phase1-sft-llm-merged (abandoned)
- **Goal**: Merge LoRA adapter into base weights for serving without PEFT
- **Why abandoned**: Not needed — serving uses PEFT adapter directly via encoder inference server
- **Checkpoint deleted**: `checkpoints/qwen3.5-4b-encoder-phase1-sft-llm-merged/`

---

## Dataset Lineage

| Dataset | Created by | Used by | Status |
|---------|-----------|---------|--------|
| `train.jsonl` / `eval.jsonl` | textbook scrape | qwen3.5-35b-sft, phase1-coach-sft | archived (used by 35B SFT recipe only) |
| `lines_sft_thinking.jsonl` | `convert_lines_to_sft_thinking.py` | phase2-lines-sft | **deleted** |
| `lines_joint_sft.jsonl` | `generate_phase2_data.py` | encoder-phase1-sft | **active** |
| `grpo_joint_prompts_sf15.jsonl` | `generate_grpo_joint_prompts.py` | encoder-phase2-grpo | **active** |
| `grpo_joint_prompts_sf15_eval50.jsonl` | sampled from above | encoder-phase2-grpo eval | **active** |
| `encoder_pretrain_1b.jsonl` | `generate_encoder_data.py` | encoder-pretrain | **active** |
| `lines_30k.jsonl` | Lichess pipeline | source for grpo prompts | **active** (source) |

---

## Key Architecture Decisions

- **Sentinel injection**: `<|vision_pad|>` (token ID 151654) replaces each SAN in user prompt; CNN injects board embeddings at these positions
- **Beta=0.05 (phase1)**: KL penalty from base model prevents entropy/mode collapse; with PEFT this is free (no second model copy — TRL disables adapter for ref logprobs)
- **Beta=0.0 (phase2)**: No reference model in GRPO (pure REINFORCE); saves ~4.3GB GPU memory
- **SF15 terms**: Use Stockfish 15 classical eval (not SF16+) for interpretable term diffs (mobility, king safety, threats, etc.)
- **Eval disabled (phase1)**: TRL eval computes ref logprobs + Accelerate fp32 cast of logits → 15GB OOM spike on 32GB cards; train reward logging is sufficient
- **Board-reading pretraining**: Train on verifiable position tasks before coaching to establish basic board perception
