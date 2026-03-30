# Training Experiments Log

Documents completed and abandoned training experiments.

---

## Active Training Pipeline

```
encoder-pretrain (v2, 656M positions) → qwen3.5-4b-encoder-phase1-sft → qwen3.5-4b-encoder-phase2-grpo
                                                                        ↗
qwen3.5-4b-grpo-phase1 (board-reading pretraining, phase1d, step ~145)
```

### encoder-pretrain (v2 — current)
- **Goal**: Pretrain ResNet CNN board encoder via regression on 13 Stockfish 15 classical eval terms + total eval score + per-square piece classification
- **Architecture**: ChessEncoder (19-ch, 10 blocks, 256 filters) → (B,64,256) spatial tokens → EncoderSF15Head (14 cross-attention queries) + piece_head (Linear 256→13 per square)
- **Loss**: `term_weight(1.0)×MSE(13 SF15 terms) + eval_weight(3.0)×MSE(eval_score) + piece_weight(1.0)×CE(piece_labels)`
- **Data**: `encoder_pretrain_sf15.jsonl` — 656M board positions with SF15 eval term targets + piece labels; eval set 6.6M samples
- **Dataset**: Streaming `IterableDataset` (zero RAM overhead); resume uses `islice` skip (~27s for 163M lines)
- **v1 result**: `checkpoints/encoder-pretrain/v1-checkpoint-580083/` (100M positions, scalar eval only) — superseded
- **v2 status**: Training in progress from checkpoint-80000 (eval_loss=0.043 → target <0.035); ~259k steps remaining (~9h)

### qwen3.5-4b-encoder-phase1-sft
- **Goal**: SFT — joint task; CNN board embeddings in user prompt for each move in Engine Key Lines; assistant outputs annotated lines + coaching comment
- **Base model**: `Qwen/Qwen3.5-4B` + pretrained encoder
- **Data**: `lines_joint_sft.jsonl` (28k textbook positions with SF15 annotations, Stockfish depth-18 lines)
- **Result**: `checkpoints/qwen3.5-4b-encoder-phase1-sft/checkpoint-890` ✅ (used by phase2 GRPO)

### qwen3.5-4b-grpo-phase1 (current)
- **Goal**: GRPO board-reading pretraining — teach the model to read chess positions from board images before tackling coaching tasks
- **Base model**: `Qwen/Qwen3.5-4B` base (not SFT checkpoint — encoder-phase1-sft was broken)
- **Data**: `grpo_phase1c_board_reading.jsonl` — 100k samples, 16,750 unique positions × 6 tasks
- **Current tasks** (phase1d, all rule-based, no LLM judge):
  - `piece_at`: what piece is on square X? → "white rook" / "empty"
  - `count_pieces`: how many white/black pieces total? → integer
  - `material_count`: count both sides → "white: N\nblack: N"
  - `piece_positions`: list squares of white/black piece type → square list or "none"
  - ~~`in_check`~~: removed — gameable via "none" default (~85% of positions not in check)
  - ~~`attacked_by`~~: removed — gameable via "none" default; also requires correct board reading first
- **Reward**: `0.01 * has_answer_tag + 0.99 * exact - len_penalty(max=0.5) - diversity_penalty(max=0.1)`
  - Jaccard similarity for square-list tasks; exact match for counts/pieces
  - Symmetric length penalty regardless of correctness — prevents KL explosion from short completions
  - Diversity penalty: mean pairwise Jaccard across 4 completions to discourage identical outputs
- **Config**: beta=0.05, lr=5e-6, warmup=50, t=1.0, r=64 LoRA, batch=8, num_generations=4
- **Key lessons learned**:
  - `beta=0.0` (pure REINFORCE) → entropy collapse by step ~735 despite t=1.3
  - `beta=0.01` → mode collapse by step ~700 (outputs `</think><answer>no</answer>`)
  - `beta=0.05` + lr=5e-6 → stable; KL stays 0.03–0.05
  - TRL PEFT resume bug: adapter weights reset on `resume_from_checkpoint`; fixed by explicit `set_peft_model_state_dict` before `trainer.train()`
  - Square-list tasks (`in_check`, `attacked_by`) still gameable when "none" is the majority answer — need to either oversample non-trivial positions or drop the task
  - Asymmetric length penalty (lenient for correct, strict for wrong) → KL explosion: model learns to output minimal `<answer>X</answer>`, huge distribution shift from verbose base → KL hit 1.3
  - Length penalty must be **symmetric** to keep policy anchored to base model verbosity
  - Diversity penalty helps but does not prevent collapse on gameable tasks
  - Eval OOM: TRL eval computes ref logprobs + Accelerate fp32 cast of logits → 15GB spike; disabled eval entirely
  - Checkpoint safety: save every 50 steps, limit=50; never delete old checkpoints
  - Model uses pixel-like grid coordinates `[4, 10]` to describe piece positions — Qwen3.5-4B vision not natively chess-coordinate-aware; `piece_at` works well (0.99 score) but spatial tasks harder
  - Board orientation confusion ("from Black's perspective") causes systematic errors; a learned skill
- **Checkpoints**: checkpoint-50, checkpoint-100 (clean), checkpoint-150 (partially trained with bad reward)
- **Status**: In progress (phase1d, step ~145 from checkpoint-100)

### qwen3.5-4b-encoder-phase2-grpo
- **Goal**: GRPO RL on the joint coaching task; rewards for SF15 annotation accuracy, comment quality, move legality
- **Base model**: phase1-sft checkpoint-890
- **Data**: `grpo_joint_prompts_sf15.jsonl` (14,950 Lichess positions with SF15 term annotations)
- **Rewards**: R0 format, R1 legality, R3b SF15 annotation, RC_tone, RC_educ
- **Status**: Paused (blocked on board-reading pretraining completing phase1d)

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
| `train.jsonl` / `eval.jsonl` | textbook scrape | qwen3.5-35b-sft, phase1-coach-sft | archived |
| `lines_sft_thinking.jsonl` | `convert_lines_to_sft_thinking.py` | phase2-lines-sft | **deleted** |
| `lines_joint_sft.jsonl` | `generate_phase2_data.py` | encoder-phase1-sft | **active** |
| `grpo_joint_prompts_sf15.jsonl` | `generate_grpo_joint_prompts.py` | encoder-phase2-grpo | **active** |
| `grpo_joint_prompts_sf15_eval50.jsonl` | sampled from above | encoder-phase2-grpo eval | **active** |
| `encoder_pretrain_1b.jsonl` | `generate_encoder_data.py` | encoder-pretrain v1 | archived (v2 supersedes) |
| `encoder_pretrain_sf15.jsonl` | `generate_encoder_pretrain_sf15.py` | encoder-pretrain v2 | **active** (656M records, 114GB) |
| `encoder_pretrain_sf15_eval.jsonl` | same script (1% split) | encoder-pretrain v2 eval | **active** (6.6M records) |
| `lines_30k.jsonl` | Lichess pipeline | source for grpo prompts | **active** (source) |
| `grpo_phase1c_board_reading.jsonl` | `generate_grpo_phase1_board_reading.py` | grpo-phase1 train | **active** |
| `grpo_phase1c_board_reading_eval.jsonl` | same script | grpo-phase1 eval (disabled) | **active** |

---

## Key Architecture Decisions

### Encoder pipeline (phase2)
- **Sentinel injection**: `<|vision_pad|>` (token ID 151654) replaces each SAN in user prompt; CNN injects board embeddings at these positions
- **Beta=0.0 (phase2)**: No reference model in GRPO (pure REINFORCE); saves ~4.3GB GPU memory
- **SF15 terms**: Use Stockfish 15 classical eval (not SF16+) for interpretable term diffs (mobility, king safety, threats, etc.)

### Encoder pretraining (v2)
- **Piece classification loss**: Forces each of 64 spatial tokens to encode its square's piece type — provides spatial grounding for downstream board reading
- **Cross-attention readout**: 14 learned queries attend over 64 spatial tokens in 256-dim space; no LLM involved during pretrain
- **Resuming large streaming datasets**: Cannot skip lines in O(1) without an index; `itertools.islice` with `start_line = step × batch × world_size` skips at C speed (~27s for 163M lines from page cache)
- **Fresh optimizer on resume = bad**: Stale Adam m/v from converged checkpoint + peak LR → loss spikes to 0.09+; use full state restore or fresh init from early checkpoint
- **IterableDataset + DDP**: Inject `_rank` / `_world_size` on the dataset object; each worker strides at `global_stride = num_workers × world_size` using `islice(f, my_slot, None, global_stride)`

### Board-reading GRPO (phase1)
- **No encoder**: `Qwen/Qwen3.5-4B` base with LoRA only — board images rendered from FEN via `chess.svg` + `cairosvg`, passed as vision input
- **Beta=0.05**: KL penalty from base model prevents entropy/mode collapse; with PEFT this is free (no second model copy — TRL disables adapter for ref logprobs)
- **Eval disabled**: TRL eval computes ref logprobs + Accelerate fp32 cast of logits → 15GB OOM spike on 32GB cards; train reward logging is sufficient
- **Symmetric length penalty**: Must apply to all completions regardless of correctness — asymmetric version causes KL explosion as model learns to output minimal answers
- **Task gamability**: Avoid tasks where a single default answer (e.g. "none") is correct >50% of the time — model collapses to that answer. Either oversample hard positions or remove the task
- **Diversity penalty**: Mean pairwise Jaccard across generation group × 0.1 max; helps discourage identical completions but insufficient alone against gameable tasks
