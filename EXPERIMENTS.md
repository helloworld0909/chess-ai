# Training Experiments Log

Documents completed and abandoned training experiments.

---

## Active Training Pipeline

```
encoder-phase0 (CLIP alignment, InfoNCE) ✅ complete → checkpoint-9000
        ↓
qwen3.5-4b-encoder-phase1-alignment (proj+LoRA alignment, 20 tasks) ← ACTIVE (step ~9100/32079)
        ↓
qwen3.5-4b-encoder-phase2-sft         (board-reading Q&A + joint task)
        ↓
qwen3.5-4b-encoder-phase3-grpo        (RL on coaching quality)
```

### qwen3.5-4b-encoder-phase0-alignment (abandoned — architecture mismatch)

- **Goal**: Align the CNN encoder's output into the LLM's embedding space before any LLM training. Analogous to LLaVA-1.5 stage 1 — train only the projection layer while both the CNN encoder and the LLM are fully frozen.
- **Motivation**: Without this phase, CNN embeddings land in a foreign region of the LLM's embedding space (large-magnitude floats never seen during LLM pretraining). This causes the LLM to immediately predict EOS when sentinel tokens are present — confirmed experimentally on `checkpoint-291` of textbook pretrain: `with_board` always generates empty string while `no_board` generates coherent text.
- **Architecture**: 2-layer MLP projector (256→2560→2560, GELU, LayerNorm). LoRA r=64 on all LLM attention+MLP layers (85M trainable). CNN trunk frozen.
- **Data**: `alignment_board_description.jsonl` — 897,628 FEN positions × 14 board-reading tasks (piece lookup, counts, castling, en passant, side to move, file/rank contents, full piece lists)
- **Prompt format**: `_ANCHORED_BOARD` prepended to user message — 64 square labels each immediately followed by `<|vision_pad|>` sentinel in row-major order (a1..h8), matching CNN output token order exactly

#### Run 1 (projector-only, no LoRA — abandoned step ~540)
Only `cnn.proj` trained, LLM fully frozen. Eval accuracy ~0/8 at step 540. Loss plateaued at 0.54. Abandoned — frozen LLM cannot route attention through foreign sentinel positions.

#### Run 2 (projector + LoRA, LR=5e-5/2e-4 — checkpoint-1000)
- Loss curve: 2.31→0.59 in first 100 steps (warmup), then 0.59→0.48 over steps 100–1000 (near-flat plateau)
- Eval loss: 0.579 → 0.531 over 5 evals — essentially flat
- Eval accuracy at step 600: 3/14 tasks correct (piece_abbr_at, count_piece_type, find_piece_type)
- Grad norm: 0.015–0.022 throughout — near-zero, bootstrap deadlock confirmed
- Model thinking: "I need to parse this text format" — ignoring CNN tokens, solving from text labels only

#### Root cause diagnosis (checkpoint-1000 post-mortem)

| Metric | Value | Expected |
|--------|-------|----------|
| Projected token norm | **9.37 ± 6.35** | ~0.66 (LLM embed norm) |
| LLM embed token norm | 0.66 ± 0.07 | — |
| Norm ratio | **14.3×** | ~1× |
| Best cosine sim to full vocab | 0.07–0.10 | >0.3 for aligned tokens |
| Mean cosine sim to random 1k tokens | -0.004 | — |
| CNN inter-token cosine sim | mean=0.21, std=0.28 | — (CNN itself is fine) |

**Root cause — Bootstrap deadlock:** The LLM can partially answer board-reading questions from the text coordinate labels alone (e.g. "a1" tells it which square). This gives the LLM an "escape hatch": it ignores the CNN sentinel tokens (which land far from its embedding manifold) and solves partial tasks from text. Gradient flows primarily through the text path → projector receives near-zero gradient → never learns to map into the LLM's embedding space.

#### Run 3 (projector-only + LayerNorm, LLM frozen — abandoned)
Added `nn.LayerNorm(out_dim)` at projector output. Removed LoRA to eliminate text escape hatch. But with frozen LLM, there is no way for the model to learn to attend to sentinel positions — loss plateaued identically. Removing LoRA eliminates the escape hatch but also eliminates the capacity to exploit CNN tokens. A frozen LLM cannot adapt.

#### Run 4 (projector + LoRA at 200× asymmetric LR, system prompt — abandoned step ~228)
- Projector LR=1e-3, LoRA LR=5e-6 (200× slower) to prevent LoRA from learning text escape hatch before projector aligns
- Added system prompt: "attend to the vision token after each square label, do not parse labels as text"
- Added LayerNorm at projector output
- Result: loss=0.57 at step 210, eval_loss=0.609 at step 200. Grad norm ~0.02 — still near-zero. No improvement.
- Model still ignores CNN tokens even at step 200+. Asymmetric LR slowed LoRA but did not fix projector alignment.

#### Root cause — architectural mismatch (final conclusion)

**The LLaVA method requires a CLIP-style encoder. Our CNN was not trained like CLIP.**

LLaVA's "train only the projector" works because the ViT encoder was pretrained with CLIP: contrastive learning against text descriptions forces the visual features to lie on a manifold that is already semantically aligned with language. A 2-layer MLP projector just needs to rotate/scale that already-aligned manifold into the LLM's specific embedding space. The hard work (text↔vision alignment) was done during CLIP pretraining.

Our CNN encoder was pretrained on SF15 regression terms + piece classification (MSE + cross-entropy) — a purely numerical/categorical objective with **no language supervision whatsoever**. The CNN features encode chess evaluation signals (mobility, king safety, pawn structure, etc.) in an arbitrary 256-dim space that has no relationship to Qwen3.5-4B's token embedding manifold. A 2-layer MLP projector has nowhere near enough capacity or signal to learn that mapping from scratch via next-token prediction alone.

**The fix**: Retrain the CNN encoder with CLIP-style contrastive alignment against text — board positions paired with natural language descriptions → InfoNCE loss aligns the CNN's 64-token output with the LLM's text embedding space before any projection is needed.

- **Status**: Abandoned — all 4 runs plateau at 0.53–0.61 eval loss with near-zero useful gradient; fundamental architecture mismatch

### encoder-pretrain (v2 — complete)
- **Goal**: Pretrain ResNet CNN board encoder via regression on 13 Stockfish 15 classical eval terms + total eval score + per-square piece classification
- **Architecture**: ChessEncoder (19-ch, 10 blocks, 256 filters) → (B,64,256) spatial tokens → EncoderSF15Head (14 cross-attention queries) + piece_head (Linear 256→13 per square)
- **Loss**: `term_weight(1.0)×MSE(13 SF15 terms) + eval_weight(3.0)×MSE(eval_score) + piece_weight(1.0)×CE(piece_labels)`
- **Data**: `encoder_pretrain_sf15.jsonl` — 656M board positions with SF15 eval term targets + piece labels; eval set 6.6M samples
- **Dataset**: Streaming `IterableDataset` (zero RAM overhead); resume uses `islice` skip (~27s for 163M lines)
- **v1 result**: `checkpoints/encoder-pretrain/v1-checkpoint-580083/` (100M positions, scalar eval only) — superseded
- **v2 result**: `checkpoints/encoder-pretrain/checkpoint-310000/encoder_weights.pt` ✅
- **Status**: Abandoned as upstream — encoder-clip retrains the CNN from scratch with CLIP-style contrastive alignment, producing a better-aligned encoder. checkpoint-310000 still used by the old phase1-sft but will be superseded.

### qwen3.5-4b-encoder-pretrain-textbook
- **Goal**: Continued causal LM pretraining on annotated chess textbook prose. Each `[Position: FEN]` in the source text is replaced by 64 `<|vision_pad|>` sentinels; the frozen CNN injects spatial board embeddings at those positions during the forward pass.
- **Architecture**: CNN encoder frozen; LoRA r=64 on LLM (same rank as phase1-sft for checkpoint compatibility); causal LM loss on all text tokens; sentinel positions masked to -100.
- **Data**: `textbook_pretrain.jsonl` — 3,093 chunks, ~2.7M tokens from 64 classic textbooks (Chernev, Nimzowitsch, Vukovic, Capablanca, Fischer, Kasparov, Smyslov, Reshevsky, …)
- **Encoder checkpoint**: `checkpoints/encoder-pretrain/checkpoint-310000/encoder_weights.pt`
- **Result**: `checkpoints/qwen3.5-4b-encoder-pretrain-textbook/checkpoint-291` (epoch 1 complete)
- **Eval finding**: After epoch 1, `with_board` always generates empty string (EOS immediately). Root cause: CNN embeddings are out-of-distribution for the LLM — the phase0 alignment step was skipped. `no_board` generates coherent chess prose, confirming the LoRA+LLM is healthy. Board token lift is 0% at this stage; meaningful lift requires phase0 alignment first.
- **Status**: Epoch 1 complete; paused pending phase0 alignment implementation

### qwen3.5-4b-encoder-phase1-sft
- **Goal**: SFT — joint task; CNN board embeddings in user prompt for each move in Engine Key Lines; assistant outputs annotated lines + coaching comment
- **Base model**: `Qwen/Qwen3.5-4B` + pretrained encoder
- **Data**: `lines_joint_sft.jsonl` (28k textbook positions with SF15 annotations, Stockfish depth-18 lines)
- **Result**: `checkpoints/qwen3.5-4b-encoder-phase1-sft/checkpoint-890` ✅
- **Status**: Complete (superseded by revised pipeline — phase0 alignment must precede this)

### qwen3.5-4b-grpo-phase1 (paused)
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
- **Status**: Paused — superseded by encoder pipeline; phase0 alignment + encoder SFT is the preferred path

### qwen3.5-4b-encoder-phase2-grpo (blocked)
- **Goal**: GRPO RL on the joint coaching task; rewards for SF15 annotation accuracy, comment quality, move legality
- **Base model**: phase1-sft checkpoint-890
- **Data**: `grpo_joint_prompts_sf15.jsonl` (14,950 Lichess positions with SF15 term annotations)
- **Rewards**: R0 format, R1 legality, R3b SF15 annotation, RC_tone, RC_educ
- **Status**: Blocked — waiting for encoder-clip training to complete, then retrain phase1-sft with aligned encoder weights

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
| `textbook_pretrain.jsonl` | `generate_textbook_pretrain.py` | encoder-pretrain-textbook train | **active** (3,093 chunks, ~2.7M tokens) |
| `textbook_pretrain_eval.jsonl` | same script | encoder-pretrain-textbook eval | **active** |
| `alignment_board_description.jsonl` | `generate_alignment_board_description.py` | phase1-alignment train | **active** (4.58M records, 20 tasks, dual-source) |
| `alignment_board_description_eval.jsonl` | same script | phase1-alignment eval | **active** (103k records) |

---

## Key Architecture Decisions

### Phase0 alignment — what actually goes wrong (empirical)

From 4 runs of `qwen3.5-4b-encoder-phase0-alignment`, all plateauing at 0.53–0.61 eval loss:

- **Text escape hatch kills gradient to the projector.** When the LLM can partially solve the task from text tokens alone (square labels like "a1" carry positional meaning), it learns to ignore the CNN sentinel tokens. The projector then receives near-zero gradient and never converges.
- **Norm mismatch is a symptom, not the root cause.** Projected token norm 9.37 vs LLM embed norm 0.66 (14× gap). Adding LayerNorm at projector output fixes the norm but does not fix the direction — the features are still random in the LLM's embedding space.
- **Cosine sim to vocab is the right diagnostic.** Well-aligned vision tokens should have cosine sim >0.3 to semantically related text tokens. Cosine sim ~0.07–0.10 = random = alignment has failed regardless of loss curve.
- **Grad norm is a lagging indicator.** Grad norm 0.015–0.022 looked "healthy" but projector was receiving near-zero useful gradient. Loss plateau + tiny grad norm = bootstrap deadlock, not convergence.
- **The LLaVA method requires a CLIP-pretrained encoder.** See "CLIP-style encoder alignment" section below for the architectural fix.

### encoder-phase0 (active — step ~5700)

**Goal**: CLIP-style contrastive alignment of the ResNet CNN encoder against frozen Qwen3.5-4B text embeddings. Replaces the failed MLP projector phase0.

**Architecture**:
- CNN: `ChessEncoder` (26M params, 19-ch, 10 blocks, 256 filters) → (65, 2560) tokens via linear projection. 64 grid tokens + 1 global token.
- Text tower: Qwen3.5-4B frozen (bfloat16), last-layer last-token hidden state as anchor.
- Loss: Symmetric InfoNCE. `L = L_grid + L_global` with `global_loss_weight=1.0`.
  - `L_grid`: 64 independent 2048-way classifications (per square position across batch)
  - `L_global`: 2048-way classification (whole-board summary token)
- Effective batch: 2048 (2× RTX 5090, DDP, cross-rank negatives via `all_gather`)

**Text labels** (generated on-the-fly from FEN):
- Grid: piece type + color + sq_color + bishop color complex + is_pinned + x-ray attacker + attack/defense counts + pawn structure
- Global: eval tier + top-3 SF15 term diffs + material imbalance + castling rights + en passant + check/checkmate/stalemate status

**Hyperparameters**:
- `lr=3e-4` (sqrt-scaled for B=2048 per CLIP literature), cosine schedule, warmup=50
- `tau`: learnable log-temperature, exp clamped [0.01, 0.5]; currently ~0.034 at step 5700
- `cache_maxsize=5M` (~25GB/rank) — embedding cache for Qwen anchor texts; FIFO eviction

**Training runs**:

| Steps | LR | Notes |
|-------|----|-------|
| 0→4500 | 1e-4 | Initial run; cache warming |
| 4500→5100 | 3e-4 | Reset scheduler; global loss spike step 4570 (5.58→5.44), recovered |
| 5100→5500 | 3e-4 | Memory fixes: CoW leak, cache save OOM, num_workers=0 |
| 5500→ | 3e-4 | Top-1 logging added; cache 5M; checkmate/stalemate labels |

**Probe results** (linear piece-identity, 2000 positions, random baseline=0.077):

| Checkpoint | Overall | wP   | wN   | wB   | wR   | wQ   | wK   | bP   | bN   | bB   | bR   | bQ   | bK   |
|------------|---------|------|------|------|------|------|------|------|------|------|------|------|------|
| ckpt-4600  | 0.920   | 0.767| 0.918| 0.936| 0.796| 0.866| 0.896| 0.702| 0.902| 0.933| 0.673| 0.905| 0.844|
| ckpt-5100  | 0.921   | 0.756| 0.886| 0.961| 0.794| 0.919| 0.859| 0.720| 0.895| 0.964| 0.711| 0.859| 0.963|
| ckpt-5300  | 0.919   | 0.748| 0.916| 0.957| 0.811| 0.932| 0.904| 0.716| 0.837| 0.912| 0.769| 0.783| 0.839|

**Retrieval metrics** (tau-agnostic Top-1, 2048-way, random baseline=0.049%):

| Step  | Top-1 Grid | Top-1 Global |
|-------|------------|--------------|
| 5510  | 0.361      | 0.264        |
| 5700  | 0.367      | 0.283        |

Grid ~730× above random — genuine alignment confirmed, not tau-cheating.

**Key engineering lessons**:
- Cache save OOM: `dict(self._cache)` snapshot doubled 20GB RAM → fixed with `list(items())` + `join()`
- `embed_texts()`: must `del out, last_layer, enc` after each batch to free Qwen hidden states
- `OrderedDict.move_to_end()` on cache hit breaks Linux CoW → +335MB/min RAM growth → fixed by removing it (FIFO eviction)
- DataLoader workers with `persistent_workers=True`: each forked worker inherits full RSS (~29GB) → fixed with `num_workers=0`

**Status**: ✅ Complete — checkpoint-9000 used as encoder in phase1-alignment

---

### qwen3.5-4b-encoder-phase1-alignment (ACTIVE — step ~9100/32079)

**Goal**: LLaVA-style alignment — map CLIP-trained CNN embeddings into Qwen3.5-4B's embedding space. Trains `cnn.proj` + `cnn.global_proj` (14M params) + LoRA r=64 (85M params) jointly. CNN trunk frozen; LLM base weights frozen.

**Why this works (vs. the failed phase0-alignment runs)**: The CNN encoder was now CLIP-trained (encoder-phase0) — its features lie on the LLM's language manifold. The projector just needs to rotate/scale into the exact LLM embedding dim. Previously the CNN had no language supervision and the projector had no useful gradient signal.

**Architecture**:
- Flat 65-token sentinel block: 64 per-square `<|vision_pad|>` + 1 global summary token
- No square-label text anchors in the prompt — CNN carries all semantic meaning
- Sentinel block placed before or after the question (50/50 random)
- Labels masked: system prompt, user turn, think block all `-100`; only assistant answer trains

**Hyperparameters**:
- LR=5e-5 (both proj and LoRA), cosine schedule, warmup=500 steps
- Batch: 8 per device × 2 GPU × 8 grad_accum = 128 effective
- max_seq_length=320 (p99=287 for flat 65-token board + question)

**Data**: `alignment_board_description.jsonl` — 4.58M train examples, 103k eval
- **Dual-source**: main tasks from `encoder_pretrain_1b.jsonl` (tail, unseen by encoder); `eval_tier` + `sf15_dominant` from `encoder_pretrain_sf15.jsonl` (tail, unseen)
- **20 tasks**: piece_abbr_at, piece_name_at, rank_contents, file_contents, count_piece_type, count_total_material, find_piece_type, is_square_occupied, attackers_at, is_pinned, mobility_at, side_to_move, castling_rights, en_passant, move_number, is_check, material_balance, who_is_better, eval_tier, sf15_dominant
- **Rare task top-up**: castling_rights, en_passant, is_check, is_pinned pool-boosted to 3500 positives; eval_tier/sf15_dominant from 500k SF15 FENs; all trimmed to ≤30% "none" answers

**Training runs**:

| Steps | Notes |
|-------|-------|
| 0→6500 | Initial run from checkpoint. Loss 2.33→0.07 over first ~2k steps, then stable 0.07–0.12 |
| 6500→7000 | Checkpoint-7000 corrupted (crashed mid-save during safetensors shared-tensor bug) |
| 6500→ | Resumed from checkpoint-6500 after fixing save bug; batch lowered 16→8 per device (game using GPU 0) |

**Key bugs fixed this session**:
1. **Token accuracy = 0**: `logit[i]` was compared to `label[i]` without causal shift. Fix: shift both by 1, deduplicate per step with `_acc_logged_at_step`.
2. **Loss jump 0.07→0.4 on resume**: `compute_loss` override called `model(**inputs)` directly, bypassing DDP+liger kernel. Fix: use `super().compute_loss(return_outputs=True)`.
3. **safetensors crash on checkpoint save**: Qwen ties `embed_tokens.weight` and `lm_head.weight`; safetensors refuses shared tensors. Fix: clone `lm_head.weight` before save, call `tie_weights()` after.
4. **`--init-proj` loss spike to 5.0**: Removed. Loading proj+LoRA from a checkpoint without restoring optimizer state caused instability. Use `--resume` instead.
5. **start.sh resume path dropped**: `$2` (checkpoint path) was not captured. Fixed.

**Eval accuracy** (fixed 40 samples, 2 per task):

| Step  | Accuracy | Notes |
|-------|----------|-------|
| 6500  | 90%      | 36/40 |
| ~7000 | 85-87.5% | 34-35/40 (game on GPU slowing training) |
| ~9100 | ~85-90%  | Stable |

**Failing tasks analysis** (not data quality — all labels verified correct):
- `attackers_at`: model outputs `none` for non-trivial cases — hardest spatial reasoning task, still learning
- `en_passant` (specific square): off by a file (`e3` vs `c3`) — very rare signal (0.1% of data), model gets direction but not exact file
- `move_number`: off by 1-3 — continuous value hard to encode precisely in CNN global token
- `mobility_at`: off by ~2 — pseudo-legal vs legal move count edge cases
- `sf15_dominant`: wrong term when top-2 SF15 terms are close in magnitude — ambiguous ground truth

**Status**: Active — step ~9100/32079, loss ~0.09, token_acc ~1.0 (easy tasks), ~85-90% eval accuracy

---

### CLIP-style chess encoder alignment (implemented — see encoder-phase0 section above)

**Why LLaVA works and our approach fails:**

LLaVA's projector-only stage works because the ViT was pretrained with CLIP — contrastive learning against image-text pairs forces visual features onto a manifold that is already semantically aligned with language. The projector only needs to rotate/scale this already-aligned manifold into the LLM's specific embedding space.

Our CNN was pretrained on SF15 regression + piece classification — a purely numerical objective with no language supervision. The CNN features exist in an arbitrary 256-dim space with no relationship to any language model's embedding manifold. No projector can bridge this gap from next-token prediction signal alone.

**Plan: retrain the CNN encoder with CLIP-style contrastive alignment:**

1. **Text encoder**: Freeze Qwen3.5-4B and use it to embed board descriptions (piece lists, square contents, game state) into the LLM's hidden space. These text embeddings serve as CLIP's "language tower".
2. **Board encoder (vision tower)**: The ChessEncoder CNN produces a board-level representation (e.g., pool 64 square tokens → 1 board vector, or use a CLS token).
3. **Contrastive objective (InfoNCE)**: Minimize InfoNCE loss between board embedding and its paired text description embedding, maximize distance to negatives (other boards in the batch). This forces CNN features to align with the LLM's semantic geometry.
4. **After CLIP alignment**: The CNN features now lie on the LLM's language manifold. A simple 2-layer MLP projector (per LLaVA-1.5) should then work as intended: just rotate/scale 256-dim CNN space into 2560-dim LLM space.

**Architecture options:**
- **Board pooling**: Mean-pool 64 square tokens → (B, 256) board vector, project to LLM hidden dim, InfoNCE against text embedding. Simple, fast to iterate.
- **Multi-vector CLIP**: Align each of the 64 square tokens to the text embedding of the corresponding square's content ("white rook on a1"). Richer supervision; aligns spatial structure.

**Text supervision candidates (for the language tower):**
- Full board description (FEN → natural language: "White: Ra1 Kg1 ..., Black: ra8 kg8 ...")
- Per-square descriptions ("a1: white rook", "b1: empty", ...)
- SF15 term descriptions ("White has mobility advantage, weak king safety, ...")

### Vision encoder alignment (ViT → LLM, industry lessons)
Research into LLaVA, Flamingo, Qwen-VL, InternVL, LLaMA-3.2-Vision reveals a universal training pattern:
- **Phase 0 (alignment)**: Only the projection layer is trained; both encoder and LLM are frozen. This teaches the projection to map encoder features into the LLM's embedding space. Without this, encoder embeddings land in a foreign part of LLM embedding space and cause immediate EOS.
- **Phase 1+ (pretraining/SFT)**: LLM (or LoRA) is unfrozen; encoder usually stays frozen until a later high-quality data phase.
- **Projection design**: LLaVA-1.5 standardized on 2-layer MLP + GELU; single linear layers are insufficient. Q-Former (BLIP-2) compresses heavily (→32 tokens) but loses spatial detail; MLP with dynamic tiling (InternVL, LLaVA-NeXT) is now preferred.
- **Loss**: Causal LM on text tokens only in all major systems; visual/board token positions always masked to -100.
- **Encoder unfreeze**: Only done in later phases with high-quality data and lower LR on the encoder (Qwen-VL stage 2, InternVL2 stage 2, Qwen2-VL). For chess, the CNN is already domain-pretrained so freezing it longer is safe.
- **Application to us**: Our `ChessEncoder.proj` (single linear) needs to become a 2-layer MLP. Phase0 alignment (projector only, frozen encoder + frozen LLM) must precede textbook pretraining and SFT.

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
