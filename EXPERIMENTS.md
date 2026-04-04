# Training Experiments Log

Documents training experiments, architecture decisions, and lessons learned.

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

---

## Active & Complete Runs

### encoder-phase0 (✅ complete — checkpoint-9000)

**Goal**: CLIP-style contrastive alignment of the ResNet CNN encoder against frozen Qwen3.5-4B text embeddings. Forces CNN features onto the LLM's language manifold so a simple projector can bridge the gap.

**Architecture**:
- **CNN trunk**: `ChessEncoder` (19-ch input, 10 ResidualBlocks, 256 filters)
  - Input stem: `Conv2d(19→256, k=3) → BN → ReLU`
  - Trunk: 10× `ResidualBlock` (Conv→BN→ReLU→Conv→BN + skip)
  - 2D learnable positional encoding: additive `pos_file (1,256,1,8)` + `pos_rank (1,256,8,1)`
  - Output of trunk: `(B, 64, 256)` spatial features
- **Grid token head** (64 per-square tokens):
  - `proj`: Linear(256→2560) → GELU → Linear(2560→2560) → LayerNorm → `(B, 64, 2560)`
  - Each token encodes its square's piece, attackers, mobility, pawn structure — full board context via ResNet receptive field
- **Global token head** (1 board-summary token):
  - 1 learnable query `(1, 1, 256)` cross-attends over 64 spatial features in CNN hidden space
  - `K = Linear(256→256)`, `V = Linear(256→256)`, scaled dot-product attention → `(B, 1, 256)`
  - `global_proj`: Linear(256→2560) → GELU → Linear(2560→2560) → LayerNorm → `(B, 1, 2560)`
  - Aggregates board-level signals (material balance, king safety, eval tier)
- **Output**: `cat([grid, global], dim=1)` → `(B, 65, 2560)` — flat block injected as sentinel tokens into LLM
- **Text tower**: Qwen3.5-4B frozen (bfloat16), last-layer last-token hidden state as CLIP anchor
- **Loss**: Symmetric InfoNCE. `L = L_grid + L_global` with `global_loss_weight=1.0`
  - `L_grid`: 64 independent 2048-way classifications (per square position across batch)
  - `L_global`: 2048-way classification (whole-board summary token)
- Effective batch: 2048 (2× RTX 5090, DDP, cross-rank negatives via `all_gather`)

**Text labels** (generated on-the-fly from FEN):
- Grid: piece type + color + sq_color + bishop color complex + is_pinned + x-ray attacker + attack/defense counts + pawn structure
- Global: eval tier + top-3 SF15 term diffs + material imbalance + castling rights + en passant + check/checkmate/stalemate status

**Hyperparameters**:
- `lr=3e-4` (sqrt-scaled for B=2048 per CLIP literature), cosine schedule, warmup=50
- `tau`: learnable log-temperature, exp clamped [0.01, 0.5]; ~0.034 at step 5700
- `cache_maxsize=5M` (~25GB/rank) — embedding cache for Qwen anchor texts; FIFO eviction

**Training runs**:

| Steps | LR | Notes |
|-------|----|-------|
| 0→4500 | 1e-4 | Initial run; cache warming |
| 4500→5100 | 3e-4 | Reset scheduler; global loss spike step 4570 (5.58→5.44), recovered |
| 5100→5500 | 3e-4 | Memory fixes: CoW leak, cache save OOM, num_workers=0 |
| 5500→9000 | 3e-4 | Top-1 logging added; cache 5M; checkmate/stalemate labels |

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
- Cache eviction: LRU via `OrderedDict` + `move_to_end()` on hit; evicts least-recently-used when full
- DataLoader workers with `persistent_workers=True`: each forked worker inherits full RSS (~29GB) → fixed with `num_workers=0`

---

### qwen3.5-4b-encoder-phase1-alignment (ACTIVE — step ~9100/32079)

**Goal**: LLaVA-style alignment — map CLIP-trained CNN embeddings into Qwen3.5-4B's embedding space. Trains `cnn.proj` + `cnn.global_proj` (14M params) + LoRA r=64 (85M params) jointly. CNN trunk frozen; LLM base weights frozen.

**Why this works (vs. the failed pre-CLIP alignment runs)**: The CNN encoder was now CLIP-trained (encoder-phase0) — its features lie on the LLM's language manifold. The projector just needs to rotate/scale into the exact LLM embedding dim. Previously the CNN had no language supervision and the projector had no useful gradient signal.

**Architecture**:
- Flat 65-token sentinel block: 64 per-square `<|vision_pad|>` (token ID 151654) + 1 global summary token
- No square-label text anchors in the prompt — CNN carries all semantic meaning
- SF15 terms: Stockfish 15 classical eval (not SF16+) for interpretable term diffs (mobility, king safety, threats, …)
- Sentinel block placed before or after the question (50/50 random)
- Labels masked: system prompt, user turn, think block all `-100`; only assistant answer trains

**Hyperparameters**:
- LR=5e-5 (both proj and LoRA), cosine schedule, warmup=500 steps
- Batch: 8 per device × 2 GPU × 8 grad_accum = 128 effective
- max_seq_length=320 (p99=287 for flat 65-token board + question)

**Data**: `alignment_board_description.jsonl` — 4.58M train, 103k eval
- **Dual-source**: main tasks from `encoder_pretrain_1b.jsonl` (tail, unseen by encoder); `eval_tier` + `sf15_dominant` from `encoder_pretrain_sf15.jsonl` (tail, unseen)
- **20 tasks**: piece_abbr_at, piece_name_at, rank_contents, file_contents, count_piece_type, count_total_material, find_piece_type, is_square_occupied, attackers_at, is_pinned, mobility_at, side_to_move, castling_rights, en_passant, move_number, is_check, material_balance, who_is_better, eval_tier, sf15_dominant
- **Rare task top-up**: castling_rights, en_passant, is_check, is_pinned pool-boosted to 3500 positives; eval_tier/sf15_dominant from 500k SF15 FENs; all trimmed to ≤30% "none" answers

**Training runs**:

| Steps | Notes |
|-------|-------|
| 0→6500 | Initial run. Loss 2.33→0.07 over first ~2k steps, then stable 0.07–0.12 |
| 6500→7000 | Checkpoint-7000 corrupted (crashed mid-save — safetensors shared-tensor bug) |
| 6500→ | Resumed from checkpoint-6500 after fixing save bug; batch 16→8 per device |

**Key bugs fixed**:
1. **Token accuracy = 0**: `logit[i]` compared to `label[i]` without causal shift. Fix: shift both by 1, deduplicate with `_acc_logged_at_step`.
2. **Loss jump 0.07→0.4 on resume**: `compute_loss` override called `model(**inputs)` directly, bypassing DDP+liger kernel. Fix: use `super().compute_loss(return_outputs=True)`.
3. **safetensors crash on save**: Qwen ties `embed_tokens.weight` and `lm_head.weight`; safetensors refuses shared tensors. Fix: clone `lm_head.weight` before save, call `tie_weights()` after.
4. **`--init-proj` loss spike to 5.0**: Removed. Loading proj+LoRA without restoring optimizer state caused instability.
5. **start.sh resume path dropped**: `$2` (checkpoint path) was not captured. Fixed.

**Eval accuracy** (fixed 40 samples, 2 per task):

| Step  | Accuracy |
|-------|----------|
| 6500  | 90% (36/40) |
| ~9100 | 85–90% (stable) |

**Failing tasks analysis** (data labels verified correct — all failures are learning difficulty):
- `attackers_at`: outputs `none` for non-trivial cases — hardest spatial reasoning task, still learning
- `en_passant` (specific square): off by a file (`e3` vs `c3`) — very rare signal (0.1% of data)
- `move_number`: off by 1–3 — continuous value, hard to encode precisely in CNN global token
- `mobility_at`: off by ~2 — pseudo-legal vs legal move count edge cases
- `sf15_dominant`: wrong term when top-2 SF15 values are close in magnitude

**Status**: Active — step ~9100/32079, loss ~0.09, token_acc ~1.0, eval acc ~85–90%

---

## Superseded & Abandoned Experiments

### encoder-pretrain (v2 — superseded)
- **Goal**: Pretrain ResNet CNN board encoder via regression on SF15 eval terms + per-square piece classification
- **Architecture**: ChessEncoder (19-ch, 10 blocks, 256 filters) → (B,64,256) spatial tokens → 14 cross-attention read-out queries for SF15 terms + per-square piece classifier head
- **Loss**: `MSE(13 SF15 terms) + 3×MSE(eval_score) + CE(piece_labels per square)`
- **Data**: `encoder_pretrain_sf15.jsonl` — 656M board positions with SF15 eval term targets + piece labels
- **Result**: `checkpoints/encoder-pretrain/checkpoint-310000/encoder_weights.pt` ✅
- **Why superseded**: Pure regression objective (no language supervision) → CNN features in arbitrary 256-dim space with no relationship to LLM embedding manifold. Projector alignment fails from this base (see phase0-alignment runs below). encoder-phase0 retrains with CLIP-style InfoNCE against Qwen3.5-4B text embeddings, producing features that lie on the LLM's language manifold.
- **Engineering lessons**: IterableDataset + DDP — inject `_rank`/`_world_size`; each worker strides at `num_workers × world_size`. Resume skip via `itertools.islice` at C speed (~27s for 163M lines from page cache).

### qwen3.5-4b-encoder-pretrain-textbook (superseded)
- **Goal**: Causal LM pretraining on annotated chess textbook prose with CNN board embeddings injected at `[Position: FEN]` markers
- **Data**: `textbook_pretrain.jsonl` — 3,093 chunks, ~2.7M tokens from 64 classic textbooks
- **Result**: `checkpoints/qwen3.5-4b-encoder-pretrain-textbook/checkpoint-291` (epoch 1)
- **Finding**: `with_board` generates empty string after epoch 1 — CNN embeddings out-of-distribution for LLM without alignment. `no_board` generates coherent prose. Phase1-alignment must precede this.
- **Why superseded**: Trained with unaligned encoder (encoder-pretrain v2); will retrain as phase2-sft after phase1-alignment completes

### qwen3.5-4b-encoder-phase1-sft (superseded)
- **Goal**: SFT — joint task; CNN board embeddings in user prompt; assistant outputs annotated lines + coaching comment
- **Data**: `lines_joint_sft.jsonl` (28k textbook positions, SF15 annotations, Stockfish depth-18)
- **Result**: `checkpoints/qwen3.5-4b-encoder-phase1-sft/checkpoint-890` ✅
- **Why superseded**: Trained without phase1-alignment; will retrain as phase2-sft with aligned encoder

### qwen3.5-4b-grpo-phase1 (superseded)
- **Goal**: GRPO board-reading pretraining — teach position reading before coaching tasks
- **Base**: `Qwen/Qwen3.5-4B` base + LoRA (no encoder — board rendered as chess SVG image)
- **Data**: `grpo_phase1c_board_reading.jsonl` — 100k samples, 16,750 positions × 6 tasks
- **Key lessons**:
  - `beta=0.0` → entropy collapse by step ~735; `beta=0.01` → mode collapse ~700; `beta=0.05` + lr=5e-6 → stable
  - TRL PEFT resume bug: adapter weights reset on resume; fixed by explicit `set_peft_model_state_dict`
  - Symmetric length penalty required — asymmetric causes KL explosion (model outputs minimal answers)
  - Gameable tasks (>50% "none" answers) cause collapse regardless of diversity penalty
  - Eval OOM: TRL eval computes ref logprobs + fp32 logit cast → 15GB spike; disabled
- **Why superseded**: Encoder pipeline (phase1-alignment → phase2-sft → phase3-grpo) is the preferred path
- **Engineering lessons**: `beta=0.05` + lr=5e-6 → stable; `beta=0.0` → entropy collapse ~735; `beta=0.01` → mode collapse ~700. Symmetric length penalty required — asymmetric causes KL explosion. Drop any task where a single default answer is correct >50% of time.

### qwen3.5-4b-encoder-phase2-grpo (superseded)
- **Goal**: GRPO RL on joint coaching task; rewards for SF15 annotation accuracy, comment quality, move legality
- **Base**: phase1-sft checkpoint-890
- **Data**: `grpo_joint_prompts_sf15.jsonl` (14,950 Lichess positions with SF15 annotations)
- **Rewards**: R0 format, R1 legality, R3b SF15 annotation, RC_tone, RC_educ
- **Why superseded**: Based on unaligned phase1-sft; will become phase3-grpo after phase2-sft is retrained with aligned encoder

### qwen3.5-4b-encoder-phase0-alignment (abandoned — architecture mismatch)

- **Goal**: Align CNN encoder output into LLM embedding space via projector-only training (LLaVA-1.5 stage 1 analogy)
- **Architecture**: 2-layer MLP projector (256→2560→2560, GELU, LayerNorm) + LoRA r=64. CNN trunk + LLM frozen.
- **Data**: `alignment_board_description.jsonl` — 897k FENs × 14 tasks with square-label text anchors in prompt

#### Run 1 (projector-only, no LoRA — abandoned step ~540)
Loss plateaued at 0.54. Frozen LLM cannot route attention through foreign sentinel positions.

#### Run 2 (projector + LoRA, LR=5e-5/2e-4 — checkpoint-1000)
- Loss: 2.31→0.59 (warmup), then flat 0.59→0.48 over 900 steps
- Eval accuracy at step 600: 3/14 tasks (piece_abbr_at, count_piece_type, find_piece_type)
- Grad norm 0.015–0.022 — near-zero throughout; bootstrap deadlock confirmed
- Model thinking: "I need to parse this text format" — solving from text labels, ignoring CNN tokens

#### Root cause diagnosis

| Metric | Value | Expected |
|--------|-------|----------|
| Projected token norm | **9.37 ± 6.35** | ~0.66 (LLM embed norm) |
| Norm ratio | **14.3×** | ~1× |
| Best cosine sim to full vocab | 0.07–0.10 | >0.3 |
| CNN inter-token cosine sim | mean=0.21, std=0.28 | — (CNN itself fine) |

**Root cause — Bootstrap deadlock**: Square-label text anchors ("a1", "e4") give the LLM a text escape hatch — it ignores CNN sentinels and solves tasks from text. Near-zero gradient to projector → never converges.

#### Run 3 (projector-only + LayerNorm, LLM frozen)
Removed LoRA to eliminate escape hatch. But frozen LLM also cannot learn to attend sentinels — loss plateaued identically.

#### Run 4 (asymmetric LR 200×, system prompt — abandoned step ~228)
Projector LR=1e-3, LoRA LR=5e-6. Added "attend to vision token" system prompt. Loss=0.57 at step 210. No improvement.

**Final conclusion — architectural mismatch**: LLaVA's projector-only stage works because the ViT was CLIP-pretrained — features already lie on the LLM's language manifold. Our CNN was pretrained on SF15 regression (no language supervision) — features exist in an arbitrary 256-dim space with no relationship to any LLM manifold. Fix: retrain CNN with CLIP-style InfoNCE against Qwen3.5-4B text embeddings (→ encoder-phase0).

**Diagnostic checklist for future alignment attempts**:
- Cosine sim to vocab >0.3 = aligned; ~0.07–0.10 = random = failed regardless of loss
- Grad norm is a lagging indicator — 0.015 can look healthy while projector is deadlocked
- Loss plateau at 0.53+ with text anchors in prompt = bootstrap deadlock

**Vision encoder alignment lessons (from LLaVA, Flamingo, Qwen-VL, InternVL)**:
- Phase 0 (projector only, frozen encoder + LLM) is required before any LLM training
- Projection design: 2-layer MLP + GELU (LLaVA-1.5 standard); single linear is insufficient
- Causal LM loss on text tokens only; visual token positions always masked to -100
- Encoder unfreeze only in later phases with high-quality data + lower LR on the encoder

- **Status**: Abandoned — all 4 runs plateau at 0.53–0.61 eval loss

### qwen3.5-4b-phase1-coach-sft (abandoned)
- SFT directly on coaching comments (no line generation, no encoder)
- **Why abandoned**: Moved to encoder architecture + structured line output for verifiability
- **Checkpoints deleted**

### qwen3.5-4b-phase2-lines-sft (abandoned)
- SFT on line generation with `(purpose)` parenthetical annotations
- **Why abandoned**: Parenthetical format replaced by SF15 term annotation format
- **Checkpoints deleted**

### qwen3.5-4b-phase3-grpo (abandoned)
- GRPO on old line-generator format (no encoder, no SF15 terms)
- **Why abandoned**: Superseded by encoder-phase2-grpo with SF15 annotation rewards
- **Checkpoints deleted**

### qwen3.5-35b-sft (abandoned)
- Scale to 35B model; too expensive to iterate; 4B + encoder + RL shows better ROI
- **Checkpoints deleted**

---

## Dataset Lineage

| Dataset | Created by | Used by | Status |
|---------|-----------|---------|--------|
| `encoder_pretrain_sf15.jsonl` | `generate_encoder_pretrain_sf15.py` | encoder-phase0 (CLIP training), phase1-alignment (SF15 tasks) | **active** (656M records, 114GB) |
| `encoder_pretrain_1b.jsonl` | `generate_encoder_data.py` | phase1-alignment (main tasks) | **active** (source) |
| `alignment_board_description.jsonl` | `generate_alignment_board_description.py` | phase1-alignment train | **active** (4.58M records, 20 tasks) |
| `alignment_board_description_eval.jsonl` | same script | phase1-alignment eval | **active** (103k records) |
| `encoder_pretrain_sf15_eval.jsonl` | same script (1% split) | encoder-pretrain v2 eval | superseded |
| `lines_joint_sft.jsonl` | `generate_phase2_data.py` | encoder-phase1-sft | superseded |
| `grpo_joint_prompts_sf15.jsonl` | `generate_grpo_joint_prompts.py` | encoder-phase2-grpo | superseded |
| `textbook_pretrain.jsonl` | `generate_textbook_pretrain.py` | encoder-pretrain-textbook | superseded |
| `lines_30k.jsonl` | Lichess pipeline | source for grpo prompts | superseded |
| `grpo_phase1c_board_reading.jsonl` | `generate_grpo_phase1_board_reading.py` | grpo-phase1 | superseded |
| `lines_sft_thinking.jsonl` | `convert_lines_to_sft_thinking.py` | phase2-lines-sft | deleted |
| `train.jsonl` / `eval.jsonl` | textbook scrape | phase1-coach-sft | archived |

