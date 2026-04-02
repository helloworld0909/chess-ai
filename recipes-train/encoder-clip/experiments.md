# Chess-CLIP Encoder Alignment — Experiment Log

> **Keep this file up to date.** Record every significant change: label updates, hyperparameter changes, new probe results, architecture decisions, and bugs found/fixed. This is the single source of truth for what was tried and why.
>
> **Probe command** (CPU-only, GPU left for training):
> ```bash
> CUDA_VISIBLE_DEVICES="" uv run python recipes-train/encoder-clip/probe_checkpoint.py \
>   --checkpoint checkpoints/encoder-clip/checkpoint-NNNN/checkpoint.pt \
>   --data data/processed/encoder_pretrain_sf15_eval.jsonl \
>   --n-positions 2000
> ```

## Architecture

- **Encoder**: ResNet CNN (26M params) — 19-channel board tensor → (65, 2560) tokens
  - 64 grid tokens (one per square) + 1 global token
  - `out_dim=2560` matches Qwen3.5-4B hidden size
- **Text tower**: Qwen3.5-4B (frozen, bfloat16) — last-layer last-token hidden state
- **Loss**: Symmetric InfoNCE
  - `L_grid`: per-square position, 64 independent 2048-way classifications
  - `L_global`: whole-board, 2048-way classification
  - `L = L_grid + L_global` (`global_loss_weight=1.0`)
- **Effective batch**: 2048 (2× RTX 5090, DDP, cross-rank negatives)
- **Text labels generated on-the-fly** from FEN via `describe_square()` + `build_global_description()`

---

## Label Evolution

### v1 (initial)
- Grid: piece type + color + basic attack/defense counts
- Global: eval tier + top-3 SF15 term diffs
- Issue: `max_length=72` silently truncated global labels (max was 129 tokens)

### v2 (commit 0a11ad5)
- Grid additions: `sq_color` (light/dark), bishop color complex, `is_pinned`, x-ray attacker
- Global: redesigned material summary → imbalance-only format (`"white +1R"` instead of full counts), reduces max tokens 129→89
- Fixed `max_length=72→96`
- Added check status: `"King is in check!"`
- Added checkmate/stalemate: `"Checkmate!"` / `"Stalemate!"`

---

## Training Runs

### Run 1 — LR=1e-4, B=2048, steps 0→4500
- Config: `lr=1e-4`, `cache_maxsize=2M`, `max_steps=10000`
- Observations: loss declining steadily, cache warming up slowly
- Stopped to increase LR and cache

### Run 2 — LR=3e-4, reset scheduler, steps 4500→5100
- Config: `lr=3e-4`, `cache_maxsize=4M`, `max_steps=20000`
- `--reset-scheduler`: cosine restarts from peak at step 4500
- Global loss spike at step 4570 (5.58) due to LR increase, recovered by step 4600 (5.44)
- System OOM crash at ~step 5200 due to cache save memory spike

### Run 3 — Memory fixes, steps 5100→5500
- Fixed `EmbeddingCache.save()`: `join()` instead of skip + `list(items())` instead of `dict()` copy
- Fixed `embed_texts()`: `del out, last_layer, enc` after each batch
- Removed `move_to_end` on cache hit → eliminated CoW page-breaking (~335MB/min RAM growth)
- Set `num_workers=0` (DataLoader workers each inherited ~29GB RSS via fork)
- Bumped cache to 5M entries (~25GB/rank)
- Added checkmate/stalemate to global labels

### Run 4 (current) — Top-1 logging, steps 5500→
- Added tau-agnostic Top-1 retrieval accuracy to every log line: `top1=grid/global`
- Resumed from checkpoint-5500

---

## Probe Results (linear piece-identity, CPU)

All probes on 2000 positions from `encoder_pretrain_sf15_eval.jsonl`.
Random baseline: 1/13 = 0.077.

| Checkpoint | Overall | wP   | wN   | wB   | wR   | wQ   | wK   | bP   | bN   | bB   | bR   | bQ   | bK   |
|------------|---------|------|------|------|------|------|------|------|------|------|------|------|------|
| ckpt-1000  | —       | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| ckpt-2500  | —       | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    |
| ckpt-4600  | 0.920   | 0.767| 0.918| 0.936| 0.796| 0.866| 0.896| 0.702| 0.902| 0.933| 0.673| 0.905| 0.844|
| ckpt-5100  | 0.921   | 0.756| 0.886| 0.961| 0.794| 0.919| 0.859| 0.720| 0.895| 0.964| 0.711| 0.859| 0.963|
| ckpt-5300  | 0.919   | 0.748| 0.916| 0.957| 0.811| 0.932| 0.904| 0.716| 0.837| 0.912| 0.769| 0.783| 0.839|

Notes:
- Rooks (wR/bR) consistently hardest — most positionally ambiguous pieces
- ckpt-5300 slight regression on bK/bB likely due to `--reset-scheduler` LR spike at step 4500
- Overall accuracy plateauing ~0.92; individual class variance still high → training ongoing

---

## Retrieval Metrics (tau-agnostic, 2048-way)

Random baseline: 1/2048 ≈ 0.049%

| Step  | Top-1 Grid | Top-1 Global | Notes |
|-------|------------|--------------|-------|
| 5510  | 0.361      | 0.264        | First reading after adding metric |
| 5680  | 0.368      | 0.286        | Continuing to improve |
| 5700  | 0.367      | 0.283        | Stable |

- Grid Top-1 ~730× above random → genuine alignment, not tau-cheating
- Global lower than grid (harder task: whole-board summary vs per-square)
- Both trending upward — model is truly learning

---

## Key Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| InfoNCE over regression | Contrastive loss naturally handles the scale mismatch between CNN and LLM embedding spaces |
| Cross-rank negatives (2048 effective) | Larger batch = harder negatives = stronger gradient signal for CLIP |
| `global_loss_weight=1.0` | Equal weighting; global token needs as much signal as grid |
| Cache key = full text string | Changing labels mid-run causes cache misses (new descriptions) but no mixed-signal; old entries evicted by FIFO |
| FIFO eviction (no LRU move_to_end) | At >90% hit rate, LRU order has no practical value; `move_to_end` was breaking Linux CoW causing steady RSS growth |
| `num_workers=0` | DataLoader workers inherit full process RSS (~29GB) via fork; IO is not the bottleneck (~125s/step dominated by Qwen embedding) |
| tau learnable, clamped [0.01, 0.5] | Follows OpenAI CLIP; tau=0.034 at step 5700 (converging toward lower bound — normal) |
| `--reset-scheduler` on LR change | Cosine schedule at step 4500/20000 had decayed LR to ~8.8e-5; reset to peak 3e-4 |
