# Chess Tutor SFT Training Summary

## 1. Objective

Fine-tune Qwen-series models as a chess coach. The model uses extended thinking (CoT) to
provide high-quality move analysis. SFT teaches coaching style; reasoning quality is deferred
to RL (GRPO) in a later phase.

---

## 2. Codebase Structure

```
training/
  lib.py                      # Shared: data loading, formatting, trainer, TrainingArguments

recipes-train/
  ds_zero2.json               # Shared DeepSpeed ZeRO-2 config
  qwen3.5-4b-sft/
    train.py                  # 4B-specific: 8-bit QLoRA, DDP (LOCAL_RANK), r=64
    config.yaml               # Hyperparameters
    start.sh / stop.sh        # Launch/stop via nohup + PID file
  qwen3.5-35b-sft/
    train.py                  # 35B-specific: 4-bit NF4, pipeline parallel (manual layer split), r=8
    config.yaml
    start.sh / stop.sh

recipes-inference/
  qwen3.5-4b-thinking-lora/     # vLLM + LoRA adapter, port 8101
  qwen3.5-35b-a3b-fp8/        # vLLM base model, port 8100
```

**Key design principle**: `training/lib.py` holds all generic machinery. Each recipe's
`train.py` owns model loading, quantization, device placement, and LoRA setup — nothing
model-specific lives in the shared lib.

---

## 3. Data

- **Source filter**: Only `source=textbook` samples used (highest quality coaching data)
- **Train**: 23,439 samples | **Eval**: 2,640 samples
- **Avg token length**: ~1,730 tokens (grown from ~1,200 after dataset rebuild — adds ~44%
  compute per step vs earlier runs)
- **Max token length**: 2,238 → `max_seq_length=2304` (covers 100% of dataset with headroom)
- **Completion-only loss**: `<think>` blocks stripped from assistant targets before training.
  Model learns coaching style only; GRPO teaches reasoning quality later.
- **Packing**: disabled — with 8-bit quantization and DDP it was 2× slower (45s/step vs 22s)
  due to irregular sequence lengths degrading memory access patterns

---

## 4. Parallelism Strategy

### 4B Model — Distributed Data Parallel (DDP)
- `torchrun --nproc_per_node=2` → each GPU gets a full model copy
- Both GPUs run forward+backward simultaneously on different mini-batches
- Gradients all-reduced across ranks each step
- **Quantization**: 8-bit (Int8) — ~4.5GB weights per GPU
- **Memory**: ~27-28GB per GPU at batch=5
- **Both GPUs at ~98-100% utilization**

### 35B Model — Pipeline Parallel (single process)
- `python3` (no torchrun) → `device_map` splits layers manually across 2 GPUs
- Even layer split: GPU0 = layers 0-N/2, GPU1 = layers N/2-N
- **Quantization**: 4-bit NF4 QLoRA — ~17.5GB weights across both GPUs
- **Trade-off**: GPUs take turns (one idle while other computes) → slower than DDP
- **LoRA rank**: r=8 (vs r=64 for 4B) to keep per-expert adapter memory manageable

---

## 5. Key Config (4B Active Run)

```yaml
model:      Qwen/Qwen3.5-4B-Instruct, 8-bit, sdpa
lora:       r=64, alpha=128, all 7 projections (q/k/v/o + gate/up/down)
batch:      5 per GPU × 2 GPUs × 8 grad_accum = 80 effective
lr:         1e-4 cosine, warmup_ratio=0.03
epochs:     3
max_seq:    2304
packing:    false
optimizer:  adamw_8bit
checkpoints: every 50 steps, keep 10
```

---

## 6. Lessons Learned

| Issue | Root Cause | Fix |
|---|---|---|
| `cudaErrorLaunchTimeout` | GPU 1 driving display → X11 watchdog kills kernels >2s | `sudo nvidia-smi --persistence-mode=1`; permanent fix: move display to AMD iGPU via BIOS "Primary Display = IGD" |
| OOM at batch=6 | ~30GB per GPU with full activations | Reduced to batch=5 |
| Packing 2× slower | 8-bit + irregular packed lengths → poor memory access | Disabled packing |
| 7hr vs 3hr run | Dataset avg length grew 1200→1730 tokens (+44% compute) | Irreducible; accepted |
| `DataCollatorForCompletionOnlyLM` removed | TRL 0.24 API change | Use `SFTConfig(completion_only_loss=True)` |
| Gradient checkpointing not working | Was called after `get_peft_model()` → didn't propagate to quantized layers → 27GB activation memory | Reverted to default (Trainer handles it on PeftModel wrapper) |
| Liger kernel crash | `device_map='auto'` causes tensors to cross GPU boundaries → Triton can't handle CPU tensors | Switch to DDP; Liger works fine when all tensors on same GPU per rank |

---

## 7. Current Training Status

- **Model**: `Qwen/Qwen3.5-4B-Instruct` + 8-bit QLoRA
- **Epoch**: ~1.3 / 3
- **Train loss**: ~0.31 | **Eval loss**: ~0.317 (tight gap → good generalization)
- **Speed**: ~21s/step → ~7 hours total
- **Checkpoints**: `checkpoints/chess-tutor-4b-poc/` every 50 steps

---

## 8. Next Steps

1. **Monitor to completion** — eval loss should continue falling through epoch 3
2. **Serve via recipe**: `./recipes-inference/qwen3.5-4b-thinking-lora/start.sh` (update checkpoint path in docker-compose)
3. **Manual review**: `uv run chess-review <username>` to evaluate coaching quality
4. **Go/No-Go for 35B SFT** based on 4B output quality
5. **Persistence mode on reboot**: `sudo ./scripts/install-nvidia-persistence.sh`
6. **AMD iGPU as primary display**: BIOS → "Primary Display = IGD" → plug HDMI into motherboard → both NVIDIA GPUs fully free for training
