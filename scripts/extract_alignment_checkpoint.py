"""Extract and merge weights from alignment checkpoint for GRPO initialization.

Reads checkpoint-20000/model.safetensors (ChessLMWithEncoder + PEFT LoRA), and produces:
  - /tmp/chess-grpo-init/model.safetensors   : LLM weights (LoRA merged, re-keyed to model.*)
  - /tmp/chess-grpo-init/encoder_weights.pt  : CNN encoder weights (cnn.* prefix preserved)

Usage:
    uv run python scripts/extract_alignment_checkpoint.py [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def merge_lora_weights(
    state: dict[str, torch.Tensor], lora_scale: float
) -> dict[str, torch.Tensor]:
    """Merge LoRA A/B into base_layer weights, return merged state with clean keys.

    Input keys (PEFT format):  llm.base_model.model.<rest>.base_layer.weight
                                llm.base_model.model.<rest>.lora_A.default.weight
                                llm.base_model.model.<rest>.lora_B.default.weight
    Output keys (plain HF):    model.<rest>.weight
    """
    # Collect base weights and LoRA components
    base: dict[str, torch.Tensor] = {}
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}

    prefix = "llm.base_model.model."

    for key, val in state.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]  # e.g. "model.layers.0.self_attn.q_proj.base_layer.weight"

        if rest.endswith(".base_layer.weight"):
            clean = rest[: -len(".base_layer.weight")] + ".weight"
            base[clean] = val
        elif rest.endswith(".lora_A.default.weight"):
            clean = rest[: -len(".lora_A.default.weight")] + ".weight"
            lora_a[clean] = val
        elif rest.endswith(".lora_B.default.weight"):
            clean = rest[: -len(".lora_B.default.weight")] + ".weight"
            lora_b[clean] = val
        else:
            # Non-LoRA parameter (layernorm, embed, etc.)
            base[rest] = val

    # Merge LoRA into base: W = W0 + lora_B @ lora_A * scale
    merged = dict(base)
    lora_keys = set(lora_a.keys()) & set(lora_b.keys())
    print(f"  Merging {len(lora_keys)} LoRA weight pairs (scale={lora_scale:.3f})")
    for key in lora_keys:
        if key not in merged:
            print(f"  WARNING: LoRA key {key} has no base weight, skipping")
            continue
        A = lora_a[key].float()
        B = lora_b[key].float()
        delta = (B @ A) * lora_scale
        merged[key] = (merged[key].float() + delta).to(base[key].dtype)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/qwen3.5-4b-base-encoder-phase1-alignment/checkpoint-20000",
    )
    parser.add_argument("--output-dir", default="/tmp/chess-grpo-init")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) / "model.safetensors"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_scale = args.lora_alpha / args.lora_r
    print(f"Loading {ckpt_path} (LoRA scale={lora_scale:.3f})")

    state: dict[str, torch.Tensor] = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)

    cnn_keys = {k: v for k, v in state.items() if k.startswith("cnn.")}
    llm_state = {k: v for k, v in state.items() if k.startswith("llm.")}

    print(f"  CNN keys: {len(cnn_keys)}, LLM keys: {len(llm_state)}")

    # --- CNN encoder weights ---
    # Keep cnn.* prefix — the GRPO loader strips it when loading into model.cnn
    cnn_out = out_dir / "encoder_weights.pt"
    torch.save(cnn_keys, cnn_out)
    print(f"Saved CNN weights → {cnn_out} ({len(cnn_keys)} tensors)")

    # --- LLM weights (merge LoRA, re-key) ---
    merged = merge_lora_weights(llm_state, lora_scale)
    llm_out = out_dir / "model.safetensors"

    # safetensors requires contiguous tensors and no shared storage
    merged_contiguous = {k: v.contiguous() for k, v in merged.items()}
    save_file(merged_contiguous, llm_out)
    print(f"Saved LLM weights → {llm_out} ({len(merged)} tensors)")
    print("Done.")


if __name__ == "__main__":
    main()
