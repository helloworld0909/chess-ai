"""Merge PEFT LoRA adapter + CNN weights from a chess-ai checkpoint into a
standard HuggingFace safetensors directory that SGLang can load directly.

The checkpoint produced by ChessLMWithEncoder has this key layout:
  cnn.*                              — CNN trunk + projector weights
  llm.base_model.model.model.*       — Qwen3.5-4B base weights (LoRA patched)
  llm.base_model.model.lm_head.*

After merging we produce two files in --output:
  config.json          — ChessLMConfig (Qwen3Config subclass, board_token_id=248055)
  model.safetensors    — merged Qwen3.5-4B weights (LoRA applied, SGLang naming)
  encoder_weights.pt   — CNN state dict (cnn.*), loaded by ChessBoardProcessor
  tokenizer*           — copied from the checkpoint (or base model)

Usage:
    python sglang/merge_adapter.py \\
        --checkpoint checkpoints/qwen3.5-4b-encoder-phase1-alignment-medium/checkpoint-9070 \\
        --base-model Qwen/Qwen3.5-4B \\
        --output /tmp/chess-merged
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CHESS_AI_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(CHESS_AI_SRC))


# ---------------------------------------------------------------------------
# Config JSON for SGLang
# ---------------------------------------------------------------------------

def build_config_json(base_model: str, checkpoint_dir: Path) -> dict:
    """Load Qwen3.5-4B config and add chess-specific fields."""
    # Try loading from checkpoint dir first (avoids HF download + model_type compatibility)
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            d = json.load(f)
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        d = cfg.to_dict()
    # Override architecture so SGLang picks up ChessQwen3ForCausalLM
    d["architectures"] = ["ChessQwen3ForCausalLM"]
    # Keep qwen3_5 model_type: SGLang registers Qwen3_5Config with AutoConfig at import
    # time, so it's available when get_config() is called. Using qwen3_5 also gives us
    # the Qwen3_5Config.layers_block_type property that Qwen3_5ForCausalLM requires.
    d["model_type"] = "qwen3_5"
    # Keep text_config intact: SGLang's Qwen3_5Config.__init__ rebuilds it into
    # Qwen3_5TextConfig (a Qwen3NextConfig subclass with layers_block_type property).
    # SGLang's get_hf_text_config() returns config.text_config for qwen3_5 model type.
    # The Qwen3_5TextConfig IS the config passed to Qwen3_5ForCausalLM — it has
    # layers_block_type, full_attention_interval, and other SSM/hybrid fields.
    # Do NOT remove text_config or flatten it.
    # Chess-specific
    d["board_token_id"] = 248055        # BOARD_TOKEN_ID = <|vision_pad|>
    d["board_tokens_per_position"] = 1
    d["encoder_in_channels"] = 19
    d["encoder_hidden_size"] = 256      # actual running config (h=256 blocks=10)
    d["encoder_num_blocks"] = 10
    return d


# ---------------------------------------------------------------------------
# Key translation: checkpoint → SGLang Qwen3ForCausalLM naming
# ---------------------------------------------------------------------------

def translate_key(name: str) -> str | None:
    """Translate a merged checkpoint key to SGLang's Qwen3_5ForCausalLM namespace.

    SGLang's Qwen3_5ForCausalLM.load_weights() expects:
        model.embed_tokens.weight
        model.layers.N.{input_layernorm,post_attention_layernorm}.weight
        model.layers.N.{q_proj,k_proj,v_proj,o_proj}.weight
        model.layers.N.{gate_proj,up_proj,down_proj}.weight
        model.norm.weight
        lm_head.weight

    Our checkpoint (after PEFT merge) has:
        llm.base_model.model.model.*   →  model.*
        llm.base_model.model.lm_head.* →  lm_head.*

    We skip:
        cnn.*          — saved separately as encoder_weights.pt
        rotary_emb.*   — not needed
    """
    if name.startswith("cnn."):
        return None  # handled separately
    if "rotary_emb.inv_freq" in name:
        return None

    # Strip PEFT wrapper prefix
    if name.startswith("llm.base_model.model."):
        name = name[len("llm.base_model.model."):]
    elif name.startswith("llm."):
        name = name[len("llm."):]

    # SGLang's Qwen3_5ForCausalLM also strips .self_attn in load_weights
    # so we keep the full name and let load_weights handle it.
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Merge chess-ai PEFT checkpoint for SGLang serving")
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to chess-ai checkpoint dir (contains model.safetensors)",
    )
    parser.add_argument(
        "--base-model", "-b",
        default="Qwen/Qwen3.5-4B",
        help="HuggingFace base model name for config (default: Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for the merged model",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    safetensors_path = checkpoint_dir / "model.safetensors"
    if not safetensors_path.exists():
        print(f"ERROR: {safetensors_path} not found")
        sys.exit(1)

    print(f"Loading checkpoint: {safetensors_path}")
    # ---- 1. Load checkpoint weights ----
    ckpt: dict[str, torch.Tensor] = {}
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            ckpt[key] = f.get_tensor(key)

    print(f"  Checkpoint keys: {len(ckpt)}")

    # ---- 2. Extract CNN weights ----
    cnn_state = {k: v for k, v in ckpt.items() if k.startswith("cnn.")}
    print(f"  CNN keys: {len(cnn_state)}")

    # ---- 3. Save CNN weights ----
    cnn_out = output_dir / "encoder_weights.pt"
    torch.save(cnn_state, str(cnn_out))
    print(f"Saved CNN weights → {cnn_out}")

    # ---- 4. Load base model weights ----
    # The checkpoint only has attention LoRA + CNN; base model has all SSM layers too.
    # We must start from the base model and apply the LoRA delta.
    #
    # Base model key format: model.language_model.layers.N.*
    # SGLang ChessQwen3ForCausalLM.model (Qwen3_5ForCausalLM) is instantiated with
    # prefix "model", so SGLang's load_weights sees: model.embed_tokens.*, model.layers.*
    # (Qwen3VLForConditionalGeneration.load_weights strips "language_model" → "model.*")
    # For our standalone backbone: translate base keys to "model.*"
    print(f"Loading base model from {args.base_model}...")
    from huggingface_hub import snapshot_download
    base_dir = Path(snapshot_download(
        args.base_model,
        local_files_only=True,
        ignore_patterns=["*.pt"],
    ))
    import json as _json
    index_path = base_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = _json.load(f)
        shard_files = list(set(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    base_weights: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        with safe_open(str(base_dir / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if "rotary_emb.inv_freq" in k:
                    continue
                # Strip model.language_model. → model.
                translated = k.replace("model.language_model.", "model.")
                base_weights[translated] = f.get_tensor(k)
    print(f"  Base model keys: {len(base_weights)}")

    # ---- 5. Extract LoRA delta from checkpoint ----
    # Checkpoint structure (PEFT):
    #   llm.base_model.model.model.layers.N.{q_proj,...}.base_layer.weight
    #   llm.base_model.model.model.layers.N.{q_proj,...}.lora_A.default.weight
    #   llm.base_model.model.model.layers.N.{q_proj,...}.lora_B.default.weight
    # SGLang name after stripping prefix: model.layers.N.{q_proj,...}.weight
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}

    for k, v in ckpt.items():
        if k.startswith("cnn.") or "rotary_emb.inv_freq" in k:
            continue
        name = k
        if name.startswith("llm.base_model.model."):
            name = name[len("llm.base_model.model."):]
        elif name.startswith("llm."):
            name = name[len("llm."):]

        if ".lora_A.default.weight" in name:
            base_name = name.replace(".lora_A.default.weight", ".weight")
            lora_a[base_name] = v
        elif ".lora_B.default.weight" in name:
            base_name = name.replace(".lora_B.default.weight", ".weight")
            lora_b[base_name] = v

    print(f"  LoRA pairs: {len(lora_a)}")

    # ---- 6. Merge: base + LoRA delta ----
    merged: dict[str, torch.Tensor] = dict(base_weights)
    n_merged = 0
    for name in list(lora_a.keys()):
        if name not in lora_b:
            continue
        if name not in merged:
            print(f"  WARNING: LoRA target {name!r} not in base model — skipping")
            continue
        w_base = merged[name]
        w_a = lora_a[name]  # (r, in)
        w_b = lora_b[name]  # (out, r)
        delta = w_b @ w_a   # (out, in)
        merged[name] = w_base + delta.to(w_base.dtype)
        n_merged += 1

    print(f"  Merged: {n_merged} LoRA pairs applied to base ({len(merged)} total keys)")

    llm_out = output_dir / "model.safetensors"
    save_file(merged, str(llm_out))
    print(f"Saved LLM weights → {llm_out}")

    # ---- 5. Write config.json ----
    config = build_config_json(args.base_model, checkpoint_dir)
    config_out = output_dir / "config.json"
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config → {config_out}")

    # ---- 6. Copy tokenizer files ----
    # Prefer base model tokenizer (has standard tokenizer_class like Qwen2Tokenizer).
    # The checkpoint's tokenizer_config.json may use TokenizersBackend which is
    # not recognized by older transformers versions in the sglang-serve venv.
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    # Try to get base model tokenizer from HF cache
    base_tokenizer_dir: Optional[Path] = None
    try:
        from huggingface_hub import snapshot_download
        base_tokenizer_dir = Path(snapshot_download(
            args.base_model,
            local_files_only=True,
            ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
        ))
        print(f"Using base model tokenizer from: {base_tokenizer_dir}")
    except Exception as e:
        print(f"Could not find base model tokenizer in cache ({e}), using checkpoint tokenizer")

    for fname in tokenizer_files:
        # Prefer base model tokenizer, fall back to checkpoint
        src = None
        if base_tokenizer_dir and (base_tokenizer_dir / fname).exists():
            src = base_tokenizer_dir / fname
        elif (checkpoint_dir / fname).exists():
            src = checkpoint_dir / fname
        if src:
            shutil.copy2(src, output_dir / fname)
            print(f"Copied tokenizer file: {fname} (from {src.parent.name})")

    print(f"\nDone. Merged model at: {output_dir}")
    print("Next: SGLANG_EXTERNAL_MODEL_PACKAGE=chess_sglang python -m sglang.launch_server --model-path", output_dir)


if __name__ == "__main__":
    main()
