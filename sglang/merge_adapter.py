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
    d["model_type"] = "chess_qwen3"
    # Flatten text_config fields to top-level so SGLang and our model code find them
    text_cfg = d.get("text_config", {})
    if isinstance(text_cfg, dict) and "hidden_size" not in d:
        for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "intermediate_size", "vocab_size",
                    "rms_norm_eps", "rope_theta", "max_position_embeddings"):
            if key in text_cfg and key not in d:
                d[key] = text_cfg[key]
    # Chess-specific
    d["board_token_id"] = 248055        # BOARD_TOKEN_ID = <|vision_pad|>
    d["board_tokens_per_position"] = 65
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
    # ---- 1. Load all weights ----
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)

    print(f"  Total keys: {len(state)}")

    # ---- 2. Split CNN vs LLM ----
    cnn_state = {k: v for k, v in state.items() if k.startswith("cnn.")}
    print(f"  CNN keys: {len(cnn_state)}")

    # ---- 3. Save CNN weights ----
    cnn_out = output_dir / "encoder_weights.pt"
    torch.save(cnn_state, str(cnn_out))
    print(f"Saved CNN weights → {cnn_out}")

    # ---- 4. Translate and save LLM weights ----
    llm_state: dict[str, torch.Tensor] = {}
    skipped = []
    for k, v in state.items():
        new_key = translate_key(k)
        if new_key is None:
            skipped.append(k)
            continue
        if new_key in llm_state:
            print(f"  WARNING: duplicate key after translation: {k} → {new_key}")
            continue
        llm_state[new_key] = v

    print(f"  LLM keys translated: {len(llm_state)}  skipped: {len(skipped)}")

    llm_out = output_dir / "model.safetensors"
    save_file(llm_state, str(llm_out))
    print(f"Saved LLM weights → {llm_out}")

    # ---- 5. Write config.json ----
    config = build_config_json(args.base_model, checkpoint_dir)
    config_out = output_dir / "config.json"
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config → {config_out}")

    # ---- 6. Copy tokenizer files ----
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    for fname in tokenizer_files:
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"Copied tokenizer file: {fname}")

    print(f"\nDone. Merged model at: {output_dir}")
    print("Next: SGLANG_EXTERNAL_MODEL_PACKAGE=chess_sglang python -m sglang.launch_server --model-path", output_dir)


if __name__ == "__main__":
    main()
