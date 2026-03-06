"""SFT training — Qwen3.5-4B-Thinking-2507 + QLoRA (8-bit, r=64).

Parallelism: DDP (torchrun --nproc_per_node=2)
  Each GPU holds a full model copy; Trainer synchronises gradients.

Usage (via recipe):
    ./recipes-train/qwen3.5-4b-sft/start.sh

Direct:
    torchrun --nproc_per_node=2 recipes-train/qwen3.5-4b-sft/train.py \\
        --config recipes-train/qwen3.5-4b-sft/config.yaml
"""

import argparse
import logging
import os
import sys

import torch

# Allow importing training.lib from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.lib import load_config, logger, run_training  # noqa: E402

_logger = logging.getLogger(__name__)


def setup(config_path: str):
    """Load Qwen3.5-4B with 8-bit QLoRA for DDP training."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = load_config(config_path)
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})

    model_name = model_cfg["model_name"]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    _logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    _logger.info("Loading model (8-bit) on GPU %d", local_rank)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": local_rank},  # DDP: each rank owns its GPU
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        target_modules=lora_cfg.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="recipes-train/qwen3.5-4b-sft/config.yaml")
    parser.add_argument("--resume", nargs="?", const=True, default=None, metavar="CHECKPOINT_DIR")
    args = parser.parse_args()

    model, tokenizer = setup(args.config)
    run_training(args.config, model, tokenizer, resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
