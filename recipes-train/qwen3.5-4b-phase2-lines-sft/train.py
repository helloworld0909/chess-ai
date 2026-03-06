"""Line-generator SFT — Qwen3.5-4B-Thinking-2507 + QLoRA (8-bit, r=64).

Teaches the model the <line> output format before GRPO.
Identical model/LoRA setup to qwen3.5-4b-sft; differs only in data loader.

Parallelism: DDP (torchrun --nproc_per_node=2), same as coach SFT.

Usage (via recipe):
    ./recipes-train/qwen3.5-4b-lines-sft/start.sh

Direct:
    torchrun --nproc_per_node=2 recipes-train/qwen3.5-4b-lines-sft/train.py \\
        --config recipes-train/qwen3.5-4b-lines-sft/config.yaml
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.lib import (  # noqa: E402
    build_trainer,
    format_dataset,
    load_config,
    load_jsonl_lines,
    logger,
    make_training_args,
)

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
        device_map={"": local_rank},
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    parser.add_argument("--config", "-c", default="recipes-train/qwen3.5-4b-lines-sft/config.yaml")
    parser.add_argument("--resume", nargs="?", const=True, default=None, metavar="CHECKPOINT_DIR")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})

    if wandb_cfg.get("enabled"):
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "chess-tutor"),
            name=wandb_cfg.get("name", "lines-sft-4b"),
            tags=wandb_cfg.get("tags", ["lines-sft"]),
        )

    model, tokenizer = setup(args.config)

    keep_think = train_cfg.get("keep_think", False)
    _logger.info(
        "Loading training data from %s (keep_think=%s)", train_cfg["train_file"], keep_think
    )
    train_dataset = format_dataset(
        load_jsonl_lines(train_cfg["train_file"]), tokenizer, keep_think=keep_think
    )

    eval_dataset = None
    if train_cfg.get("eval_file"):
        _logger.info("Loading eval data from %s", train_cfg["eval_file"])
        eval_dataset = format_dataset(
            load_jsonl_lines(train_cfg["eval_file"]), tokenizer, keep_think=keep_think
        )

    training_args = make_training_args(config)

    trainer = build_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        max_seq_length=train_cfg.get("max_seq_length", 1280),
        packing=train_cfg.get("packing", False),
        training_args=training_args,
    )

    _logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    _logger.info("Saving model to %s", training_args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    _logger.info("Done.")


if __name__ == "__main__":
    main()
