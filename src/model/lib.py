"""Shared SFT training utilities."""

from __future__ import annotations

import json
import logging
import re

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl_lines(path: str) -> list[dict]:
    """Load JSONL lines for the SFT task (any source tag accepted)."""
    data = []
    total = 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                sample = json.loads(raw)
                total += 1
                if "messages" in sample:
                    data.append(
                        {
                            "messages": sample["messages"],
                            "metadata": sample.get("metadata", {}),
                        }
                    )
            except json.JSONDecodeError:
                continue
    logger.info("Loaded %d samples (out of %d total) from %s", len(data), total, path)
    return data


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_from_target(messages: list[dict]) -> list[dict]:
    """Remove <think>...</think> blocks from assistant targets.

    SFT trains on output format only; thinking quality is left for GRPO.
    """
    result = []
    for msg in messages:
        if msg["role"] == "assistant":
            content = _THINK_RE.sub("", msg.get("content") or "").strip()
            result.append({**msg, "content": content})
        else:
            result.append(msg)
    return result


def make_training_args(config: dict):
    """Build TrainingArguments from a loaded YAML config dict."""
    from transformers import TrainingArguments

    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})

    return TrainingArguments(
        output_dir=config.get("output_dir", "checkpoints/output"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        max_steps=train_cfg.get("max_steps", -1),
        optim=train_cfg.get("optim", "adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        logging_first_step=True,
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 200),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        accelerator_config={"dispatch_batches": train_cfg.get("dispatch_batches", None)},
        report_to="wandb" if wandb_cfg.get("enabled") else "none",
        use_liger_kernel=train_cfg.get("use_liger_kernel", True),
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
    )
