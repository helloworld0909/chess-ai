"""Shared SFT training utilities for Chess Tutor.

Provides data loading, dataset formatting, and trainer construction.
Model/LoRA setup is recipe-specific — see recipes-train/*/train.py.
"""

import json
import logging
import re
from typing import Optional

import torch
import yaml
from datasets import Dataset

# ── caching_allocator_warmup monkey-patch ────────────────────────────────────
# transformers 5.0+ runs caching_allocator_warmup() which pre-allocates up to
# half the GPU VRAM before the quantized model has been created. With QLoRA
# (4-bit) on 32 GiB GPUs this causes an OOM before any training weights are
# ever loaded. Replacing with a no-op is safe — the CUDA caching allocator
# will grow on-demand instead.
try:
    import transformers.modeling_utils as _mu

    def _noop_warmup(*args, **kwargs):  # noqa: D401
        pass

    _mu.caching_allocator_warmup = _noop_warmup
    _patch_logger = logging.getLogger(__name__).getChild("patch")
    _patch_logger.info("caching_allocator_warmup patched to no-op (OOM prevention)")

    import transformers.core_model_loading as _cml

    _cml.GLOBAL_WORKERS = 1
    _patch_logger.info("GLOBAL_WORKERS patched to 1 (serialized loading for OOM prevention)")
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> Dataset:
    """Load JSONL file, keeping only samples with source='textbook'."""
    data = []
    total_found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                total_found += 1
                metadata = sample.get("metadata", {})
                if metadata.get("source") == "textbook":
                    if "messages" in sample:
                        data.append({"messages": sample["messages"]})
            except json.JSONDecodeError:
                continue
    logger.info(
        "Loaded %d textbook samples (out of %d total) from %s", len(data), total_found, path
    )
    return data


def load_jsonl_lines(path: str) -> Dataset:
    """Load JSONL for the line-generator task.

    Unlike load_jsonl, accepts any source tag — the lines SFT files use
    'lichess_lines_sft' rather than 'textbook'.
    """
    data = []
    total_found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                total_found += 1
                if "messages" in sample:
                    data.append(
                        {
                            "messages": sample["messages"],
                            "metadata": sample.get("metadata", {}),
                        }
                    )
            except json.JSONDecodeError:
                continue
    logger.info(
        "Loaded %d line-generator samples (out of %d total) from %s",
        len(data),
        total_found,
        path,
    )
    return data


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_from_target(messages: list[dict]) -> list[dict]:
    """Remove <think>...</think> blocks from assistant message targets.

    Why: pipeline-generated thinking blocks contain meta-reasoning (notation
    checks, legality loops) that SFT would faithfully imitate. We want SFT to
    teach *what a good coaching comment looks like*, not *how the pipeline
    model happened to think*. Thinking quality is left for GRPO to teach via
    verifiable rewards.
    """
    result = []
    for msg in messages:
        if msg["role"] == "assistant":
            content = _THINK_RE.sub("", msg.get("content") or "").strip()
            result.append({**msg, "content": content})
        else:
            result.append(msg)
    return result


def format_dataset(dataset: Dataset, tokenizer, keep_think: bool = False) -> Dataset:
    """Apply the model's native chat template to every sample.

    By default strips <think> blocks from assistant targets so SFT only trains
    on the output format. Pass keep_think=True to include the thinking scaffold
    in the loss (used for Phase 2 thinking-distillation SFT).
    Samples that fail chat template application are filtered out.
    """

    def _fmt(example: dict) -> dict:
        try:
            messages = (
                example["messages"] if keep_think else strip_think_from_target(example["messages"])
            )
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text, "_ok": True}
        except Exception as exc:
            logger.debug("Skipping sample, apply_chat_template failed: %s", exc)
            return {"text": "", "_ok": False}

    dataset = dataset.map(_fmt, remove_columns=["messages"], desc="Applying chat template")
    before = len(dataset)
    dataset = dataset.filter(lambda x: x["_ok"])
    dataset = dataset.remove_columns(["_ok"])
    skipped = before - len(dataset)
    if skipped:
        logger.warning("Filtered %d / %d samples that failed chat template", skipped, before)
    return dataset


def build_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    max_seq_length: int,
    packing: bool,
    training_args,
):
    """Construct SFTTrainer with response-only loss masking.

    TRL 0.10+ bakes completion_only_loss into SFTConfig — no separate
    DataCollatorForCompletionOnlyLM needed.
    """
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    dict_args = training_args.to_dict()
    dict_args["hub_token"] = training_args.hub_token
    dict_args.pop("push_to_hub_token", None)
    sft_config = SFTConfig(
        **dict_args,
        max_length=max_seq_length,
        packing=packing,
        dataset_text_field="text",
        completion_only_loss=True,
    )

    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )


def make_training_args(config: dict):
    """Build TrainingArguments from a loaded YAML config dict."""
    from transformers import TrainingArguments

    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})
    deepspeed_cfg = config.get("deepspeed", {})
    deepspeed_path = deepspeed_cfg.get("config_file") if deepspeed_cfg.get("enabled") else None

    return TrainingArguments(
        output_dir=config.get("output_dir", "checkpoints/chess-tutor-v1"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
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
        report_to="wandb" if wandb_cfg.get("enabled") else "none",
        deepspeed=deepspeed_path,
        use_liger_kernel=train_cfg.get("use_liger_kernel", True),
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
    )


def run_training(config_path: str, model, tokenizer, resume_from_checkpoint=None):
    """Load data, build trainer, run training, save model.

    Called by each recipe's train.py after model/tokenizer setup.
    """
    config = load_config(config_path)
    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})

    if wandb_cfg.get("enabled"):
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "chess-tutor"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags", []),
        )

    max_seq_length = train_cfg.get("max_seq_length", 8192)
    packing = train_cfg.get("packing", False)

    logger.info("Loading training data from %s", train_cfg["train_file"])
    train_dataset = format_dataset(
        Dataset.from_list(load_jsonl(train_cfg["train_file"])), tokenizer
    )

    eval_dataset = None
    if train_cfg.get("eval_file"):
        logger.info("Loading eval data from %s", train_cfg["eval_file"])
        eval_dataset = format_dataset(
            Dataset.from_list(load_jsonl(train_cfg["eval_file"])), tokenizer
        )

    training_args = make_training_args(config)

    trainer = build_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        max_seq_length,
        packing,
        training_args,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logger.info("Saving model to %s", training_args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Done.")
