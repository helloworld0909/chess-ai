"""GRPO training — Phase 2 SFT checkpoint + QLoRA (8-bit, r=32).

Stage 1 line generator: teaches the model to produce legally correct engine
lines with accurate eval labels and structural move annotations.

Starts from Phase 2 SFT checkpoint (checkpoint-350) which already knows the
output format.  GRPO grounds the annotations with verifiable Stockfish rewards.

Parallelism: single GPU (GRPO rollout generation is the bottleneck, not
weight updates — DDP adds synchronisation overhead without throughput gain).

Reward functions (Phase 1, all free/cheap):
    R1  — Move legality          hard gate (-1.0 if illegal, skip rest)
    R2  — Eval label accuracy    weight 0.28  (Stockfish depth 18)
    R3a — Structural annotations weight 0.12  (python-chess, free)
    R4  — Line depth             weight 0.10  (encourages ≥6 half-moves)
    R5  — Line breadth           weight 0.10  (unique first moves)
    R6  — Line relevance         weight 0.05  (first move is legal from FEN)

Usage (via recipe):
    ./recipes-train/qwen3.5-4b-phase3-grpo/start.sh

Direct:
    STOCKFISH_PATH=/path/to/stockfish \\
    python recipes-train/qwen3.5-4b-phase3-grpo/train.py \\
        --config recipes-train/qwen3.5-4b-phase3-grpo/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_grpo_dataset(jsonl_path: str):
    """Load the line-generator SFT JSONL as a GRPO prompt dataset.

    GRPOTrainer expects a dataset with a 'prompt' column — a list of messages
    up to (but not including) the final assistant turn.  We strip the last
    assistant message (which contains the <line> targets) and keep the
    system + user turns as the prompt.
    """
    from datasets import Dataset

    rows: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages", [])
            if not msgs:
                continue
            # Strip trailing assistant turn — that's what GRPO generates
            if msgs[-1]["role"] == "assistant":
                prompt_msgs = msgs[:-1]
            else:
                prompt_msgs = msgs
            rows.append(
                {
                    "prompt": prompt_msgs,
                    "fen": rec.get("metadata", {}).get("fen", ""),
                    "move_san": rec.get("metadata", {}).get("move_san", ""),
                }
            )
    log.info("Loaded %d GRPO prompt rows from %s", len(rows), jsonl_path)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def setup_model_and_tokenizer(config: dict):
    import os

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    model_name = model_cfg["model_name"]

    # Detect whether model_name points to a PEFT adapter checkpoint
    # (has adapter_config.json) or a full / base model directory.
    adapter_config_path = os.path.join(model_name, "adapter_config.json")
    is_peft_checkpoint = os.path.isfile(adapter_config_path)

    if is_peft_checkpoint:
        import json

        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]
        log.info(
            "Detected PEFT checkpoint — warm-start from Phase 2 adapter. Base: %s  Adapter: %s",
            base_model_name,
            model_name,
        )
    else:
        base_model_name = model_name

    log.info("Loading tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,  # tokenizer is saved in the adapter dir too
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPOTrainer requires left-padding for generation

    n_gpus = torch.cuda.device_count()
    # DDP: each rank gets its own GPU replica.  local_rank is set by torchrun
    # via LOCAL_RANK env var. Falls back to 0 for single-process launch.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    log.info("Loading base model in 8-bit on GPU %d (of %d)", local_rank, n_gpus)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": local_rank},
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    # Always return the base model + LoraConfig so GRPOTrainer wraps it correctly.
    # If warm-starting from a PEFT checkpoint, we store the adapter path for
    # load_grpo_sft_weights() to call after GRPOTrainer has set up the adapter.
    base_model._phase2_adapter_path = model_name if is_peft_checkpoint else None

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )

    return base_model, tokenizer, lora_config


# ---------------------------------------------------------------------------
# Reward wrappers
# ---------------------------------------------------------------------------


def _make_weighted_reward(fn, weight: float):
    """Wrap a reward function to apply a scalar weight."""

    def _wrapped(prompts, completions, **kwargs):
        raw = fn(prompts, completions, **kwargs)
        return [weight * v for v in raw]

    _wrapped.__name__ = f"weighted_{fn.__name__}"
    return _wrapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="recipes-train/qwen3.5-4b-grpo/config.yaml")
    parser.add_argument("--resume", nargs="?", const=True, default=None, metavar="CHECKPOINT_DIR")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    wandb_cfg = config.get("wandb", {})

    # ── Weights & Biases ──────────────────────────────────────────────────
    if wandb_cfg.get("enabled"):
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "chess-tutor-grpo"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags", []),
        )

    # ── Model ─────────────────────────────────────────────────────────────
    model, tokenizer, lora_config = setup_model_and_tokenizer(config)

    # ── Dataset ───────────────────────────────────────────────────────────
    train_dataset = load_grpo_dataset(train_cfg["train_file"])
    eval_dataset = None
    if train_cfg.get("eval_file"):
        eval_dataset = load_grpo_dataset(train_cfg["eval_file"])

    # ── Reward functions ──────────────────────────────────────────────────
    # R1 (legality) is a hard gate: illegal completion → -1.0, all others 0.
    # Implemented by combined_reward; here we pass individual weighted fns to
    # GRPOTrainer so each reward is logged separately in wandb / stdout.
    from verification import rewards as _rewards_mod
    from verification.rewards import (
        reward_annotation_structural,
        reward_breadth,
        reward_depth,
        reward_eval_accuracy,
        reward_format,
        reward_legality,
        reward_relevance,
    )

    # Configure Stockfish eval depth from config (default 12 — fast enough for reward signal)
    sf_depth = train_cfg.get("stockfish_depth", 12)
    _rewards_mod._SF_DEPTH = sf_depth
    log.info("Stockfish reward depth: %d", sf_depth)

    def reward_legality_gate(prompts: list, completions: list, **kwargs) -> list[float]:
        """R1 smooth legality: passes through the raw score from reward_legality.

        reward_legality now returns a continuous value in [-1.0, +1.0] based on
        the fraction of legal moves across all lines. This gives the model a
        gradient to push more moves through legality rather than a hard cliff.
        +1.0 = all lines fully legal; -1.0 = zero legal moves anywhere.
        """
        return reward_legality(prompts, completions, **kwargs)

    # ── Output logger: writes all completions + per-reward breakdown each batch ──
    # Log file rotates every logging_steps batches to avoid unbounded growth.
    _reward_names = [
        "R0_format",
        "R1_legality",
        "R2_eval",
        "R3a_annot",
        "R4_depth",
        "R5_breadth",
        "R6_relevance",
    ]

    class _RewardLogger:
        """Logs every completion with per-reward breakdown each batch."""

        def __init__(self, num_funcs: int, log_path: str = "completions.log"):
            self.num_funcs = num_funcs
            self.log_path = log_path
            self.step = 0

        def _write_parallel(
            self, step: int, completions: list, all_scores: list, reward_secs: float = 0.0
        ) -> None:
            """Log reward summary table + best completion text each batch."""
            import datetime

            n = len(completions)
            totals = [sum(all_scores[r][i] for r in range(self.num_funcs)) for i in range(n)]
            best_i = max(range(n), key=lambda i: totals[i])
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(self.log_path, "a") as f:
                f.write(f"\n{'─' * 80}\n")
                f.write(
                    f"STEP {step}  {ts}  ({n} completions)  "
                    f"reward_time={reward_secs:.2f}s  "
                    f"mean={sum(totals) / n:+.3f}  "
                    f"best={totals[best_i]:+.3f}  "
                    f"worst={min(totals):+.3f}\n"
                )
                # Full text of best completion first
                best_text = (
                    completions[best_i][-1]["content"]
                    if isinstance(completions[best_i], list)
                    else str(completions[best_i])
                )
                f.write(f"\n── Best completion [{best_i}] ──\n")
                f.write(best_text.strip() + "\n")
                # Compact reward table: one row per completion
                header = f"\n  {'#':>2}  {'total':>7}  " + "  ".join(
                    f"{name:>10}" for name in _reward_names
                )
                f.write(header + "\n")
                for i, total in enumerate(totals):
                    marker = " *" if i == best_i else "  "
                    row = f"{marker}{i:>2}  {total:>+7.3f}  " + "  ".join(
                        f"{all_scores[r][i]:>+10.3f}" for r in range(self.num_funcs)
                    )
                    f.write(row + "\n")

    _logger = _RewardLogger(num_funcs=7)

    # Individual weighted reward functions (for logging breakdown)
    _reward_fns_individual = [
        _make_weighted_reward(reward_format, 0.20),  # R0
        reward_legality_gate,  # R1
        _make_weighted_reward(reward_eval_accuracy, 0.28),  # R2
        _make_weighted_reward(reward_annotation_structural, 0.12),  # R3a
        _make_weighted_reward(reward_depth, 0.10),  # R4
        _make_weighted_reward(reward_breadth, 0.10),  # R5
        _make_weighted_reward(reward_relevance, 0.05),  # R6
    ]

    # Run all 7 reward functions in parallel — TRL calls reward functions sequentially,
    # so we expose a single function that fans out to a thread pool internally.
    # This overlaps Stockfish evals (R2), annotation checks (R3a), and other CPU work,
    # cutting reward wall-time from ~sum(fn_times) to ~max(fn_times).
    from concurrent.futures import ThreadPoolExecutor as _TPE

    _reward_executor = _TPE(max_workers=7, thread_name_prefix="reward")

    def combined_parallel_reward(prompts, completions, **kwargs):
        import time

        t0 = time.perf_counter()
        futures = [
            _reward_executor.submit(fn, prompts, completions, **kwargs)
            for fn in _reward_fns_individual
        ]
        all_scores = [f.result() for f in futures]  # [[s0..sN], [s0..sN], ...]
        reward_secs = time.perf_counter() - t0

        # Log breakdown and return sum across reward functions per completion
        _logger._write_parallel(_logger.step + 1, completions, all_scores, reward_secs)
        _logger.step += 1

        n = len(completions)
        return [sum(all_scores[r][i] for r in range(7)) for i in range(n)]

    combined_parallel_reward.__name__ = "combined_reward"

    reward_fns = [combined_parallel_reward]

    # ── GRPOConfig ────────────────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=config.get("output_dir", "checkpoints/chess-tutor-4b-grpo"),
        # Rollout generation
        num_generations=train_cfg.get("num_generations", 8),
        max_prompt_length=train_cfg.get("max_prompt_length", 1024),
        max_completion_length=train_cfg.get("max_completion_length", 512),
        # Training
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 10),
        optim=train_cfg.get("optim", "adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.1),
        # Logging / checkpointing
        logging_steps=train_cfg.get("logging_steps", 5),
        logging_first_step=True,
        eval_strategy=train_cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 100),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 5),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        seed=train_cfg.get("seed", 42),
        # GRPO-specific
        beta=train_cfg.get("beta", 0.04),  # KL penalty coefficient
        epsilon=train_cfg.get("epsilon", 0.2),  # PPO clip ratio
        report_to="wandb" if wandb_cfg.get("enabled") else "none",
        # Generation params — allow thinking
        temperature=train_cfg.get("temperature", 0.9),
        top_p=train_cfg.get("top_p", 0.95),
    )

    # ── Compatibility shim: TRL 0.24 expects model.warnings_issued ───────
    # transformers 5.2 doesn't have this attribute; add it so GRPOTrainer.__init__
    # doesn't crash when it tries to suppress a spurious warning.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ── Warm-start: load Phase 2 SFT adapter weights ──────────────────────
    # GRPOTrainer created a fresh LoRA adapter above. Now load Phase 2 weights
    # into it so GRPO fine-tunes from the SFT checkpoint, not random init.
    phase2_path = getattr(model, "_phase2_adapter_path", None)
    if phase2_path:
        from safetensors.torch import load_file

        adapter_weights = load_file(f"{phase2_path}/adapter_model.safetensors")
        peft_model = trainer.model
        state = peft_model.state_dict()
        # PEFT saves without adapter name: "...lora_A.weight"
        # GRPOTrainer state dict uses adapter name "default": "...lora_A.default.weight"
        # Remap by inserting ".default" before the final ".weight"
        matched = 0
        for k, v in adapter_weights.items():
            remapped = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight"
            )
            target_key = remapped if remapped in state else k
            if target_key in state:
                state[target_key] = v.to(state[target_key].device, dtype=state[target_key].dtype)
                matched += 1
        peft_model.load_state_dict(state, strict=False)
        log.info("Warm-started GRPO adapter from Phase 2 checkpoint (%d tensors loaded)", matched)

    log.info("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=args.resume)

    out = config.get("output_dir", "checkpoints/chess-tutor-4b-grpo")
    log.info("Saving model to %s", out)
    trainer.save_model()
    tokenizer.save_pretrained(out)
    log.info("Done.")


if __name__ == "__main__":
    main()
