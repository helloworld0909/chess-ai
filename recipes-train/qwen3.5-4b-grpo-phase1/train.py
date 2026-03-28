"""GRPO Phase 1 — Board-reading pretraining on verifiable chess tasks.

Tasks: legal_moves, captures, in_check, piece_at.
Reward: rule-based exact match (no LLM judge needed).
Both GPUs used for training (no judge inference server required).

Each prompt is multimodal: board image + FEN text + task question.
The board is rendered from FEN at dataset-load time (flipped for black-to-move).

The model must emit answers in a structured format:
  <think>...</think>
  <answer>...</answer>

Reward components:
  - R_format  (0.10): <answer> tag present
  - R_exact   (0.90): answer content matches ground truth exactly
    - legal_moves / captures: Jaccard similarity over SAN sets
    - in_check: "yes"/"no" exact match
    - piece_at: piece description exact match

Usage:
    ./recipes-train/qwen3.5-4b-grpo-phase1/start.sh
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import logging
import os
import re
import sys

import yaml
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Improved system prompt — injected at dataset load time (overrides JSONL)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a chess board reader. Answer directly from the position shown.

<think>
[brief reasoning]
</think>
<answer>
[your answer]
</answer>

Rules:
- legal_moves / captures: SAN notation, one per line, alphabetically sorted
- in_check: exactly "yes" or "no"
- piece_at: color + type (e.g. "white rook") or "empty"
"""
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Board image rendering
# ---------------------------------------------------------------------------


_IMAGE_CACHE_DIR = os.path.expanduser("~/.cache/chess-ai/board-images")


def _cache_key(fen: str, size: int, flipped: bool) -> str:
    import hashlib

    tag = f"{fen}|{size}|{flipped}"
    return hashlib.md5(tag.encode()).hexdigest()


def _get_cached_image_path(fen: str, size: int = 256, flipped: bool = False) -> str:
    """Return the cache path for a board image, rendering it if not already cached."""
    import cairosvg
    import chess
    import chess.svg

    os.makedirs(_IMAGE_CACHE_DIR, exist_ok=True)
    key = _cache_key(fen, size, flipped)
    cache_path = os.path.join(_IMAGE_CACHE_DIR, f"{key}.png")

    if not os.path.exists(cache_path):
        board = chess.Board(fen)
        svg = chess.svg.board(board, size=size, flipped=flipped)
        png = cairosvg.svg2png(bytestring=svg.encode())
        assert png is not None, f"cairosvg returned None for {fen}"
        with open(cache_path, "wb") as f:
            f.write(png)

    return cache_path


def render_board_png_cached(fen: str, size: int = 256, flipped: bool = False) -> bytes:
    """Render a chess position to PNG bytes, caching to disk."""
    import cairosvg
    import chess
    import chess.svg

    os.makedirs(_IMAGE_CACHE_DIR, exist_ok=True)
    key = _cache_key(fen, size, flipped)
    cache_path = os.path.join(_IMAGE_CACHE_DIR, f"{key}.png")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    board = chess.Board(fen)
    svg = chess.svg.board(board, size=size, flipped=flipped)
    png = cairosvg.svg2png(bytestring=svg.encode())

    with open(cache_path, "wb") as f:
        f.write(png)
    return png


def _fen_is_black_to_move(fen: str) -> bool:
    parts = fen.split()
    return len(parts) > 1 and parts[1] == "b"


def png_to_pil(png_bytes: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_grpo_dataset(jsonl_path: str, board_size: int = 256, task_filter: list[str] | None = None):
    """Load dataset and render board images from FEN.

    TRL GRPOTrainer expects:
      - "prompt"  : list of chat messages (plain text, no image dicts)
      - "images"  : list of PIL Images for that sample (one image per sample)
    TRL calls prepare_multimodal_messages internally to inject images into prompts.
    PIL Images must be stored via datasets.Image feature to avoid PyArrow errors.
    """
    from datasets import Dataset, Features, Image, Value

    rows: list[dict] = []
    skipped = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msgs = rec.get("messages", [])
            if not msgs:
                continue
            meta = rec.get("metadata", {})
            fen = meta.get("fen", "")
            if not fen:
                skipped += 1
                continue

            if task_filter and meta.get("task", "") not in task_filter:
                skipped += 1
                continue

            flipped = _fen_is_black_to_move(fen)
            try:
                image_path = _get_cached_image_path(fen, size=board_size, flipped=flipped)
            except Exception as exc:
                log.debug("Image render failed for %s: %s", fen, exc)
                skipped += 1
                continue

            # Inject improved system prompt, keep user messages as-is
            prompt = []
            for m in msgs:
                if m["role"] == "assistant":
                    continue
                if m["role"] == "system":
                    prompt.append({"role": "system", "content": _SYSTEM_PROMPT})
                else:
                    prompt.append(m)

            rows.append(
                {
                    "prompt": prompt,
                    "images": [image_path],  # file path — datasets.Image loads lazily
                    "fen": fen,
                    "task": meta.get("task", ""),
                    "answer": meta.get("answer", ""),
                }
            )

    log.info("Loaded %d GRPO rows from %s (skipped=%d)", len(rows), jsonl_path, skipped)

    # datasets.Image() accepts file paths and lazy-loads — much faster than encoding PIL objects
    features = Features(
        {
            "prompt": [{"role": Value("string"), "content": Value("string")}],
            "images": [Image()],
            "fen": Value("string"),
            "task": Value("string"),
            "answer": Value("string"),
        }
    )
    return Dataset.from_list(rows, features=features)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def setup_model(config: dict):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_cfg = config["model"]
    lora_cfg = config["lora"]

    log.info("Loading %s", model_cfg["model_name"])
    model = AutoModelForImageTextToText.from_pretrained(
        model_cfg["model_name"],
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(model_cfg["model_name"], trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


# ---------------------------------------------------------------------------
# Reward: rule-based exact match
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def _normalize_move_list(text: str) -> frozenset[str]:
    tokens = re.split(r"[\n,]+", text)
    return frozenset(t.strip() for t in tokens if t.strip())


def _score_answer(task: str, predicted: str, gold: str) -> float:
    if task in ("legal_moves", "captures"):
        pred_set = _normalize_move_list(predicted)
        gold_set = _normalize_move_list(gold)
        if not gold_set:
            return 0.0
        intersection = len(pred_set & gold_set)
        union = len(pred_set | gold_set)
        return intersection / union if union > 0 else 0.0
    elif task in ("in_check", "piece_at"):
        return 1.0 if predicted.lower().strip() == gold.lower().strip() else 0.0
    return 0.0


def build_reward_fn(train_cfg: dict):
    log_path = "completions_phase1_board_reading.jsonl"
    step_counter = [0]

    # Linear length penalty: 0 at ≤ free_tokens, scales to max_penalty at max_completion_length
    len_penalty_free = train_cfg.get("len_penalty_free_tokens", 500)
    len_penalty_max = train_cfg.get("len_penalty_max", 0.2)
    len_penalty_cap = train_cfg.get("max_completion_length", 6144)

    def _len_penalty(text: str) -> float:
        n = len(text.split())
        excess = max(0, n - len_penalty_free)
        ratio = min(1.0, excess / max(1, len_penalty_cap - len_penalty_free))
        return len_penalty_max * ratio

    def _completion_text(completion) -> str:
        if isinstance(completion, list):
            for msg in reversed(completion):
                if msg.get("content"):
                    return msg["content"]
        return str(completion).strip()

    def board_reading_reward(prompts, completions, **kwargs) -> list[float]:
        tasks = kwargs.get("task", [""] * len(prompts))
        gold_answers = kwargs.get("answer", [""] * len(prompts))
        fens = kwargs.get("fen", [""] * len(prompts))

        scores: list[float] = []
        log_records: list[dict] = []

        for prompt, comp, task, gold, fen in zip(prompts, completions, tasks, gold_answers, fens):
            text = _completion_text(comp)
            # Extract user message text from prompt (last user turn)
            prompt_text = ""
            if isinstance(prompt, list):
                for msg in reversed(prompt):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            # Multimodal: extract text parts only (skip PngImageFile etc.)
                            prompt_text = " ".join(
                                c.get("text", "") if isinstance(c, dict) else str(c)
                                for c in content
                                if not isinstance(c, dict) or c.get("type") == "text"
                            )
                        else:
                            prompt_text = str(content)
                        break

            has_answer_tag = 1.0 if _ANSWER_RE.search(text) else 0.0
            predicted = _extract_answer(text)
            exact = _score_answer(task, predicted, gold) if predicted else 0.0
            penalty = _len_penalty(text)
            # Rewards: format(0.10) + exact(0.90) - length_penalty(0.1)
            score = 0.10 * has_answer_tag + 0.90 * exact - penalty
            scores.append(score)
            log_records.append(
                {
                    "fen": fen,
                    "task": task,
                    "gold": gold,
                    "predicted": predicted,
                    "score": score,
                    "penalty": penalty,
                    "has_tag": bool(has_answer_tag),
                    "prompt": prompt_text,
                    "completion": text,
                }
            )

        step_counter[0] += 1
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        valid = [(i, s) for i, s in enumerate(scores)]
        best_i = max(valid, key=lambda x: x[1])[0]
        worst_i = min(valid, key=lambda x: x[1])[0]

        record = {
            "step": step_counter[0],
            "ts": ts,
            "mean": sum(scores) / len(scores) if scores else None,
            "scores": scores,
            "best": log_records[best_i],
            "worst": log_records[worst_i] if worst_i != best_i else None,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return scores

    board_reading_reward.__name__ = "board_reading_reward"
    return board_reading_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="recipes-train/qwen3.5-4b-grpo-phase1/config.yaml"
    )
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    wandb_cfg = config.get("wandb", {})

    if wandb_cfg.get("enabled"):
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "chess-tutor-grpo"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags", []),
        )

    model, processor = setup_model(config)

    board_size = train_cfg.get("board_size", 256)
    task_filter = train_cfg.get("task_filter", None)
    if task_filter:
        log.info("Curriculum task filter: %s", task_filter)
    train_dataset = load_grpo_dataset(
        train_cfg["train_file"], board_size=board_size, task_filter=task_filter
    )
    eval_dataset = None
    if train_cfg.get("eval_file"):
        eval_dataset = load_grpo_dataset(
            train_cfg["eval_file"], board_size=board_size, task_filter=task_filter
        )

    reward_fn = build_reward_fn(train_cfg)

    grpo_config = GRPOConfig(
        output_dir=config.get("output_dir", "checkpoints/qwen3.5-4b-grpo-phase1"),
        num_generations=train_cfg.get("num_generations", 8),
        max_completion_length=train_cfg.get("max_completion_length", 1024),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        steps_per_generation=train_cfg.get("steps_per_generation", None),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 10),
        optim=train_cfg.get("optim", "adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.1),
        logging_steps=train_cfg.get("logging_steps", 1),
        logging_first_step=True,
        eval_strategy=train_cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 100),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 50),
        save_total_limit=train_cfg.get("save_total_limit", 10),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_empty_cache_steps=train_cfg.get("torch_empty_cache_steps", None),
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        beta=train_cfg.get("beta", 0.0),
        epsilon=train_cfg.get("epsilon", 0.2),
        temperature=train_cfg.get("temperature", 0.8),
        top_p=train_cfg.get("top_p", 0.95),
        use_vllm=False,
        use_liger_kernel=True,
        report_to="wandb" if wandb_cfg.get("enabled") else "none",
        remove_unused_columns=False,
    )

    # TRL accesses model.warnings_issued — PEFT doesn't proxy this attribute.
    if not hasattr(model.base_model.model, "warnings_issued"):
        model.base_model.model.warnings_issued = {}

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    log.info("Starting GRPO Phase-1 board-reading training (both GPUs)...")
    resume = args.resume
    if resume == "latest":
        # Find latest checkpoint in output_dir
        import glob as _glob

        ckpts = sorted(
            _glob.glob(
                os.path.join(
                    config.get("output_dir", "checkpoints/qwen3.5-4b-grpo-phase1"), "checkpoint-*"
                )
            )
        )
        resume = ckpts[-1] if ckpts else None
        if resume:
            log.info("Resuming from latest checkpoint: %s", resume)
    trainer.train(resume_from_checkpoint=resume)

    out = config.get("output_dir", "checkpoints/qwen3.5-4b-grpo-phase1")
    log.info("Saving model to %s", out)
    trainer.save_model()
    processor.save_pretrained(out)
    log.info("Done.")


if __name__ == "__main__":
    main()
