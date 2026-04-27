"""GRPO Phase 1 — Qwen3.5-4B + LoRA + LLM Judge reward.

Vanilla Qwen3.5-4B with LoRA, no chess encoder. Goal: verify the GRPO +
judge reward loop produces improving completions before adding the encoder.

Reward: GCC-Eval judge (Qwen3.5-35B-A3B-GPTQ-Int4, port 8400).
Both GPUs available for training — judge runs via OpenRouter API.

Usage:
    ./recipes-train/qwen3.5-4b-grpo-phase1/start.sh
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

import unsloth  # must be imported before transformers/trl
import yaml
from trl import GRPOConfig, GRPOTrainer

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
# Tool definitions (injected via chat template)
# ---------------------------------------------------------------------------
# Tools (passed to GRPOTrainer — TRL injects them via apply_chat_template)
# ---------------------------------------------------------------------------

CHESS_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "stockfish_eval",
            "description": (
                "Evaluate a chess position with Stockfish. "
                "Returns centipawn score (white's perspective), best move in UCI and SAN, "
                "and mate_in (null if no forced mate)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {"type": "string", "description": "FEN string of the position."},
                    "depth": {"type": "integer", "description": "Search depth (default 18)."},
                },
                "required": ["fen"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sf15_term_diff",
            "description": (
                "Run Stockfish 15 classical evaluation before and after a move, "
                "returning per-term diffs from the moving side's perspective. "
                "Terms: Mobility, King safety, Threats, Material, Pawns, Bishops, "
                "Rooks, Queens, Space, Passed, Initiative. "
                "Positive diff = term improved for the player who made the move."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "FEN of the position BEFORE the move.",
                    },
                    "move_san": {"type": "string", "description": "Move in SAN notation."},
                },
                "required": ["fen", "move_san"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_best_move",
            "description": "Get the best move for a position according to Stockfish.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {"type": "string", "description": "FEN string of the position."},
                },
                "required": ["fen"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert chess coach giving move-by-move feedback to a student.

You have tools to analyse the position. Use them immediately — don't speculate, just call the tool.

Workflow:
1. <think>: Keep it SHORT. Identify the move, call sf15_term_diff right away, read the result, done. \
No long reasoning — the tool gives you the facts.
2. After </think>: write your coaching comment — direct, second person, 2-5 sentences max.

Think block rules:
- CALL TOOLS FIRST, reason after. Do not think for more than a few sentences before calling a tool.
- One or two tool calls is enough. Do not over-analyse.
- Keep the entire think block under 300 words.

Coaching comment rules:
- Second person ("You centralise the knight", "You play Nd5 to...").
- Name the key strategic or tactical idea and explain WHY this move achieves it.
- Reference specific moves, squares, or pieces — never vague platitudes.
- OPENING (moves 1-12): 2-3 sentences, identify opening/variation.
- MIDDLEGAME: key idea + top alternative, 3-4 sentences.
- ENDGAME: key lines, 4-5 sentences.
- Do NOT parrot engine numbers, generic study advice, or opening recommendations.
"""

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_messages(raw_messages: list[dict]) -> list[dict]:
    """Pass through messages, skipping the assistant turn (GRPO generates it)."""
    return [msg for msg in raw_messages if msg["role"] != "assistant"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_grpo_dataset(jsonl_path: str, tokenizer):
    from datasets import Dataset

    rows: list[dict] = []
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
            move_san = meta.get("move_san", "")

            prompt_msgs = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs
            phase_msgs = _build_messages(prompt_msgs)

            rows.append(
                {
                    "prompt": phase_msgs,
                    "fen": fen,
                    "move_san": move_san,
                    "chat_template_kwargs": {"tools": CHESS_TOOLS},
                }
            )

    log.info("Loaded %d GRPO rows from %s", len(rows), jsonl_path)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def setup_model(config: dict):
    from unsloth import FastLanguageModel

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    base_model_name = model_cfg["model_name"]
    train_cfg = config["training"]

    max_seq_len = train_cfg.get("max_completion_length", 8192) + 512  # prompt headroom

    log.info("Loading %s with Unsloth (max_seq_len=%d)", base_model_name, max_seq_len)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_len,
        dtype=None,  # auto-detect bfloat16
        load_in_4bit=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log.info("Applying LoRA (r=%d, alpha=%d)", lora_cfg.get("r", 32), lora_cfg.get("alpha", 64))
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Reward function with completion logging
# ---------------------------------------------------------------------------


def build_reward_fn(train_cfg: dict):
    from verification import rewards_llm_judge as _rj

    judge_url = train_cfg.get("judge_server_url", "http://localhost:8400")
    judge_timeout = float(train_cfg.get("judge_timeout", 120.0))
    _rj.JUDGE_SERVER_URL = judge_url
    _rj._JUDGE_TIMEOUT = judge_timeout
    log.info("Judge server: %s (timeout=%.1fs)", judge_url, judge_timeout)

    log_path = "completions_phase2_grpo.jsonl"
    step_counter = [0]

    _judge_metric_keys = (
        "correctness",
        "think_quality",
        "completeness",
        "relevance",
        "clarity",
        "fluency",
    )

    def _prompt_text(prompt) -> str:
        if isinstance(prompt, list):
            parts = [f"[{m.get('role', '?').upper()}]\n{m.get('content', '')}" for m in prompt]
            return "\n\n".join(parts)
        return str(prompt)

    def _completion_text(completion) -> str:
        if isinstance(completion, list):
            for msg in reversed(completion):
                if msg.get("content"):
                    return msg["content"]
        return str(completion).strip()

    def _build_sample(label, i, ts, prompt, completion, fen, move_san, bd, score, judge_input):
        return {
            "label": label,
            "idx": i,
            "ts": ts,
            "reward": score,
            "fen": fen,
            "move": move_san,
            "trainee_input": _prompt_text(prompt),
            "trainee_output": _completion_text(completion),
            "judge_input": judge_input,
            "judge_output": bd,
        }

    def _log_completions(
        prompts, completions, reward_scores, judge_breakdowns, fens, move_sans, judge_inputs
    ):
        n = len(completions)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step_counter[0] += 1

        valid = [(i, s) for i, s in enumerate(reward_scores) if s is not None]
        best_i = max(valid, key=lambda x: x[1])[0] if valid else 0
        worst_i = min(valid, key=lambda x: x[1])[0] if valid else 0
        mean_score = sum(s for s in reward_scores if s is not None) / len(valid) if valid else None

        def build(label, i):
            return _build_sample(
                label,
                i,
                ts,
                prompts[i],
                completions[i],
                fens[i] if i < len(fens) else "",
                move_sans[i] if i < len(move_sans) else "",
                judge_breakdowns[i],
                reward_scores[i],
                judge_inputs[i] if i < len(judge_inputs) else {},
            )

        record = {
            "step": step_counter[0],
            "ts": ts,
            "n": n,
            "n_judged": len(valid),
            "mean": mean_score,
            "scores": reward_scores,
            "best": build("BEST", best_i),
            "worst": build("WORST", worst_i) if worst_i != best_i else None,
            "all": [build(f"#{i}", i) for i in range(n)],
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def combined_reward(prompts, completions, **kwargs):
        scores = _rj.combined_reward(prompts, completions, **kwargs)

        none_count = sum(s is None for s in scores)
        if none_count == len(scores):
            raise RuntimeError(
                f"Judge returned None for ALL {len(scores)} completions. "
                "Inference server is down — stopping training. "
                "Restart with: ./recipes-inference/start.sh && ./recipes-train/qwen3.5-4b-grpo-phase1/start.sh --resume"
            )
        if none_count > 0:
            log.warning("%d/%d judge scores are None (partial failure)", none_count, len(scores))

        fens = kwargs.get("fen", [""] * len(prompts))
        move_sans = kwargs.get("move_san", [""] * len(prompts))
        judge_breakdowns: list[dict] = []
        judge_inputs: list[dict] = []

        for i, (prompt, completion, fen, move_san) in enumerate(
            zip(prompts, completions, fens, move_sans)
        ):
            traj = _rj._completion_str(completion)
            comment = _rj.extract_comment(traj)
            engine_eval = _rj.extract_engine_eval_from_trajectory(traj)
            judge_inputs.append(
                {"fen": fen, "move_san": move_san, "engine_eval": engine_eval, "comment": comment}
            )
            if scores[i] is None:
                judge_breakdowns.append({k: None for k in _judge_metric_keys})
            else:
                judge_breakdowns.append(
                    {k: None for k in _judge_metric_keys}
                )  # breakdown via score only

        _log_completions(
            prompts, completions, scores, judge_breakdowns, fens, move_sans, judge_inputs
        )
        return scores

    combined_reward.__name__ = "combined_reward"
    return combined_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="recipes-train/qwen3.5-4b-grpo-phase1/config.yaml"
    )
    parser.add_argument("--resume", nargs="?", const=True, default=None)
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

    model, tokenizer = setup_model(config)

    train_dataset = load_grpo_dataset(train_cfg["train_file"], tokenizer)
    eval_dataset = None
    if train_cfg.get("eval_file"):
        eval_dataset = load_grpo_dataset(train_cfg["eval_file"], tokenizer)

    combined_reward = build_reward_fn(train_cfg)

    grpo_config = GRPOConfig(
        output_dir=config.get("output_dir", "checkpoints/qwen3.5-4b-grpo-phase1"),
        num_generations=train_cfg.get("num_generations", 4),
        max_completion_length=train_cfg.get("max_completion_length", 8192),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        steps_per_generation=train_cfg.get("steps_per_generation", None),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        learning_rate=train_cfg.get("learning_rate", 3e-6),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 5),
        optim=train_cfg.get("optim", "adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.1),
        logging_steps=train_cfg.get("logging_steps", 1),
        logging_first_step=True,
        eval_strategy=train_cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 100),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 10),
        save_total_limit=train_cfg.get("save_total_limit", 10),
        bf16=train_cfg.get("bf16", True),
        # Gradient checkpointing is handled by Unsloth's "unsloth" mode in get_peft_model
        gradient_checkpointing=False,
        torch_empty_cache_steps=train_cfg.get("torch_empty_cache_steps", 1),
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        beta=train_cfg.get("beta", 0.0),
        epsilon=train_cfg.get("epsilon", 0.2),
        temperature=train_cfg.get("temperature", 1.0),
        top_p=train_cfg.get("top_p", 0.95),
        use_vllm=False,
        report_to="wandb" if wandb_cfg.get("enabled") else "none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[combined_reward],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    log.info("Starting GRPO Phase-1 training (vanilla Qwen3.5-4B + LoRA + judge reward)...")
    trainer.train(resume_from_checkpoint=args.resume)

    out = config.get("output_dir", "checkpoints/qwen3.5-4b-grpo-phase1")
    log.info("Saving model to %s", out)
    trainer.save_model()
    tokenizer.save_pretrained(out)
    log.info("Done.")


if __name__ == "__main__":
    main()
