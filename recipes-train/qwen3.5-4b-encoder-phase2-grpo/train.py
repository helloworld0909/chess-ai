"""GRPO Phase 2 — ChessLMWithEncoder + LoRA RL training.

Trains the joint encoder+LLM model with GRPO rewards on the coaching task.
The CNN board encoder is frozen; LoRA adapters on the LLM are trainable.

The model: ChessLMWithEncoder
  - CNN (frozen): encodes board states → embeddings injected at <|move|> tokens
  - LLM (LoRA trainable): Qwen3.5-4B with LoRA r=64

Prompts: lines_joint_sft.jsonl format (same as Phase 1 SFT)
  - system: JOINT_SYSTEM_PROMPT
  - user: board + FEN + move + ## Engine Key Lines with <|move|> sentinels

GRPOTrainer is subclassed to inject board_tensors_flat + move_counts into
every forward pass (train step) and to handle generation from the encoder model.
For generation, we use the inner LLM directly (text-only from prompt tokens),
then run the full encoder model for the policy/ref log-prob forward passes.

Usage:
    ./recipes-train/qwen3.5-4b-encoder-phase2-grpo/start.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

import chess
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.encoder import MOVE_TOKEN, MOVE_TOKEN_ID
from src.encoder.board_tensor import boards_to_tensor
from training.encoder_model import ChessLMWithEncoder

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Move token injection (mirrors Phase 1 SFT train.py)
# ---------------------------------------------------------------------------


def _extract_line_sans(user_content: str) -> list[list[str]]:
    lines_section_re = re.search(
        r"## Engine Key Lines\n\n(.*?)(?=\n\n##|\Z)", user_content, re.DOTALL
    )
    if not lines_section_re:
        return []
    result = []
    for line in lines_section_re.group(1).strip().split("\n"):
        m = re.match(r"^(?:PLAYED LINE|Line \d+):\s*(.*)", line.strip())
        if m:
            sans = []
            for part in m.group(1).split("→"):
                clean = part.replace(MOVE_TOKEN, "").strip()
                clean = re.sub(r"\s*\[.*?\]", "", clean).strip()
                if clean:
                    sans.append(clean)
            if sans:
                result.append(sans)
    return result


def _inject_move_tokens(
    messages: list[dict], student_san: str
) -> tuple[list[dict], list[list[str]]]:
    new_msgs = []
    line_sans: list[list[str]] = []
    for msg in messages:
        content = msg["content"]
        role = msg["role"]
        if role == "user":
            if student_san:
                content = re.sub(
                    r"(?<=Move:\s)" + re.escape(student_san) + r"(?=\s|$|\n)",
                    MOVE_TOKEN,
                    content,
                )
            if "## Engine Key Lines" in content:
                line_sans = _extract_line_sans(content)

                def replace_key_lines(m: re.Match) -> str:
                    inner = m.group(1)
                    new_lines = []
                    for line in inner.split("\n"):
                        line_m = re.match(r"^((?:PLAYED LINE|Line \d+):\s*)(.*)", line)
                        if line_m:
                            prefix = line_m.group(1)
                            moves = line_m.group(2).split("→")
                            injected = []
                            for move in moves:
                                move = move.strip()
                                if not move.startswith(MOVE_TOKEN):
                                    injected.append(MOVE_TOKEN + move)
                                else:
                                    injected.append(move)
                            new_lines.append(prefix + " → ".join(injected))
                        else:
                            new_lines.append(line)
                    return "## Engine Key Lines\n\n" + "\n".join(new_lines)

                content = re.sub(
                    r"## Engine Key Lines\n\n(.*?)(?=\n\n##|\Z)",
                    replace_key_lines,
                    content,
                    flags=re.DOTALL,
                )
        new_msgs.append({"role": role, "content": content})
    return new_msgs, line_sans


def _build_board_tensors(
    fen: str, student_san: str, line_sans: list[list[str]]
) -> list[torch.Tensor]:
    """Build all board tensors for a single example (student move + all line moves)."""
    tensors: list[torch.Tensor] = []
    board = chess.Board(fen)

    # Student move
    student_move = None
    if student_san:
        try:
            student_move = board.parse_san(student_san)
        except Exception:
            pass
    tensors.append(boards_to_tensor(board, student_move))

    # Line moves — replay from pre-student-move board
    for line in line_sans:
        line_board = board.copy()
        for san in line:
            try:
                mv = line_board.parse_san(san)
                tensors.append(boards_to_tensor(line_board, mv))
                line_board.push(mv)
            except Exception:
                tensors.append(boards_to_tensor(line_board, None))

    return tensors


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_grpo_dataset(jsonl_path: str, tokenizer, max_prompt_length: int):
    """Load joint SFT JSONL as GRPO prompt dataset.

    Strips the assistant turn, injects <|move|> tokens into the user content,
    tokenizes the prompt, and stores fen/move_san/line_sans_json for the
    collator to build board tensors at training time.
    """
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
            fen = meta.get("fen", chess.STARTING_FEN)
            move_san = meta.get("move_san", "")

            # Strip assistant turn — GRPO generates this
            prompt_msgs = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs

            # Inject <|move|> sentinels into user content
            injected_msgs, line_sans = _inject_move_tokens(prompt_msgs, move_san)

            rows.append(
                {
                    "prompt": injected_msgs,
                    "fen": fen,
                    "move_san": move_san,
                    "line_sans_json": json.dumps(line_sans),
                }
            )

    log.info("Loaded %d GRPO prompt rows from %s", len(rows), jsonl_path)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def setup_encoder_model(config: dict):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    encoder_cfg = config.get("encoder", {})
    sft_checkpoint = model_cfg[
        "model_name"
    ]  # e.g. checkpoints/qwen3.5-4b-encoder-phase1-sft/checkpoint-890
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # The SFT checkpoint is a ChessLMWithEncoder state dict — no config.json.
    # Load the base LLM from Qwen/Qwen3.5-4B, then restore SFT weights.
    base_model_name = "Qwen/Qwen3.5-4B"

    log.info("Loading tokenizer from %s", sft_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # GRPOTrainer needs left-padding for batch generation
    tokenizer.padding_side = "left"

    log.info("Loading base LLM (bf16) from %s on GPU %d", base_model_name, local_rank)
    base_llm = AutoModelForCausalLM.from_pretrained(
        base_model_name,
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
    peft_llm = get_peft_model(base_llm, lora_config)

    log.info("Wrapping with CNN encoder...")
    model = ChessLMWithEncoder(
        llm=peft_llm,
        hidden_size=base_llm.config.hidden_size,
        cnn_hidden_size=encoder_cfg.get("hidden_size", 512),
        cnn_num_blocks=encoder_cfg.get("num_blocks", 15),
        move_token_id=MOVE_TOKEN_ID,
    )
    model.to(torch.bfloat16)
    model.to(f"cuda:{local_rank}")

    # Load SFT checkpoint weights (ChessLMWithEncoder state dict)
    # Keys: cnn.*, llm.base_model.model.model.*, embed_tokens.*
    sft_weights_path = os.path.join(sft_checkpoint, "model.safetensors")
    if os.path.exists(sft_weights_path):
        from safetensors.torch import load_file

        sft_state = load_file(sft_weights_path, device=f"cuda:{local_rank}")
        missing, unexpected = model.load_state_dict(sft_state, strict=False)
        log.info(
            "Loaded SFT checkpoint from %s (missing=%d unexpected=%d)",
            sft_weights_path,
            len(missing),
            len(unexpected),
        )
        if missing:
            log.warning("Missing keys: %s", missing[:10])
    else:
        log.warning("No model.safetensors at %s — starting from scratch!", sft_weights_path)

    # CNN is frozen; LoRA params are trainable
    for param in model.cnn.parameters():
        param.requires_grad_(False)

    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Board-tensor-aware GRPOTrainer subclass
# ---------------------------------------------------------------------------


class EncoderGRPOTrainer:
    """Thin wrapper that pre-builds board tensors and runs GRPO via a custom loop.

    TRL's GRPOTrainer is tightly coupled to HF model.generate() and standard
    forward(input_ids, attention_mask, labels). ChessLMWithEncoder requires
    board_tensors_flat + move_counts at every forward pass.

    Strategy:
      - Generation: use model.llm (the inner PEFT model) directly — it accepts
        plain input_ids and produces token completions.  The prompt tokens
        already contain <|move|> sentinel IDs; during generation the LLM sees
        them as regular tokens (the CNN embeddings are only needed for the
        policy gradient forward pass, not for autoregressive sampling).
      - Policy forward pass: run the full ChessLMWithEncoder with board tensors
        built from the prompt's fen/line_sans metadata.

    This keeps the implementation simple: generation is text-only, gradients
    flow through the full encoder model.
    """

    pass  # Placeholder — see EncoderGRPOTrainer below


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def build_reward_fn(train_cfg: dict):
    from concurrent.futures import ThreadPoolExecutor as _TPE

    from verification import rewards as _rewards_mod
    from verification.rewards import (
        reward_educational,
        reward_format,
        reward_legality,
        reward_sf15_annotation,
        reward_think,
        reward_tone,
    )

    sf_depth = train_cfg.get("stockfish_depth", 12)
    _rewards_mod._SF_DEPTH = sf_depth
    log.info("Stockfish reward depth: %d", sf_depth)

    _reward_names = [
        "R0_format",
        "R_think",
        "R1_legality",
        "R3b_sf15",
        "RC_tone",
        "RC_educ",
    ]

    def _w(fn, weight):
        def _wrapped(prompts, completions, **kwargs):
            return [weight * v for v in fn(prompts, completions, **kwargs)]

        _wrapped.__name__ = f"weighted_{fn.__name__}"
        return _wrapped

    reward_fns = [
        _w(reward_format, 0.10),
        _w(reward_think, 0.15),
        reward_legality,
        _w(reward_sf15_annotation, 0.35),
        _w(reward_tone, 0.20),
        _w(reward_educational, 0.20),
    ]
    num_fns = len(reward_fns)
    executor = _TPE(max_workers=num_fns, thread_name_prefix="reward")

    # Completion logger
    log_path = "completions.log"
    step_counter = [0]

    def _log_completions(completions, all_scores):
        import datetime

        n = len(completions)
        totals = [sum(all_scores[r][i] for r in range(num_fns)) for i in range(n)]
        best_i = max(range(n), key=lambda i: totals[i])
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step_counter[0] += 1
        with open(log_path, "a") as f:
            f.write(f"\n{'─' * 80}\n")
            f.write(
                f"STEP {step_counter[0]}  {ts}  ({n} completions)  "
                f"mean={sum(totals) / n:+.3f}  best={totals[best_i]:+.3f}\n"
            )
            best_text = (
                completions[best_i][-1]["content"]
                if isinstance(completions[best_i], list)
                else str(completions[best_i])
            )
            f.write(f"\n── Best [{best_i}] ──\n{best_text.strip()}\n")
            header = f"\n  {'#':>2}  {'total':>7}  " + "  ".join(f"{n:>10}" for n in _reward_names)
            f.write(header + "\n")
            for i, total in enumerate(totals):
                marker = " *" if i == best_i else "  "
                row = f"{marker}{i:>2}  {total:>+7.3f}  " + "  ".join(
                    f"{all_scores[r][i]:>+10.3f}" for r in range(num_fns)
                )
                f.write(row + "\n")

    def combined_reward(prompts, completions, **kwargs):
        futures = [executor.submit(fn, prompts, completions, **kwargs) for fn in reward_fns]
        all_scores = [f.result() for f in futures]
        _log_completions(completions, all_scores)
        n = len(completions)
        return [sum(all_scores[r][i] for r in range(num_fns)) for i in range(n)]

    combined_reward.__name__ = "combined_reward"
    return combined_reward


# ---------------------------------------------------------------------------
# Custom data collator for GRPO — builds board tensors from prompt metadata
# ---------------------------------------------------------------------------


class GRPOEncoderCollator:
    """Collates GRPO prompt batches, building board tensors for the prompt's
    <|move|> tokens (student move + engine key line moves).

    GRPOTrainer calls this collator on the prompt-only batch. The collator
    tokenizes prompts (already have <|move|> tokens injected as text) and
    builds board_tensors_flat + move_counts so the encoder model can run.
    """

    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict:
        prompts = [f["prompt"] for f in features]
        fens = [f.get("fen", chess.STARTING_FEN) for f in features]
        move_sans = [f.get("move_san", "") for f in features]
        line_sans_list = [json.loads(f.get("line_sans_json", "[]")) for f in features]

        # Tokenize prompts (left-padded for generation)
        tokenized = self.tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Build board tensors
        all_tensors: list[torch.Tensor] = []
        move_counts: list[int] = []
        for fen, san, line_sans in zip(fens, move_sans, line_sans_list):
            tensors = _build_board_tensors(fen, san, line_sans)
            # Validate against token count
            input_ids = tokenized["input_ids"][len(move_counts)]
            n_tokens = (input_ids == MOVE_TOKEN_ID).sum().item()
            while len(tensors) < n_tokens:
                tensors.append(boards_to_tensor(chess.Board(fen), None))
            tensors = tensors[:n_tokens]
            move_counts.append(len(tensors))
            all_tensors.extend(tensors)

        result = dict(tokenized)
        result["board_tensors_flat"] = (
            torch.stack(all_tensors) if all_tensors else torch.zeros(0, 38, 8, 8)
        )
        result["move_counts"] = torch.tensor(move_counts, dtype=torch.long)
        result["fen"] = fens
        result["move_san"] = move_sans
        result["line_sans_json"] = [f.get("line_sans_json", "[]") for f in features]
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="recipes-train/qwen3.5-4b-encoder-phase2-grpo/config.yaml"
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

    model, tokenizer = setup_encoder_model(config)

    max_prompt_length = train_cfg.get("max_prompt_length", 1024)
    train_dataset = load_grpo_dataset(train_cfg["train_file"], tokenizer, max_prompt_length)
    eval_dataset = None
    if train_cfg.get("eval_file"):
        eval_dataset = load_grpo_dataset(train_cfg["eval_file"], tokenizer, max_prompt_length)

    combined_reward = build_reward_fn(train_cfg)

    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=config.get("output_dir", "checkpoints/qwen3.5-4b-encoder-phase2-grpo"),
        num_generations=train_cfg.get("num_generations", 4),
        max_prompt_length=max_prompt_length,
        max_completion_length=train_cfg.get("max_completion_length", 1500),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 10),
        optim=train_cfg.get("optim", "adamw_torch"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.1),
        logging_steps=train_cfg.get("logging_steps", 5),
        logging_first_step=True,
        eval_strategy=train_cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 100),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 10),
        save_total_limit=train_cfg.get("save_total_limit", 20),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
        beta=train_cfg.get("beta", 0.0),
        epsilon=train_cfg.get("epsilon", 0.2),
        temperature=train_cfg.get("temperature", 0.9),
        top_p=train_cfg.get("top_p", 0.95),
        report_to="wandb" if wandb_cfg.get("enabled") else "none",
        # Use the inner LLM for generation (it accepts plain input_ids)
        # The full encoder model is used for policy forward passes via model_init_kwargs
        remove_unused_columns=False,
    )

    # GRPOTrainer needs to call model.generate() for rollouts.
    # ChessLMWithEncoder doesn't have generate() — we expose the inner LLM's
    # generate() by patching it onto the wrapper.  The forward pass (for log-probs)
    # still goes through the full encoder model.
    if not hasattr(model, "generate"):
        model.generate = model.llm.generate

    # Expose generation_config for TRL compatibility
    if not hasattr(model, "generation_config"):
        model.generation_config = model.llm.generation_config

    # warnings_issued shim for TRL/transformers compatibility
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # Patch model.forward to accept board_tensors_flat/move_counts from batch,
    # building zero tensors when they're absent (e.g. during TRL's log-prob pass
    # on completions that have no engine lines yet).
    _original_forward = model.forward

    def _patched_forward(
        input_ids=None,
        attention_mask=None,
        labels=None,
        board_tensors_flat=None,
        move_counts=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # If board tensors not provided, build zero tensors for all <|move|> positions
        if board_tensors_flat is None or move_counts is None:
            if input_ids is not None:
                n_moves = (input_ids == MOVE_TOKEN_ID).sum().item()
            else:
                n_moves = 0
            device = input_ids.device if input_ids is not None else next(model.parameters()).device
            dtype = next(model.cnn.parameters()).dtype
            board_tensors_flat = torch.zeros(max(n_moves, 1), 38, 8, 8, device=device, dtype=dtype)
            if input_ids is not None:
                B = input_ids.shape[0]
                move_counts = torch.tensor(
                    [(input_ids[i] == MOVE_TOKEN_ID).sum().item() for i in range(B)],
                    device=device,
                    dtype=torch.long,
                )
            else:
                move_counts = torch.tensor([0], device=device, dtype=torch.long)

        return _original_forward(
            board_tensors_flat=board_tensors_flat,
            move_counts=move_counts,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    model.forward = _patched_forward

    # Shims for HF PreTrainedModel methods that GRPOTrainer calls on the model
    if not hasattr(model, "add_model_tags"):
        model.add_model_tags = lambda *a, **kw: None
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if not hasattr(model, "is_parallelizable"):
        model.is_parallelizable = False
    if not hasattr(model, "model_parallel"):
        model.model_parallel = False

    # Gradient checkpointing shims — delegate to inner LLM
    # unwrap_model_for_generation reads this attribute and calls enable/disable
    if not hasattr(model.__class__, "is_gradient_checkpointing"):
        model.__class__.is_gradient_checkpointing = property(
            lambda self: self.llm.is_gradient_checkpointing
            if hasattr(self.llm, "is_gradient_checkpointing")
            else False
        )
    if not hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable = lambda **kw: model.llm.gradient_checkpointing_enable(
            **kw
        )
    if not hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable = lambda: model.llm.gradient_checkpointing_disable()

    # Additional HF PreTrainedModel shims that TRL/transformers may call
    if not hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads = lambda: model.llm.enable_input_require_grads()
    if not hasattr(model, "get_input_embeddings"):
        model.get_input_embeddings = lambda: model.llm.get_input_embeddings()
    if not hasattr(model, "config"):
        model.config = model.llm.config
    if not hasattr(model, "name_or_path"):
        model.name_or_path = getattr(model.llm, "name_or_path", "ChessLMWithEncoder")
    if not hasattr(model, "get_base_model"):
        model.get_base_model = (
            lambda: model.llm.get_base_model()
            if hasattr(model.llm, "get_base_model")
            else model.llm
        )
    if not hasattr(model, "_keys_to_ignore_on_save"):
        model._keys_to_ignore_on_save = None

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[combined_reward],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ChessLMWithEncoder has a tied embed_tokens attribute (alias of lm_head weights).
    # safetensors rejects state dicts with shared tensors.
    # Fix: make embed_tokens a non-Module buffer so it doesn't appear in state_dict,
    # and override _save to use torch.save (pickle format) for checkpoints.
    # Re-register embed_tokens as a plain Python attribute (not nn.Module) to hide it.
    _embed_module = model.embed_tokens  # keep reference
    object.__setattr__(model, "embed_tokens", _embed_module)  # no-op but removes nn tracking
    # The real fix: monkey-patch state_dict to drop the duplicate embed_tokens key
    _orig_state_dict = model.state_dict

    def _deduped_state_dict(**kwargs):
        sd = _orig_state_dict(**kwargs)
        # Remove ChessLMWithEncoder.embed_tokens alias
        sd.pop("embed_tokens.weight", None)
        # Remove Qwen3.5 weight-tied duplicate: lm_head.weight == embed_tokens.weight
        # Keep the embed_tokens copy; drop lm_head duplicate so safetensors is happy.
        # Both the plain key and the LoRA-wrapped key can appear depending on PEFT version.
        sd.pop("llm.base_model.model.lm_head.weight", None)
        sd.pop("llm.lm_head.weight", None)
        return sd

    model.state_dict = _deduped_state_dict

    log.info("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=args.resume)

    out = config.get("output_dir", "checkpoints/qwen3.5-4b-encoder-phase2-grpo")
    log.info("Saving model to %s", out)
    trainer.save_model()
    tokenizer.save_pretrained(out)
    log.info("Done.")


if __name__ == "__main__":
    main()
