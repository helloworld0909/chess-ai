"""Phase 0: MLP Projector + LoRA Alignment (run 4)

Trains cnn.proj (2-layer MLP + LayerNorm) + LoRA on LLM jointly.
CNN trunk is frozen. LLM base weights are frozen; only LoRA adapters train.

Key fix vs run 2: heavily asymmetric LRs — projector at 1e-3, LoRA at 5e-6 (200× slower).
LoRA can't exploit text coordinate labels fast enough; projector establishes alignment first.
LayerNorm at projector output stabilizes training.

Data format: alignment_board_description.jsonl
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}],
   "metadata": {"fen": ..., "task": ...}}

The user message does NOT contain sentinel tokens.
This script injects the 64-sentinel block into the user content at data-loading time,
randomly placing it before or after the question (50/50) so the model learns to attend
to board tokens at any position. Labels are set so loss is only computed on the assistant answer.
Think block tokens (<think>\n\n</think>\n\n) are masked from loss.
"""

import argparse
import logging
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.encoder import BOARD_TOKEN, BOARD_TOKEN_ID, BOARD_TOKENS_PER_POSITION
from src.encoder.board_tensor import board_to_tensor
from training.encoder_collator import EncoderDataCollator
from training.encoder_model import ChessLMWithEncoder
from training.lib import load_config, make_training_args

_logger = logging.getLogger(__name__)

_SENTINEL_BLOCK = BOARD_TOKEN * BOARD_TOKENS_PER_POSITION

# TODO: This anchored board format MUST be used consistently everywhere the encoder
# is used — SFT data generation, GRPO prompts, and inference (encoder_inference.py).
# Any mismatch (flat sentinels vs anchored) will break spatial grounding.

# Prompt-anchored board: each sentinel explicitly labeled with its square name.
# CNN output is row-major a1=idx0, b1=idx1, ..., h1=idx7, a2=idx8, ..., h8=idx63.
# Display rank 1→8 to match CNN output order exactly:
# text token N gets injected with CNN output token N → a1<tok0> b1<tok1> ... h8<tok63>
_FILES = "abcdefgh"
_ANCHORED_BOARD_LINES = []
for _rank in range(0, 8):  # rank 1→8, matching CNN output order (a1=idx0 ... h8=idx63)
    _cells = []
    for _file in range(8):
        _sq = f"{_FILES[_file]}{_rank + 1}"
        _cells.append(f"{_sq}{BOARD_TOKEN}")
    _ANCHORED_BOARD_LINES.append(" ".join(_cells))
_ANCHORED_BOARD = "\n".join(_ANCHORED_BOARD_LINES)

_SYSTEM_PROMPT = (
    "You are a chess assistant. The board is encoded as a grid of square labels (a1–h8), "
    "each immediately followed by a vision token that encodes the piece occupying that square. "
    "To answer questions, attend to the vision token after each square label to identify the piece — "
    "do not try to parse the square labels as text notation."
)


def setup_model(config_path: str, init_proj_path: str | None = None) -> tuple:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(config_path)
    model_cfg = config.get("model", {})
    encoder_cfg = config.get("encoder", {})
    lora_cfg = config.get("lora", {})
    model_name = model_cfg["model_name"]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    _logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    _logger.info("Loading base LLM (bf16) on GPU %d", local_rank)
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
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
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    peft_llm = get_peft_model(base_llm, lora_config)

    _logger.info("Wrapping with ChessLMWithEncoder...")
    model = ChessLMWithEncoder(
        llm=peft_llm,
        hidden_size=base_llm.config.hidden_size,
        cnn_in_channels=encoder_cfg.get("in_channels", 19),
        cnn_hidden_size=encoder_cfg.get("hidden_size", 256),
        cnn_num_blocks=encoder_cfg.get("num_blocks", 10),
        move_token_id=BOARD_TOKEN_ID,
    )
    model.to(torch.bfloat16)
    model.to(f"cuda:{local_rank}")

    # Load encoder CNN trunk weights
    encoder_weights_path = encoder_cfg.get("pretrained_weights")
    if encoder_weights_path:
        state = torch.load(
            encoder_weights_path, map_location=f"cuda:{local_rank}", weights_only=True
        )
        trunk_state = {k: v for k, v in state.items() if not k.startswith("proj.")}
        missing, unexpected = model.cnn.load_state_dict(trunk_state, strict=False)
        if missing:
            proj_missing = [k for k in missing if k.startswith("proj.")]
            other_missing = [k for k in missing if not k.startswith("proj.")]
            if other_missing:
                _logger.warning("Missing non-proj keys in encoder weights: %s", other_missing)
            _logger.info(
                "Loaded encoder trunk weights; proj keys not loaded (random init): %s",
                proj_missing,
            )
        if unexpected:
            _logger.warning("Unexpected keys in encoder weights: %s", unexpected)
        _logger.info("Loaded encoder trunk from %s", encoder_weights_path)
    else:
        _logger.warning("No encoder.pretrained_weights set — encoder is randomly initialised!")

    # Optionally warm-init projector from a prior phase0 checkpoint
    if init_proj_path:
        import glob as _glob
        import os as _os

        ckpt_file = _os.path.join(init_proj_path, "pytorch_model.bin")
        if not _os.path.exists(ckpt_file):
            shards = sorted(_glob.glob(_os.path.join(init_proj_path, "model*.safetensors")))
            if shards:
                from safetensors.torch import load_file as _load_safetensors

                merged = {}
                for s in shards:
                    merged.update(_load_safetensors(s, device=f"cuda:{local_rank}"))
                proj_state = {
                    k.removeprefix("cnn."): v
                    for k, v in merged.items()
                    if k.startswith("cnn.proj.")
                }
            else:
                raise FileNotFoundError(f"No checkpoint found in {init_proj_path}")
        else:
            ckpt = torch.load(ckpt_file, map_location=f"cuda:{local_rank}", weights_only=True)
            proj_state = {
                k.removeprefix("cnn."): v for k, v in ckpt.items() if k.startswith("cnn.proj.")
            }
        missing, unexpected = model.cnn.load_state_dict(proj_state, strict=False)
        other_missing = [k for k in missing if k.startswith("proj.")]
        if other_missing:
            _logger.warning("Missing proj keys from init checkpoint: %s", other_missing)
        _logger.info("Loaded cnn.proj weights from %s (warm init)", init_proj_path)

    # Freeze CNN trunk. LoRA adapters + cnn.proj are trainable.
    for param in model.cnn.parameters():
        param.requires_grad_(False)
    for param in model.cnn.proj.parameters():
        param.requires_grad_(True)

    model.print_trainable_parameters()

    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="recipes-train/qwen3.5-4b-encoder-phase0-alignment/config.yaml",
    )
    parser.add_argument("--resume", nargs="?", const=True, default=None)
    parser.add_argument(
        "--init-proj",
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Load cnn.proj weights from a prior phase0 checkpoint to continue training on new data. "
        "Unlike --resume, this starts a fresh optimizer/scheduler (new dataset, new run).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})

    if wandb_cfg.get("enabled") and not args.dry_run:
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "chess-tutor"),
            name=wandb_cfg.get("name", "qwen3.5-4b-encoder-phase0-alignment"),
            tags=wandb_cfg.get("tags", ["alignment"]),
        )

    model, tokenizer = setup_model(args.config, init_proj_path=args.init_proj)
    max_seq_length: int = train_cfg.get("max_seq_length", 512)

    def _fmt(example: dict) -> dict:
        """Tokenize a board-reading Q&A example for projector alignment.

        Injects 64 sentinel tokens into the user message. Position is randomized
        (50/50 before or after the question) so the model learns to attend to
        the board tokens regardless of where they appear in context.
        Labels are set so loss is computed only on assistant answer tokens.
        """
        messages = example["messages"]
        fen: str = example.get("metadata", {}).get("fen", "")

        board_before = random.random() < 0.5

        patched_messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for msg in messages:
            if msg["role"] == "user":
                q = msg["content"]
                if board_before:
                    user_content = _ANCHORED_BOARD + "\n\n" + q
                else:
                    user_content = q + "\n\n" + _ANCHORED_BOARD
                patched_messages.append({"role": "user", "content": user_content})
            else:
                patched_messages.append(msg)

        # Apply chat template: full conversation including assistant turn
        full_text = tokenizer.apply_chat_template(
            patched_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Prompt only (no assistant turn) — used to find where answer starts
        prompt_messages = [m for m in patched_messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            return_attention_mask=True,
        )
        prompt_ids = tokenizer(
            prompt_text,
            truncation=False,
            return_attention_mask=False,
        )

        input_ids = full_ids["input_ids"]
        attention_mask = full_ids["attention_mask"]
        prompt_len = len(prompt_ids["input_ids"])

        # Loss only on assistant answer tokens.
        # Also mask the empty <think>\n\n</think>\n\n block Qwen3 always emits —
        # we don't want the model rewarded for predicting boilerplate think tokens.
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))
        labels = labels[: len(input_ids)]

        # Qwen3 chat template with add_generation_prompt=True appends "<think>\n" to the prompt.
        # So the label region starts with: </think>(248069) \n\n(271) <answer>...
        # Mask </think> and \n\n so loss is computed only on actual answer tokens.
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        i = prompt_len
        if i < len(labels) and input_ids[i] == think_end_id:  # </think>
            labels[i] = -100
            i += 1
        if i < len(labels) and input_ids[i] == 271:  # \n\n
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "fen": fen if fen else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "move_san": "",
            "line_sans_json": "[]",
        }

    import json

    from datasets import Dataset

    def _load_jsonl(path: str, limit: int | None = None) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                    if limit and len(records) >= limit:
                        break
        _logger.info("Loaded %d records from %s", len(records), path)
        return records

    raw_train = _load_jsonl(train_cfg["train_file"], limit=100 if args.dry_run else None)
    raw_eval = _load_jsonl(train_cfg["eval_file"], limit=20 if args.dry_run else 2000)

    num_workers = train_cfg.get("dataloader_num_workers", 4)
    remove_cols = ["messages", "metadata"]
    train_dataset = Dataset.from_list(raw_train).map(
        _fmt, num_proc=num_workers, remove_columns=remove_cols
    )
    eval_dataset = Dataset.from_list(raw_eval).map(
        _fmt, num_proc=num_workers, remove_columns=remove_cols
    )

    training_args = make_training_args(config)
    training_args.remove_unused_columns = False

    # Single-board collator — each example has exactly one board (fen field)
    class AlignmentCollator(EncoderDataCollator):
        """Collator that builds a single board_to_tensor per example from the fen field."""

        def __call__(self, features):
            import chess

            batch = super().__call__(features)

            # Rebuild board_tensors_flat using the fen field (one board per example)
            tensors = []
            for feat in features:
                fen = feat.get("fen", chess.STARTING_FEN)
                try:
                    board = chess.Board(fen)
                except Exception:
                    board = chess.Board()
                tensors.append(board_to_tensor(board))

            if tensors:
                batch["board_tensors_flat"] = torch.stack(tensors).to(torch.bfloat16)
            return batch

    collator = AlignmentCollator(tokenizer=tokenizer)

    from transformers import Trainer

    lora_lr = train_cfg.get("lora_learning_rate", 5e-6)

    class EncoderTrainer(Trainer):
        def create_optimizer(self):
            """Two param groups: projector at 1e-3, LoRA at 5e-6 (200× slower)."""
            proj_params = list(self.model.cnn.proj.parameters())
            proj_ids = {id(p) for p in proj_params}
            lora_params = [
                p for p in self.model.parameters() if p.requires_grad and id(p) not in proj_ids
            ]
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": proj_params, "lr": self.args.learning_rate},
                    {"params": lora_params, "lr": lora_lr},
                ],
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer

        def _save(self, output_dir=None, state_dict=None):
            if state_dict is None:
                state_dict = self.model.state_dict()
            seen: dict[int, torch.Tensor] = {}
            deduped: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                ptr = v.data_ptr()
                if ptr in seen:
                    deduped[k] = v.detach().clone()
                else:
                    seen[ptr] = v
                    deduped[k] = v
            super()._save(output_dir=output_dir, state_dict=deduped)

    # Pick one fixed example per task type for eval logging
    _log_samples: list[dict] = []
    seen_tasks: set[str] = set()
    for ex in raw_eval:
        task = ex.get("metadata", {}).get("task", "")
        if task and task not in seen_tasks:
            seen_tasks.add(task)
            _log_samples.append(ex)

    from transformers import TrainerCallback

    class SampleLogCallback(TrainerCallback):
        """Log model predictions on fixed eval samples every eval_steps."""

        def on_evaluate(self, args, state, control, **kwargs):
            if not state.is_local_process_zero:
                return
            import chess as _chess

            model.eval()
            device = model.llm.device
            _logger.info("=== Sample predictions at step %d ===", state.global_step)
            for ex in _log_samples:
                fen = ex.get("metadata", {}).get("fen", _chess.STARTING_FEN)
                task = ex.get("metadata", {}).get("task", "")
                q = next(m["content"] for m in ex["messages"] if m["role"] == "user")
                expected = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
                try:
                    board = _chess.Board(fen)
                except Exception:
                    board = _chess.Board()
                board_tensor = board_to_tensor(board).unsqueeze(0).to(torch.bfloat16).to(device)

                results = {}
                for think in (False, True):
                    prompt_text = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": _ANCHORED_BOARD + "\n\n" + q},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=think,
                    )
                    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
                    with torch.no_grad():
                        cnn_tokens = model.cnn(board_tensor)
                        text_embeds = model.embed_tokens(input_ids)
                        sentinel_mask = (input_ids == model.move_token_id)[0]
                        text_embeds[0, sentinel_mask] = cnn_tokens[0, : sentinel_mask.sum()]
                        out = model.llm.generate(
                            inputs_embeds=text_embeds,
                            max_new_tokens=64,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    results[think] = tokenizer.decode(out[0], skip_special_tokens=False).strip()

                _logger.info(
                    "  [%-20s] Q: %-45s | expected: %-12r | no-think: %-15r | think: %r",
                    task,
                    q[:45],
                    expected,
                    results[False],
                    results[True],
                )
            model.train()

    trainer = EncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[SampleLogCallback()],
    )

    if args.dry_run:
        _logger.info("Dry run complete — exiting before training.")
        return

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    _logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
