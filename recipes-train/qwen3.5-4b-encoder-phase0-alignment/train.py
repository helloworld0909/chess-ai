"""Phase 0: MLP Projector + LoRA Alignment

Trains cnn.proj + cnn.global_proj (2-layer MLPs + LayerNorm) + LoRA on LLM jointly.
CNN trunk is frozen. LLM base weights are frozen; only LoRA adapters train.

Asymmetric LRs: projector at 1e-3, LoRA at 5e-6 (200× slower).
The projector establishes alignment first; LoRA refines the LLM's reading.

Board is injected as a flat block of 65 sentinel tokens (64 per-square + 1 global).
No square-label text anchors — the CNN (CLIP-trained) carries full semantic meaning.

Data format: alignment_board_description.jsonl
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}],
   "metadata": {"fen": ..., "task": ...}}

The user message does NOT contain sentinel tokens.
This script injects the 65-sentinel block into the user content at data-loading time,
randomly placing it before or after the question (50/50).
Labels are set so loss is only computed on the assistant answer.
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

# Flat block of 65 sentinel tokens representing one board position.
# 64 per-square tokens (row-major a1..h8) + 1 global summary token.
# No square-label text anchors — the CLIP-trained CNN encodes full semantics.
_BOARD_BLOCK = BOARD_TOKEN * BOARD_TOKENS_PER_POSITION

_SYSTEM_PROMPT = (
    "You are a chess assistant. The board position is encoded as a sequence of vision tokens. "
    "Use them to identify pieces and answer questions about the position."
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
        missing, unexpected = model.cnn.load_state_dict(state, strict=False)
        proj_missing = [k for k in missing if k.startswith("proj.") or k.startswith("global_proj.")]
        other_missing = [k for k in missing if k not in proj_missing]
        if other_missing:
            _logger.warning("Missing trunk keys in encoder weights: %s", other_missing)
        if unexpected:
            _logger.warning("Unexpected keys in encoder weights: %s", unexpected)
        if proj_missing:
            _logger.info(
                "Loaded encoder trunk from %s; proj/global_proj randomly initialized",
                encoder_weights_path,
            )
        else:
            _logger.info("Loaded full encoder (trunk + proj) from %s", encoder_weights_path)
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
                    if k.startswith("cnn.proj.") or k.startswith("cnn.global_proj.")
                }
            else:
                raise FileNotFoundError(f"No checkpoint found in {init_proj_path}")
        else:
            ckpt = torch.load(ckpt_file, map_location=f"cuda:{local_rank}", weights_only=True)
            proj_state = {
                k.removeprefix("cnn."): v
                for k, v in ckpt.items()
                if k.startswith("cnn.proj.") or k.startswith("cnn.global_proj.")
            }
        missing, unexpected = model.cnn.load_state_dict(proj_state, strict=False)
        other_missing = [
            k for k in missing if not k.startswith("proj.") and not k.startswith("global_proj.")
        ]
        if other_missing:
            _logger.warning("Missing proj keys from init checkpoint: %s", other_missing)
        _logger.info(
            "Loaded cnn.proj + cnn.global_proj weights from %s (warm init)", init_proj_path
        )

    # Freeze CNN trunk. LoRA adapters + cnn.proj + cnn.global_proj are trainable.
    for param in model.cnn.parameters():
        param.requires_grad_(False)
    for param in model.cnn.proj.parameters():
        param.requires_grad_(True)
    for param in model.cnn.global_proj.parameters():
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

        Injects 65 sentinel tokens into the user message. Position is randomized
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
                    user_content = _BOARD_BLOCK + "\n\n" + q
                else:
                    user_content = q + "\n\n" + _BOARD_BLOCK
                patched_messages.append({"role": "user", "content": user_content})
            else:
                patched_messages.append(msg)

        # Build prompt without assistant turn
        prompt_messages = [m for m in patched_messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Qwen3 with add_generation_prompt=True appends "<think>\n" to the prompt.
        # Our data has no think blocks, so full_text won't contain <think>\n.
        # Strip it from prompt_text so prompt_len matches the actual boundary in full_ids.
        if prompt_text.endswith("<think>\n"):
            prompt_text = prompt_text[:-8]

        # Extract assistant answer and manually assemble full sequence.
        # This keeps prompt boundary exactly aligned between prompt_text and full_text.
        assistant_content = next(m["content"] for m in patched_messages if m["role"] == "assistant")
        full_text = prompt_text + assistant_content + "<|im_end|>\n"

        full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_length)["input_ids"]
        prompt_ids = tokenizer(prompt_text, truncation=False)["input_ids"]

        prompt_len = len(prompt_ids)

        # Verify the prompt prefix is byte-for-byte identical in both tokenizations.
        # If full_ids[:prompt_len] != prompt_ids, the label boundary is wrong.
        # This can happen if there's a tokenization boundary effect at the join point.
        if full_ids[:prompt_len] != prompt_ids:
            raise ValueError(
                f"Tokenization boundary mismatch: prompt_ids and full_ids diverge at the join. "
                f"prompt_ids[-5:]={prompt_ids[-5:]}, full_ids[prompt_len-5:prompt_len+5]="
                f"{full_ids[max(0, prompt_len - 5) : prompt_len + 5]}"
            )

        # Loss only on assistant answer tokens.
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        if len(labels) > len(full_ids):
            labels = labels[: len(full_ids)]
        elif len(labels) < len(full_ids):
            labels += [-100] * (len(full_ids) - len(labels))

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "fen": fen,
            "move_san": "",
            "line_sans_json": "[]",
        }

    import itertools
    import json

    from datasets import Dataset

    class AlignmentJsonlDataset(torch.utils.data.IterableDataset):
        """Streaming JSONL dataset with manual DDP + DataLoader worker sharding.

        Each DDP rank × DataLoader worker reads its own stride of lines, so no
        dispatch_batches coordination is needed. _fmt is applied lazily per record.
        Mirrors the ChessClipJsonlDataset pattern from encoder-clip/train.py.
        """

        def __init__(self, path: str, limit: int = 0) -> None:
            self.path = path
            self.limit = limit
            self._rank = 0
            self._world_size = 1

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            num_workers = worker_info.num_workers if worker_info else 1

            global_stride = num_workers * self._world_size
            my_slot = self._rank * num_workers + worker_id

            emitted = 0
            with open(self.path, "rb") as f:
                for raw in itertools.islice(f, my_slot, None, global_stride):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        example = json.loads(raw)
                        yield _fmt(example)
                        emitted += 1
                        if self.limit and emitted * global_stride >= self.limit:
                            break
                    except Exception:
                        continue

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

    num_workers = train_cfg.get("dataloader_num_workers", 4)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.dry_run:
        remove_cols = ["messages", "metadata"]
        raw_train = _load_jsonl(train_cfg["train_file"], limit=100)
        train_dataset = Dataset.from_list(raw_train).map(
            _fmt, num_proc=1, remove_columns=remove_cols
        )
    else:
        train_dataset = AlignmentJsonlDataset(
            train_cfg["train_file"],
            limit=100 if args.dry_run else 0,
        )
        train_dataset._rank = local_rank
        train_dataset._world_size = world_size

        # Trainer requires max_steps for IterableDataset (no __len__).
        train_size = sum(1 for line in open(train_cfg["train_file"]) if line.strip())
        effective_batch = (
            train_cfg.get("per_device_train_batch_size", 8)
            * world_size
            * train_cfg.get("gradient_accumulation_steps", 8)
        )
        epochs = train_cfg.get("num_train_epochs", 1)
        _logger.info(
            "Streaming dataset: %d examples, effective_batch=%d → max_steps=%d",
            train_size,
            effective_batch,
            train_size * epochs // effective_batch,
        )

    raw_eval = _load_jsonl(train_cfg["eval_file"], limit=20 if args.dry_run else 2000)
    remove_cols = ["messages", "metadata"]
    eval_dataset = Dataset.from_list(raw_eval).map(
        _fmt, num_proc=min(num_workers, 4), remove_columns=remove_cols
    )

    training_args = make_training_args(config)
    training_args.remove_unused_columns = False
    if not args.dry_run:
        training_args.max_steps = train_size * epochs // effective_batch

    # EncoderDataCollator already builds board_tensors_flat from the fen field correctly.
    # A subclass that calls super() then re-reads feat["fen"] is wrong — super() pops
    # fen from each feature dict, so the second pass gets chess.STARTING_FEN for all.
    collator = EncoderDataCollator(tokenizer=tokenizer)

    from transformers import Trainer

    lora_lr = train_cfg.get("lora_learning_rate", 5e-6)

    class EncoderTrainer(Trainer):
        def create_optimizer(self):
            """Two param groups: proj+global_proj at learning_rate (1e-4), LoRA at lora_lr (5e-6)."""
            proj_params = list(self.model.cnn.proj.parameters()) + list(
                self.model.cnn.global_proj.parameters()
            )
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
                            {"role": "user", "content": _BOARD_BLOCK + "\n\n" + q},
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
