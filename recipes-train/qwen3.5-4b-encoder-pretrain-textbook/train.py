"""Encoder Continued Pretraining on Chess Textbooks

Causal LM loss on annotated grandmaster prose. Each [Position: FEN] in the
source text has been replaced by 64 <|vision_pad|> sentinels; the frozen CNN
encoder injects spatial board embeddings at those positions.

Key differences from phase1-SFT:
  - Labels = input_ids everywhere (causal LM, not assistant-turn-only)
  - Sentinel positions masked to -100 inside ChessLMWithEncoder.forward()
  - Encoder is frozen — no requires_grad_, no gradient back-propagation
  - Sequences up to max_seq_length=2048 (full annotated game sections)
  - Multiple board positions per sequence (n_boards = n_sentinels / 64)
"""

import argparse
import logging
import os
import sys

import chess
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.encoder import BOARD_TOKEN_ID, BOARD_TOKENS_PER_POSITION
from training.encoder_collator import EncoderDataCollator
from training.encoder_model import ChessLMWithEncoder
from training.lib import load_config, make_training_args

_logger = logging.getLogger(__name__)


def setup_model(config_path: str) -> tuple:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(config_path)
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    encoder_cfg = config.get("encoder", {})
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

    _logger.info("Wrapping with CNN Encoder (frozen)...")
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

    # Load encoder weights
    encoder_weights_path = encoder_cfg.get("pretrained_weights")
    if encoder_weights_path:
        state = torch.load(
            encoder_weights_path, map_location=f"cuda:{local_rank}", weights_only=True
        )
        missing, unexpected = model.cnn.load_state_dict(state, strict=True)
        if missing:
            _logger.warning("Missing keys in encoder weights: %s", missing)
        if unexpected:
            _logger.warning("Unexpected keys in encoder weights: %s", unexpected)
        _logger.info("Loaded encoder weights from %s", encoder_weights_path)
    else:
        _logger.warning("No encoder.pretrained_weights set — encoder is randomly initialised!")

    # Encoder is FROZEN — no gradient flows back through the CNN.
    # PEFT already disables all non-LoRA gradients; encoder must also stay frozen.
    for param in model.cnn.parameters():
        param.requires_grad_(False)

    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="recipes-train/qwen3.5-4b-encoder-pretrain-textbook/config.yaml",
    )
    parser.add_argument("--resume", nargs="?", const=True, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})

    if wandb_cfg.get("enabled") and not args.dry_run:
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "chess-tutor"),
            name=wandb_cfg.get("name", "qwen3.5-4b-encoder-pretrain-textbook"),
            tags=wandb_cfg.get("tags", ["encoder-pretrain"]),
        )

    model, tokenizer = setup_model(args.config)
    max_seq_length: int = train_cfg.get("max_seq_length", 2048)

    def _fmt(example: dict) -> dict:
        """Tokenize a textbook chunk; labels = input_ids (causal LM)."""
        text: str = example["text"]
        fens: list[str] = example.get("fens", [])

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            return_attention_mask=True,
        )
        input_ids: list[int] = tokenized["input_ids"]
        attention_mask: list[int] = tokenized["attention_mask"]

        # Causal LM: labels == input_ids. Sentinel positions are masked
        # to -100 inside ChessLMWithEncoder.forward() so no loss is computed on them.
        labels: list[int] = list(input_ids)

        # Use the first FEN for the collator's board tensor construction.
        # The collator's sentinel-counting logic handles multiple boards per sequence.
        primary_fen = fens[0] if fens else chess.STARTING_FEN

        # Validate: every 64 sentinels → one board tensor
        n_sentinels = input_ids.count(BOARD_TOKEN_ID)
        n_boards_expected = n_sentinels // BOARD_TOKENS_PER_POSITION
        if n_boards_expected != len(fens):
            _logger.debug(
                "FEN count mismatch: text has %d sentinel groups, record has %d FENs — collator will pad/trim",
                n_boards_expected,
                len(fens),
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # collator fields — fen triggers board tensor for primary position;
            # collator pads additional boards from fens list via sentinel count
            "fen": primary_fen,
            "move_san": "",
            "line_sans_json": "[]",
            # Store full fens list so collator can build all board tensors
            "_fens_json": __import__("json").dumps(fens),
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
    train_dataset = (
        Dataset.from_list(raw_train)
        .map(_fmt, num_proc=4, remove_columns=["text", "fens", "source"])
        .select_columns(
            [
                "input_ids",
                "attention_mask",
                "labels",
                "fen",
                "move_san",
                "line_sans_json",
                "_fens_json",
            ]
        )
    )

    eval_dataset = None
    if train_cfg.get("eval_file"):
        raw_eval = _load_jsonl(train_cfg["eval_file"], limit=20 if args.dry_run else None)
        if raw_eval:
            eval_dataset = (
                Dataset.from_list(raw_eval)
                .map(_fmt, num_proc=2, remove_columns=["text", "fens", "source"])
                .select_columns(
                    [
                        "input_ids",
                        "attention_mask",
                        "labels",
                        "fen",
                        "move_san",
                        "line_sans_json",
                        "_fens_json",
                    ]
                )
            )

    training_args = make_training_args(config)
    training_args.remove_unused_columns = False
    if eval_dataset is None:
        training_args.eval_strategy = "no"

    # Subclass collator to use the full fens list (not just the primary FEN)
    class TextbookCollator(EncoderDataCollator):
        """EncoderDataCollator extended to build board tensors for all positions in a chunk.

        The base collator only builds one tensor per example (from `fen`). For textbook
        pretraining, each record can have many positions. We override the board-tensor
        building to use `_fens_json` which contains all FENs in sequence order.
        """

        def __call__(self, features):
            import json as _json

            from src.encoder.board_tensor import board_to_tensor

            # Extract fens lists before the base collator sees them
            fens_lists: list[list[str]] = []
            for feat in features:
                raw = feat.pop("_fens_json", "[]")
                try:
                    fens_lists.append(_json.loads(raw))
                except Exception:
                    fens_lists.append([])

            # Let base collator handle padding + primary board tensor scaffolding
            batch = super().__call__(features)

            # Rebuild board_tensors_flat using all FENs in sequence order.
            # Use board_to_tensor (19ch) — no move context in textbook positions,
            # matching the encoder's in_channels=19 pretrain config.
            all_tensors = []
            board_counts = []
            for b_idx, fens in enumerate(fens_lists):
                n_sentinel_tokens = (batch["input_ids"][b_idx] == self._board_token_id).sum().item()
                n_boards = n_sentinel_tokens // BOARD_TOKENS_PER_POSITION

                tensors = []
                for fen in fens[:n_boards]:
                    try:
                        board = chess.Board(fen)
                    except Exception:
                        board = chess.Board()
                    tensors.append(board_to_tensor(board))

                # Pad to n_boards if fens list is shorter
                while len(tensors) < n_boards:
                    tensors.append(board_to_tensor(chess.Board()))

                board_counts.append(len(tensors))
                all_tensors.extend(tensors)

            batch["board_tensors_flat"] = (
                torch.stack(all_tensors) if all_tensors else torch.zeros(0, 19, 8, 8)
            )
            batch["move_counts"] = torch.tensor(board_counts, dtype=torch.long)
            return batch

    data_collator = TextbookCollator(tokenizer=tokenizer)

    from transformers import Trainer

    class EncoderTrainer(Trainer):
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

    trainer = EncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if args.dry_run:
        _logger.info("Dry run complete.")
        return

    _logger.info("Starting textbook continued pretraining...")
    trainer.train(resume_from_checkpoint=args.resume)

    _logger.info("Saving model to %s", training_args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    _logger.info("Done.")


if __name__ == "__main__":
    main()
