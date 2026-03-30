"""Encoder SFT training — Phase 1 Board-Reading Grounding

Each sample has one sentinel token <|vision_pad|> in the user message.
EncoderDataCollator replaces it with CNN board embeddings at training time.
Target: <answer>X</answer> with no thinking block.
"""

import argparse
import json
import logging
import os
import re
import sys

import chess
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.encoder import BOARD_TOKEN, BOARD_TOKEN_ID, BOARD_TOKENS_PER_POSITION
from training.encoder_collator import EncoderDataCollator
from training.encoder_model import ChessLMWithEncoder
from training.lib import (
    load_config,
    load_jsonl_lines,
    make_training_args,
    strip_think_from_target,
)

_logger = logging.getLogger(__name__)


def _verify_sentinel(messages: list[dict]) -> None:
    """Verify exactly BOARD_TOKENS_PER_POSITION sentinel tokens exist in the user message."""
    for msg in messages:
        if msg["role"] == "user":
            count = msg["content"].count(BOARD_TOKEN)
            if count != BOARD_TOKENS_PER_POSITION:
                _logger.warning(
                    "Expected %d board sentinels in user message, got %d",
                    BOARD_TOKENS_PER_POSITION,
                    count,
                )
            return


def setup_encoder_model(config_path: str) -> tuple:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

    actual_id = tokenizer.convert_tokens_to_ids(BOARD_TOKEN)
    if actual_id != BOARD_TOKEN_ID:
        raise RuntimeError(
            f"BOARD_TOKEN {BOARD_TOKEN!r} resolved to id={actual_id}, expected {BOARD_TOKEN_ID}."
        )

    quant_mode = model_cfg.get("quantization", "8bit")
    if quant_mode == "8bit":
        _logger.info("Loading base LLM (8-bit) on GPU %d", local_rank)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_mode == "4bit":
        _logger.info("Loading base LLM (4-bit) on GPU %d", local_rank)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        _logger.info("Loading base LLM (bf16, no quantization) on GPU %d", local_rank)
        bnb_config = None
    base_llm = AutoModelForCausalLM.from_pretrained(
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
    peft_llm = get_peft_model(base_llm, lora_config)

    _logger.info("Wrapping with CNN Encoder...")
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
    else:
        _logger.warning("No encoder.pretrained_weights set!")

    # PEFT get_peft_model() disables all gradients globally except LoRA.
    # We must explicitly re-enable them for the newly attached CNN encoder!
    for param in model.cnn.parameters():
        param.requires_grad_(True)

    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="recipes-train/qwen3.5-4b-encoder-phase1-sft/config.yaml"
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
            name=wandb_cfg.get("name", "qwen3.5-4b-encoder-phase1-sft"),
            tags=wandb_cfg.get("tags", ["encoder-sft"]),
        )

    model, tokenizer = setup_encoder_model(args.config)
    board_token_id: int = BOARD_TOKEN_ID

    keep_think = train_cfg.get("keep_think", True)
    max_seq_length: int = train_cfg.get("max_seq_length", 2400)

    _response_tpl: list[int] = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    _tpl_len = len(_response_tpl)

    def _make_labels(input_ids: list[int]) -> list[int]:
        labels = [-100] * len(input_ids)
        i = 0
        while i <= len(input_ids) - _tpl_len:
            if input_ids[i : i + _tpl_len] == _response_tpl:
                j = i + _tpl_len
                while j < len(input_ids) and input_ids[j] != 151645:
                    j += 1
                if j < len(input_ids):
                    j += 1
                for k in range(i + _tpl_len, j):
                    labels[k] = input_ids[k]
                i = j
            else:
                i += 1
        return labels

    def _fmt(example: dict) -> dict:
        messages = example["messages"]
        if not keep_think:
            messages = strip_think_from_target(messages)

        meta = example.get("metadata", {})
        fen = meta.get("fen", chess.STARTING_FEN)

        # Board-reading SFT: sentinel is already in the user message from the dataset.
        # EncoderDataCollator will replace it with boards_to_tensor(board, None).
        _verify_sentinel(messages)

        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask", [1] * len(input_ids))
        labels = _make_labels(input_ids)

        actual_count = input_ids.count(move_token_id)
        if actual_count != 1:
            _logger.warning(
                "move token count mismatch in _fmt: expected 1 got %d (fen=%s)",
                actual_count,
                fen,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "fen": fen,
            "move_san": "",
            "line_sans_json": "[]",
        }

    from datasets import Dataset

    raw_train = load_jsonl_lines(train_cfg["train_file"])
    if args.dry_run:
        raw_train = raw_train[:100]

    train_dataset = (
        Dataset.from_list(raw_train)
        .map(_fmt, num_proc=1)
        .select_columns(
            ["input_ids", "attention_mask", "labels", "fen", "move_san", "line_sans_json"]
        )
    )

    eval_dataset = None
    if train_cfg.get("eval_file"):
        raw_eval = load_jsonl_lines(train_cfg["eval_file"])
        if args.dry_run:
            raw_eval = raw_eval[:20]
        if raw_eval:
            eval_dataset = (
                Dataset.from_list(raw_eval)
                .map(_fmt, num_proc=1)
                .select_columns(
                    ["input_ids", "attention_mask", "labels", "fen", "move_san", "line_sans_json"]
                )
            )

    training_args = make_training_args(config)
    training_args.remove_unused_columns = False
    if eval_dataset is None:
        training_args.eval_strategy = "no"

    data_collator = EncoderDataCollator(tokenizer=tokenizer)

    from transformers import Trainer

    class EncoderTrainer(Trainer):
        """Trainer subclass that deduplicates shared-memory tensors before saving.

        Qwen3 ties embed_tokens.weight / lm_head.weight to the same storage.
        safetensors refuses to serialize any shared-memory tensors, so we
        collect the full state dict and clone any tensor whose data_ptr has
        already been seen — covering all tied pairs regardless of naming.
        """

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

    _logger.info("Starting encoder integration training...")
    trainer.train(resume_from_checkpoint=args.resume)

    _logger.info("Saving model to %s", training_args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    _logger.info("Done.")


if __name__ == "__main__":
    main()
