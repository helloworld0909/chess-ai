"""Encoder SFT training — Phase 2 Joint Task"""

import argparse
import json
import logging
import os
import re
import sys

import chess
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.encoder import MOVE_TOKEN, MOVE_TOKEN_ID
from training.encoder_collator import EncoderDataCollator
from training.encoder_model import ChessLMWithEncoder
from training.lib import (
    load_config,
    load_jsonl_lines,
    make_training_args,
    strip_think_from_target,
)

_logger = logging.getLogger(__name__)


def _extract_line_sans(user_content: str) -> list[list[str]]:
    """Parse ## Engine Key Lines section from user content.

    Handles both "PLAYED LINE: ..." and "Line N: ..." formats.
    Returns all lines in order (PLAYED LINE first if present).
    """
    lines_section_re = re.search(
        r"## Engine Key Lines\n\n(.*?)(?=\n\n##|\Z)", user_content, re.DOTALL
    )
    if not lines_section_re:
        return []

    result = []
    lines_text = lines_section_re.group(1).strip().split("\n")
    for line in lines_text:
        m = re.match(r"^(?:PLAYED LINE|Line \d+):\s*(.*)", line.strip())
        if m:
            parts = m.group(1).split("→")
            sans = []
            for part in parts:
                clean = part.replace(MOVE_TOKEN, "").strip()
                if clean:
                    sans.append(clean)
            if sans:
                result.append(sans)
    return result


def _inject_move_tokens(
    messages: list[dict],
    student_san: str,
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
                        # Match PLAYED LINE or Line N
                        line_m = re.match(r"^((?:PLAYED LINE|Line \d+):\s*)(.*)", line)
                        if line_m:
                            prefix = line_m.group(1)
                            moves_str = line_m.group(2)
                            moves = moves_str.split("→")
                            injected_moves = []
                            for move in moves:
                                move = move.strip()
                                if not move.startswith(MOVE_TOKEN):
                                    injected_moves.append(MOVE_TOKEN + move)
                                else:
                                    injected_moves.append(move)
                            new_lines.append(prefix + " → ".join(injected_moves))
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

    actual_id = tokenizer.convert_tokens_to_ids(MOVE_TOKEN)
    if actual_id != MOVE_TOKEN_ID:
        raise RuntimeError(
            f"MOVE_TOKEN {MOVE_TOKEN!r} resolved to id={actual_id}, expected {MOVE_TOKEN_ID}."
        )

    _logger.info("Loading base LLM (8-bit) on GPU %d", local_rank)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
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
        cnn_hidden_size=encoder_cfg.get("hidden_size", 512),
        cnn_num_blocks=encoder_cfg.get("num_blocks", 15),
        move_token_id=MOVE_TOKEN_ID,
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
    move_token_id: int = MOVE_TOKEN_ID

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
        student_san = meta.get("move_san", "")

        modified_messages, line_sans = _inject_move_tokens(messages, student_san)

        tokenized = tokenizer.apply_chat_template(
            modified_messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask", [1] * len(input_ids))
        labels = _make_labels(input_ids)

        expected_count = (1 if student_san else 0) + sum(len(ls) for ls in line_sans)
        actual_count = input_ids.count(move_token_id)
        if actual_count != expected_count:
            _logger.warning(
                "move token count mismatch in _fmt: expected %d got %d (fen=%s san=%s)",
                expected_count,
                actual_count,
                fen,
                student_san,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "fen": fen,
            "move_san": student_san,
            "line_sans_json": json.dumps(line_sans),
        }

    from datasets import Dataset

    raw_train = load_jsonl_lines(train_cfg["train_file"])
    if args.dry_run:
        raw_train = raw_train[:100]

    train_dataset = (
        Dataset.from_list(raw_train)
        .map(_fmt, num_proc=4)
        .select_columns(
            ["input_ids", "attention_mask", "labels", "fen", "move_san", "line_sans_json"]
        )
    )

    eval_dataset = None
    if train_cfg.get("eval_file"):
        raw_eval = load_jsonl_lines(train_cfg["eval_file"])
        if args.dry_run:
            raw_eval = raw_eval[:20]
        eval_dataset = (
            Dataset.from_list(raw_eval)
            .map(_fmt, num_proc=4)
            .select_columns(
                ["input_ids", "attention_mask", "labels", "fen", "move_san", "line_sans_json"]
            )
        )

    training_args = make_training_args(config)
    training_args.remove_unused_columns = False

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
