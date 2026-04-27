"""Phase 1b: SFT Cold-start on Lichess puzzle traces.

Trains on puzzle_sft_coldstart.jsonl — each record has:
  messages: [system, user (<|vision_pad|> + puzzle prompt), assistant (<think>...</think>\n{"move": "..."})]
  metadata: {fen, solution_uci, themes, color, rating}

Trains cnn.proj + cnn.global_proj + LoRA. CNN trunk frozen.
Loss only on assistant tokens (system + user masked).
Think block tokens masked from loss (model learns to produce them but not optimized on them).
"""

import argparse
import json
import logging
import os
import sys

import chess
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.encoder import BOARD_TOKEN_ID, BOARD_TOKENS_PER_POSITION
from src.encoder.board_tensor import board_to_tensor
from src.model.encoder_model import ChessLMWithEncoder
from src.model.lib import load_config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PuzzleSFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_length: int):
        self.records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        log.info("Loaded %d records from %s", len(self.records), path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class PuzzleSFTCollator:
    """Tokenize messages, mask prompt tokens from loss, inject board tensors."""

    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: list[dict]) -> dict:
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_board_tensors = []

        for record in batch:
            messages = record["messages"]
            fen = record["metadata"]["fen"]

            # Build full text with generation prompt disabled (we supply assistant turn)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )

            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length,
            ).input_ids

            # Expand board sentinel tokens (1 → BOARD_TOKENS_PER_POSITION)
            expanded = []
            for tok in ids:
                if tok == BOARD_TOKEN_ID:
                    expanded.extend([BOARD_TOKEN_ID] * BOARD_TOKENS_PER_POSITION)
                else:
                    expanded.append(tok)
            ids = expanded[: self.max_seq_length]

            # Build labels: mask everything before assistant turn
            # Find the last occurrence of the assistant start marker
            labels = self._build_labels(ids, messages)

            all_input_ids.append(ids)
            all_labels.append(labels)
            all_attention_mask.append([1] * len(ids))
            all_board_tensors.append(board_to_tensor(chess.Board(fen)))

        # Pad to max length in batch
        max_len = max(len(x) for x in all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = torch.tensor(
            [x + [pad_id] * (max_len - len(x)) for x in all_input_ids], dtype=torch.long
        )
        labels = torch.tensor(
            [x + [-100] * (max_len - len(x)) for x in all_labels], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [x + [0] * (max_len - len(x)) for x in all_attention_mask], dtype=torch.long
        )
        board_tensors = torch.stack(all_board_tensors)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "board_tensors": board_tensors,
        }

    def _build_labels(self, ids: list[int], messages: list[dict]) -> list[int]:
        """Mask system + user tokens; compute loss on assistant turn only.

        Assistant turn is purely the piece list (no thinking block, no move).
        Loss on piece list tokens trains CNN embeddings → piece identity mapping.
        """
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        expanded_prompt = []
        for tok in prompt_ids:
            if tok == BOARD_TOKEN_ID:
                expanded_prompt.extend([BOARD_TOKEN_ID] * BOARD_TOKENS_PER_POSITION)
            else:
                expanded_prompt.append(tok)
        prompt_len = len(expanded_prompt)
        return [-100] * prompt_len + ids[prompt_len:]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PuzzleSFTTrainer:
    """Minimal SFT trainer wrapping HuggingFace Trainer."""

    def __init__(self, config_path: str, resume: bool = False):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, Trainer, TrainingArguments

        cfg = load_config(config_path)
        model_cfg = cfg["model"]
        enc_cfg = cfg["encoder"]
        lora_cfg = cfg["lora"]
        train_cfg = cfg["training"]
        output_dir = cfg["output_dir"]

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["model_name"], trust_remote_code=True
        )

        # Load base LLM
        from transformers import AutoModelForCausalLM
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        base_llm = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_name"],
            dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
            trust_remote_code=True,
            device_map={"": local_rank},
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        )

        # Apply LoRA to LLM before wrapping
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            task_type="CAUSAL_LM",
        )
        peft_llm = get_peft_model(base_llm, lora_config)

        # Wrap with chess encoder
        self.model = ChessLMWithEncoder(
            llm=peft_llm,
            hidden_size=base_llm.config.hidden_size,
            cnn_in_channels=enc_cfg["in_channels"],
            cnn_hidden_size=enc_cfg["hidden_size"],
            cnn_num_blocks=enc_cfg["num_blocks"],
            move_token_id=BOARD_TOKEN_ID,
        )
        self.model.to(torch.bfloat16)

        # Load pretrained encoder weights
        enc_weights_path = enc_cfg["pretrained_weights"]
        if os.path.exists(enc_weights_path):
            state = torch.load(enc_weights_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.cnn.load_state_dict(state, strict=False)
            log.info("Loaded encoder weights from %s (missing=%d unexpected=%d)",
                     enc_weights_path, len(missing), len(unexpected))
        else:
            log.warning("Encoder weights not found: %s", enc_weights_path)

        # Freeze CNN trunk, keep proj trainable
        for name, param in self.model.cnn.named_parameters():
            if "proj" not in name and "global_proj" not in name:
                param.requires_grad_(False)

        self.model.print_trainable_parameters()

        # Dataset & collator
        dataset = PuzzleSFTDataset(
            train_cfg["train_file"],
            self.tokenizer,
            train_cfg["max_seq_length"],
        )
        eval_dataset = None
        if train_cfg.get("eval_file"):
            eval_dataset = PuzzleSFTDataset(
                train_cfg["eval_file"],
                self.tokenizer,
                train_cfg["max_seq_length"],
            )
        collator = PuzzleSFTCollator(self.tokenizer, train_cfg["max_seq_length"])

        # Training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            num_train_epochs=train_cfg["num_train_epochs"],
            max_steps=train_cfg.get("max_steps", -1),
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_steps=train_cfg["warmup_steps"],
            weight_decay=train_cfg["weight_decay"],
            max_grad_norm=train_cfg["max_grad_norm"],
            optim=train_cfg["optim"],
            bf16=train_cfg["bf16"],
            gradient_checkpointing=train_cfg["gradient_checkpointing"],
            dataloader_num_workers=train_cfg["dataloader_num_workers"],
            logging_steps=train_cfg["logging_steps"],
            eval_strategy=train_cfg["eval_strategy"],
            save_strategy=train_cfg["save_strategy"],
            save_steps=train_cfg["save_steps"],
            save_total_limit=train_cfg["save_total_limit"],
            seed=train_cfg["seed"],
            remove_unused_columns=False,
            report_to="none",
        )

        # Two-param-group optimizer: proj at higher LR, LoRA at lower LR
        lora_lr = train_cfg.get("lora_learning_rate", train_cfg["learning_rate"])

        class EncoderSFTTrainer(Trainer):
            def save_model(self_t, output_dir=None, _internal_call=False):
                # Qwen ties embed_tokens.weight and lm_head.weight. safetensors refuses to
                # serialize shared tensors. Untie before save and retie after.
                # self.model = ChessLMWithEncoder; .llm = PeftModel wrapping base LLM
                m = self_t.model
                m = m.module if hasattr(m, "module") else m  # unwrap DDP
                llm = m.llm  # PeftModel
                tied = getattr(llm.config, "tie_word_embeddings", False)
                if tied:
                    llm.lm_head.weight = torch.nn.Parameter(llm.lm_head.weight.detach().clone())
                super().save_model(output_dir, _internal_call)
                if tied:
                    llm.tie_weights()

            def create_optimizer(self_t):
                proj_params = [p for n, p in self.model.named_parameters()
                               if p.requires_grad and ("proj" in n or "global_proj" in n)]
                lora_params = [p for n, p in self.model.named_parameters()
                               if p.requires_grad and ("proj" not in n and "global_proj" not in n)]
                self_t.optimizer = torch.optim.AdamW([
                    {"params": proj_params, "lr": train_cfg["learning_rate"]},
                    {"params": lora_params, "lr": lora_lr},
                ], weight_decay=train_cfg["weight_decay"])
                return self_t.optimizer

            def compute_loss(self_t, model, inputs, return_outputs=False, **kwargs):
                board_tensors = inputs.pop("board_tensors", None)
                if board_tensors is not None:
                    # Unwrap DDP → PeftModel → ChessLMWithEncoder
                    m = model.module if hasattr(model, "module") else model
                    m = m.base_model if hasattr(m, "base_model") else m
                    device = next(m.cnn.parameters()).device
                    m._board_tensors_flat = board_tensors.to(device)
                outputs = model(**inputs)
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

        self.trainer = EncoderSFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )
        self.resume = resume
        self.output_dir = output_dir

    def train(self):
        checkpoint = None
        if self.resume:
            import glob as _glob
            ckpts = sorted(_glob.glob(os.path.join(self.output_dir, "checkpoint-*")))
            if ckpts:
                checkpoint = ckpts[-1]
                log.info("Resuming from %s", checkpoint)
        self.trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer.save_model()
        log.info("Training complete. Model saved to %s", self.output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    trainer = PuzzleSFTTrainer(args.config, resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
