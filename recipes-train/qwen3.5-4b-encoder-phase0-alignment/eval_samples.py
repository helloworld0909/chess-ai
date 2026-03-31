"""Quick per-task eval: run one example per task type and print model output.

Usage:
    PYTHONPATH=src:. python recipes-train/qwen3.5-4b-encoder-phase0-alignment/eval_samples.py \
        --checkpoint checkpoints/qwen3.5-4b-encoder-phase0-alignment/checkpoint-1000
"""

import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.encoder import BOARD_TOKEN, BOARD_TOKEN_ID, BOARD_TOKENS_PER_POSITION
from src.encoder.board_tensor import board_to_tensor
from training.encoder_model import ChessLMWithEncoder
from training.lib import load_config

_logger = logging.getLogger(__name__)

_FILES = "abcdefgh"
_ANCHORED_BOARD_LINES = []
for _rank in range(0, 8):
    _cells = []
    for _file in range(8):
        _sq = f"{_FILES[_file]}{_rank + 1}"
        _cells.append(f"{_sq}{BOARD_TOKEN}")
    _ANCHORED_BOARD_LINES.append(" ".join(_cells))
_ANCHORED_BOARD = "\n".join(_ANCHORED_BOARD_LINES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/qwen3.5-4b-encoder-phase0-alignment/checkpoint-1000",
    )
    parser.add_argument(
        "--config",
        default="recipes-train/qwen3.5-4b-encoder-phase0-alignment/config.yaml",
    )
    parser.add_argument(
        "--eval-file",
        default="data/processed/alignment_board_description_eval.jsonl",
    )
    parser.add_argument("--samples-per-task", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    encoder_cfg = config.get("encoder", {})
    model_name = model_cfg["model_name"]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _logger.info("Loading base LLM...")
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    model = ChessLMWithEncoder(
        llm=base_llm,
        hidden_size=base_llm.config.hidden_size,
        cnn_in_channels=encoder_cfg.get("in_channels", 19),
        cnn_hidden_size=encoder_cfg.get("hidden_size", 256),
        cnn_num_blocks=encoder_cfg.get("num_blocks", 10),
        move_token_id=BOARD_TOKEN_ID,
    )
    model.to(torch.bfloat16).to("cuda:0")

    # Load checkpoint projector weights
    _logger.info("Loading checkpoint from %s...", args.checkpoint)
    import glob

    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(args.checkpoint, "model*.safetensors")))
    if shards:
        merged = {}
        for s in shards:
            merged.update(load_file(s, device="cuda:0"))
        proj_state = {
            k.removeprefix("cnn."): v for k, v in merged.items() if k.startswith("cnn.proj.")
        }
        model.cnn.load_state_dict(proj_state, strict=False)
        _logger.info("Loaded projector from safetensors shards.")
    else:
        ckpt = torch.load(
            os.path.join(args.checkpoint, "pytorch_model.bin"),
            map_location="cuda:0",
            weights_only=True,
        )
        proj_state = {
            k.removeprefix("cnn."): v for k, v in ckpt.items() if k.startswith("cnn.proj.")
        }
        model.cnn.load_state_dict(proj_state, strict=False)

    # Load CNN trunk
    trunk_path = encoder_cfg.get("pretrained_weights")
    if trunk_path:
        state = torch.load(trunk_path, map_location="cuda:0", weights_only=True)
        trunk_state = {k: v for k, v in state.items() if not k.startswith("proj.")}
        model.cnn.load_state_dict(trunk_state, strict=False)

    model.eval()

    # Load eval examples, one per task
    _logger.info("Loading eval examples...")
    from collections import defaultdict

    by_task: dict[str, list[dict]] = defaultdict(list)
    with open(args.eval_file) as f:
        for line in f:
            d = json.loads(line.strip())
            task = d.get("metadata", {}).get("task", "")
            if len(by_task[task]) < args.samples_per_task:
                by_task[task].append(d)

    import chess

    print(f"\n{'=' * 80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'=' * 80}\n")

    for task in sorted(by_task.keys()):
        print(f"── {task} ──")
        for ex in by_task[task]:
            fen = ex.get("metadata", {}).get("fen", chess.STARTING_FEN)
            q = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            expected = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": _ANCHORED_BOARD + "\n\n" + q}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to("cuda:0")

            try:
                board = chess.Board(fen)
            except Exception:
                board = chess.Board()
            board_tensor = board_to_tensor(board).unsqueeze(0).to(torch.bfloat16).to("cuda:0")

            with torch.no_grad():
                cnn_tokens = model.cnn(board_tensor)  # (1, 64, hidden)
                text_embeds = model.embed_tokens(input_ids)  # (1, L, hidden)
                sentinel_mask = (input_ids == model.move_token_id)[0]
                n_sentinels = sentinel_mask.sum().item()
                text_embeds[0, sentinel_mask] = cnn_tokens[0, :n_sentinels]
                out = model.llm.generate(
                    inputs_embeds=text_embeds,
                    max_new_tokens=48,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            correct = "✓" if expected.lower() in answer.lower() else "✗"
            print(f"  Q: {q[:70]}")
            print(f"  expected: {expected!r:30s}  got: {answer!r}  {correct}")
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
