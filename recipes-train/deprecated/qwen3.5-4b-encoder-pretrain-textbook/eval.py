"""Probe evaluation for the textbook continued pretraining checkpoint.

Verifies the model uses CNN board embeddings by running board-reading tasks
with real board tokens vs zeroed board tokens. A model that genuinely reads
the board will perform significantly better with real tokens.

Usage:
    # Evaluate latest textbook pretrain checkpoint
    PYTHONPATH=. python recipes-train/qwen3.5-4b-encoder-pretrain-textbook/eval.py

    # Evaluate a specific checkpoint
    PYTHONPATH=. python recipes-train/qwen3.5-4b-encoder-pretrain-textbook/eval.py \
        --checkpoint checkpoints/qwen3.5-4b-encoder-pretrain-textbook/checkpoint-100

    # Compare against base Qwen (no encoder) as baseline
    PYTHONPATH=. python recipes-train/qwen3.5-4b-encoder-pretrain-textbook/eval.py --base-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.encoder import BOARD_TOKEN, BOARD_TOKEN_ID, BOARD_TOKENS_PER_POSITION
from src.encoder.board_tensor import board_to_tensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_SENTINEL_BLOCK = BOARD_TOKEN * BOARD_TOKENS_PER_POSITION

# ---------------------------------------------------------------------------
# Probe tasks — each has a FEN, a question, and an expected answer.
# Chosen to be unambiguous and require reading the board.
# ---------------------------------------------------------------------------
_PROBES = [
    # piece_at
    {
        "task": "piece_at",
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "question": "What piece is on e4?",
        "answer": "white pawn",
    },
    {
        "task": "piece_at",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "question": "What piece is on c4?",
        "answer": "white bishop",
    },
    {
        "task": "piece_at",
        "fen": "8/8/8/8/3Q4/8/8/4K2k w - - 0 1",
        "question": "What piece is on d4?",
        "answer": "white queen",
    },
    {
        "task": "piece_at",
        "fen": "8/8/8/8/3Q4/8/8/4K2k w - - 0 1",
        "question": "What piece is on h1?",
        "answer": "black king",
    },
    # in_check
    {
        "task": "in_check",
        "fen": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "question": "Is the side to move in check?",
        "answer": "yes",
    },
    {
        "task": "in_check",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "question": "Is the side to move in check?",
        "answer": "no",
    },
    # material_count
    {
        "task": "material",
        "fen": "8/8/8/8/8/8/8/R3K2k w Q - 0 1",
        "question": "Count the total material for each side. Use pawns=1, knights=3, bishops=3, rooks=5, queens=9.",
        "answer": "white: 5\nblack: 0",
    },
    {
        "task": "material",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "question": "Count the total material for each side. Use pawns=1, knights=3, bishops=3, rooks=5, queens=9.",
        "answer": "white: 39\nblack: 39",
    },
]


def _load_model(checkpoint_dir: str, config_path: str, device: str):
    """Load ChessLMWithEncoder from a checkpoint directory.

    EncoderTrainer._save stores the full ChessLMWithEncoder state dict in
    model.safetensors (not a PEFT adapter directory). We rebuild the model
    architecture, then load the merged state dict.
    """
    import yaml
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.model.encoder_model import ChessLMWithEncoder

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    encoder_cfg = cfg.get("encoder", {})
    lora_cfg = cfg.get("lora", {})
    model_cfg = cfg.get("model", {})
    model_name = model_cfg["model_name"]

    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info("Loading base LLM (bf16)...")
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
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

    model = ChessLMWithEncoder(
        llm=peft_llm,
        hidden_size=base_llm.config.hidden_size,
        cnn_in_channels=encoder_cfg.get("in_channels", 19),
        cnn_hidden_size=encoder_cfg.get("hidden_size", 256),
        cnn_num_blocks=encoder_cfg.get("num_blocks", 10),
        move_token_id=BOARD_TOKEN_ID,
    ).to(torch.bfloat16)

    # Load full state dict saved by EncoderTrainer._save
    safetensors_path = Path(checkpoint_dir) / "model.safetensors"
    if safetensors_path.exists():
        log.info("Loading full state dict from %s", safetensors_path)
        state = load_file(str(safetensors_path), device=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning("Missing keys: %s", missing[:10])
        if unexpected:
            log.warning("Unexpected keys: %s", unexpected[:10])
    else:
        log.error("No model.safetensors found at %s", checkpoint_dir)

    # Move CNN to the target device (LLM landed there via device_map)
    model.cnn.to(device)
    model.eval()
    return model, tokenizer


def _generate(
    model,
    tokenizer,
    fen: str,
    question: str,
    use_board_tokens: bool,
    device: str,
    max_new_tokens: int = 64,
) -> str:
    """Generate a response for a board-reading question."""
    sentinel_block = _SENTINEL_BLOCK if use_board_tokens else ""
    messages = [{"role": "user", "content": f"{sentinel_block}\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    n_sent = (input_ids[0] == BOARD_TOKEN_ID).sum().item()

    # Build board tensor
    board = chess.Board(fen)
    if use_board_tokens and n_sent == BOARD_TOKENS_PER_POSITION:
        board_tensor = board_to_tensor(board).unsqueeze(0).to(torch.bfloat16).to(device)
        move_counts = torch.tensor([1], device=device)
    else:
        board_tensor = torch.zeros(0, 19, 8, 8, dtype=torch.bfloat16, device=device)
        move_counts = torch.tensor([0], device=device)

    with torch.no_grad():
        out = model.generate_with_encoder(
            input_ids=input_ids,
            board_tensors_flat=board_tensor,
            move_counts=move_counts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = out[0][input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _score(prediction: str, expected: str) -> bool:
    """Loose match: check if expected answer appears in prediction."""
    pred = prediction.lower().strip()
    exp = expected.lower().strip()
    return exp in pred or all(line.strip() in pred for line in exp.split("\n") if line.strip())


def run_eval(
    checkpoint_dir: str,
    config_path: str,
    device: str = "cuda:0",
) -> dict:
    model, tokenizer = _load_model(checkpoint_dir, config_path, device)

    # Add generate_with_encoder helper if not present
    if not hasattr(model, "generate_with_encoder"):
        _add_generate_helper(model, tokenizer)

    results = {"with_board": [], "without_board": [], "probes": []}

    for probe in _PROBES:
        fen = probe["fen"]
        question = probe["question"]
        expected = probe["answer"]
        task = probe["task"]

        pred_with = _generate(model, tokenizer, fen, question, use_board_tokens=True, device=device)
        pred_without = _generate(
            model, tokenizer, fen, question, use_board_tokens=False, device=device
        )

        correct_with = _score(pred_with, expected)
        correct_without = _score(pred_without, expected)

        results["with_board"].append(correct_with)
        results["without_board"].append(correct_without)
        results["probes"].append(
            {
                "task": task,
                "fen": fen,
                "question": question,
                "expected": expected,
                "pred_with_board": pred_with,
                "pred_without_board": pred_without,
                "correct_with": correct_with,
                "correct_without": correct_without,
            }
        )

        status_with = "✓" if correct_with else "✗"
        status_without = "✓" if correct_without else "✗"
        log.info(
            "[%s] %s | with_board=%s (%s) | no_board=%s (%s)",
            task,
            question[:40],
            status_with,
            pred_with[:30].replace("\n", " "),
            status_without,
            pred_without[:30].replace("\n", " "),
        )

    acc_with = sum(results["with_board"]) / len(results["with_board"])
    acc_without = sum(results["without_board"]) / len(results["without_board"])
    results["accuracy_with_board"] = acc_with
    results["accuracy_without_board"] = acc_without
    results["board_token_lift"] = acc_with - acc_without

    print()
    print(f"{'─' * 50}")
    print(
        f"  Accuracy with board tokens : {acc_with:.0%}  ({sum(results['with_board'])}/{len(_PROBES)})"
    )
    print(
        f"  Accuracy without board     : {acc_without:.0%}  ({sum(results['without_board'])}/{len(_PROBES)})"
    )
    print(f"  Board token lift           : {results['board_token_lift']:+.0%}")
    print(f"{'─' * 50}")

    if results["board_token_lift"] > 0.1:
        print("  ✓ Model uses board tokens (meaningful lift detected)")
    elif acc_with > 0.5:
        print("  ~ Model answers well but board token lift is small")
    else:
        print("  ✗ Model does not appear to use board tokens")

    return results


def _add_generate_helper(model, tokenizer):
    """Attach a generate_with_encoder method to the model."""
    import types

    def generate_with_encoder(self, input_ids, board_tensors_flat, move_counts, **gen_kwargs):
        # Inject CNN embeddings into inputs_embeds, then call LLM generate
        device = input_ids.device
        dtype = next(iter(self.llm.parameters())).dtype

        inputs_embeds = self.embed_tokens(input_ids).to(dtype)
        move_mask = input_ids == self.move_token_id
        n_sentinels = move_mask.sum().item()

        if n_sentinels > 0 and board_tensors_flat.shape[0] > 0:
            from src.encoder import BOARD_TOKENS_PER_POSITION

            n_boards = n_sentinels // BOARD_TOKENS_PER_POSITION
            cnn_dtype = next(iter(self.cnn.parameters()), None)
            cnn_dtype = cnn_dtype.dtype if cnn_dtype is not None else dtype
            cnn_out = self.cnn(board_tensors_flat.to(device=device, dtype=cnn_dtype))
            cnn_out = cnn_out.to(dtype=dtype)
            if cnn_out.shape[0] < n_boards:
                pad = torch.zeros(
                    n_boards - cnn_out.shape[0],
                    BOARD_TOKENS_PER_POSITION,
                    inputs_embeds.shape[-1],
                    dtype=dtype,
                    device=device,
                )
                cnn_out = torch.cat([cnn_out, pad], dim=0)
            cnn_embs = cnn_out[:n_boards].reshape(n_boards * BOARD_TOKENS_PER_POSITION, -1)
            n_filled = cnn_embs.shape[0]
            positions = move_mask.nonzero(as_tuple=False)[:n_filled]
            inputs_embeds = inputs_embeds.index_put(
                (positions[:, 0], positions[:, 1]), cnn_embs, accumulate=False
            )

        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    model.generate_with_encoder = types.MethodType(generate_with_encoder, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint dir. Defaults to latest in checkpoints/qwen3.5-4b-encoder-pretrain-textbook/",
    )
    parser.add_argument(
        "--config",
        default="recipes-train/qwen3.5-4b-encoder-pretrain-textbook/config.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        ckpt_root = Path("checkpoints/qwen3.5-4b-encoder-pretrain-textbook")
        candidates = sorted(ckpt_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if not candidates:
            print(f"No checkpoints found in {ckpt_root}")
            sys.exit(1)
        checkpoint = str(candidates[-1])
        log.info("Using latest checkpoint: %s", checkpoint)

    results = run_eval(checkpoint, args.config, args.device)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        log.info("Results saved to %s", args.output)
