"""Alignment eval: measure per-task accuracy on the medium alignment eval set.

Loads the alignment eval JSONL and runs the encoder model on each example,
reporting exact-match accuracy broken down by task category (easy vs medium).

Usage:
    python scripts/eval_alignment.py \\
        --checkpoint checkpoints/qwen3.5-4b-encoder-phase1-alignment-medium/checkpoint-7500 \\
        --config recipes-train/qwen3.5-4b-encoder-phase1-alignment/config.yaml

    # Custom eval file
    python scripts/eval_alignment.py --checkpoint ... \\
        --eval-file data/processed/alignment_board_description_medium_eval.jsonl

    # Limit examples (quick sanity check)
    python scripts/eval_alignment.py --checkpoint ... --max-examples 500

    # Save JSON results
    python scripts/eval_alignment.py --checkpoint ... --output results/alignment_eval.json

    # DDP (2 GPUs)
    torchrun --nproc_per_node=2 --master_port=29502 scripts/eval_alignment.py --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import chess
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder import BOARD_TOKEN, BOARD_TOKENS_PER_POSITION
from src.encoder.board_tensor import board_to_tensor
from src.tutor.encoder_inference import load_encoder_model

_logger = logging.getLogger(__name__)

_BOARD_BLOCK = BOARD_TOKEN * BOARD_TOKENS_PER_POSITION

_SYSTEM_PROMPT = (
    "You are a chess assistant. The board position is encoded as a sequence of vision tokens. "
    "Analyse the position using the encoded board."
)

_MEDIUM_TASKS = {
    "hanging_pieces",
    "capture_on_square",
    "give_check",
    "threaten_piece_with",
    "fork_move",
    "doubled_pawns",
    "isolated_pawn_at",
    "passed_pawn",
    "checkmate_in_one",
    "board_inventory",
}


def infer_batch(model, tokenizer, batch: list[dict], max_new_tokens: int) -> list[str]:
    """Run batched inference. Each item: {fen, question}. Returns predictions."""
    device = next(model.cnn.parameters()).device

    board_tensors = []
    prompts = []
    for ex in batch:
        try:
            board = chess.Board(ex["fen"])
            board_tensors.append(board_to_tensor(board).to(torch.bfloat16))
        except Exception:
            board_tensors.append(torch.zeros(19, 8, 8, dtype=torch.bfloat16))

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _BOARD_BLOCK + "\n\n" + ex["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if prompt.endswith("<think>\n"):
            prompt = prompt + "</think>\n\n"
        prompts.append(prompt)

    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    boards_batch = torch.stack(board_tensors).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            board_tensors_flat=boards_batch,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in out]


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-task alignment eval for ChessLMWithEncoder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="recipes-train/qwen3.5-4b-encoder-phase1-alignment/config.yaml",
    )
    parser.add_argument(
        "--eval-file",
        default="data/processed/alignment_board_description_medium_eval.jsonl",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # DDP init before any CUDA ops
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    import torch.distributed as dist

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("datasets").setLevel(logging.WARNING)

    # Load eval data on all ranks (small file, OK to read redundantly)
    examples = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            meta = obj.get("metadata", {})
            question = next((m["content"] for m in msgs if m["role"] == "user"), "")
            answer = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            examples.append(
                {
                    "fen": meta.get("fen", chess.STARTING_FEN),
                    "task": meta.get("task", "unknown"),
                    "question": question,
                    "answer": answer,
                }
            )

    if args.max_examples:
        examples = examples[: args.max_examples]

    # Shard across ranks
    shard = examples[local_rank::world_size]

    _logger.info(
        "Rank %d/%d — %d/%d examples on GPU %d",
        local_rank,
        world_size,
        len(shard),
        len(examples),
        local_rank,
    )

    model, tokenizer = load_encoder_model(args.checkpoint, args.config, device=local_rank)
    model.eval()
    _logger.info("Rank %d model ready.", local_rank)

    results = []
    for i in tqdm(
        range(0, len(shard), args.batch_size),
        desc=f"Alignment eval [rank {local_rank}]",
        disable=local_rank != 0,
    ):
        batch = shard[i : i + args.batch_size]
        preds = infer_batch(model, tokenizer, batch, args.max_new_tokens)
        for ex, pred in zip(batch, preds):
            results.append(
                {
                    "task": ex["task"],
                    "tier": "medium" if ex["task"] in _MEDIUM_TASKS else "easy",
                    "fen": ex["fen"],
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "predicted": pred,
                    "exact": pred.strip() == ex["answer"].strip(),
                }
            )

    # Gather from all ranks
    if world_size > 1:
        import pickle

        payload = pickle.dumps(results)
        all_bytes = [None] * world_size
        dist.all_gather_object(all_bytes, payload)
        if local_rank == 0:
            results = []
            for b in all_bytes:
                assert b is not None
                results.extend(pickle.loads(b))
        dist.destroy_process_group()

    if local_rank != 0:
        return

    # Aggregate
    by_task: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "tier": "easy"})
    for r in results:
        by_task[r["task"]]["correct"] += int(r["exact"])
        by_task[r["task"]]["total"] += 1
        by_task[r["task"]]["tier"] = r["tier"]

    total = len(results)
    total_correct = sum(r["exact"] for r in results)
    medium_results = [r for r in results if r["tier"] == "medium"]
    easy_results = [r for r in results if r["tier"] == "easy"]
    medium_correct = sum(r["exact"] for r in medium_results)
    easy_correct = sum(r["exact"] for r in easy_results)

    SEP = "=" * 72
    print(f"\n{SEP}")
    print(f"  Alignment Eval — {Path(args.checkpoint).name}")
    print(SEP)
    print(f"  Overall  {total_correct}/{total} ({100 * total_correct / total:.1f}%)")
    print(
        f"  Medium   {medium_correct}/{len(medium_results)} "
        f"({100 * medium_correct / len(medium_results):.1f}%)"
        if medium_results
        else "  Medium   n/a"
    )
    print(
        f"  Easy     {easy_correct}/{len(easy_results)} "
        f"({100 * easy_correct / len(easy_results):.1f}%)"
        if easy_results
        else "  Easy     n/a"
    )

    print(f"\n  {'Task':<30} {'Tier':<7} {'Correct':>8} {'Total':>7} {'Acc':>7}")
    print(f"  {'-' * 30} {'-' * 6} {'-' * 8} {'-' * 7} {'-' * 7}")
    for task, stats in sorted(by_task.items(), key=lambda x: (x[1]["tier"], x[0])):
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] else 0
        mark = "●" if stats["tier"] == "medium" else " "
        print(
            f"  {mark}{task:<29} {stats['tier']:<7} {stats['correct']:>8} "
            f"{stats['total']:>7} {acc:>6.1f}%"
        )
    print(f"\n{SEP}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "eval_file": args.eval_file,
            "total": total,
            "overall_acc": total_correct / total if total else 0.0,
            "medium_acc": medium_correct / len(medium_results) if medium_results else None,
            "easy_acc": easy_correct / len(easy_results) if easy_results else None,
            "per_task": {t: {**s, "acc": s["correct"] / s["total"]} for t, s in by_task.items()},
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        _logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
