"""ChessQA-Benchmark evaluation for ChessLMWithEncoder.

Loads wieeii/ChessQA-Benchmark (3,400 examples across 5 subsets) and evaluates
the encoder model. Designed to be run after phase-1 alignment or SFT training
to measure how much chess reasoning the model has acquired.

Dataset subsets:
  motifs (600)          — tactical pattern recognition
  position_judgement (500) — evaluating positions
  semantic (400)        — language/rule understanding
  short_tactics (900)   — short tactical sequences
  structural (1.1k)     — pawn structure, piece placement

Answer format: structured coordinate notation (e.g. "f3>e4" or "f3>e4-g5>f4").
The prompt includes format_examples so the model can learn the expected format
even without fine-tuning on this benchmark.

Matching:
  exact  — prediction == correct_answer (stripped, lowercase)
  soft   — correct_answer appears as a substring of prediction

Usage:
    python scripts/eval_chessqa.py \\
        --checkpoint checkpoints/qwen3.5-4b-encoder-phase1-alignment/checkpoint-4000 \\
        --config recipes-train/qwen3.5-4b-encoder-phase1-alignment/config.yaml

    # Specific subset only
    python scripts/eval_chessqa.py --checkpoint ... --subset short_tactics

    # Limit examples (quick sanity check)
    python scripts/eval_chessqa.py --checkpoint ... --max-examples 100

    # Save JSON results
    python scripts/eval_chessqa.py --checkpoint ... --output results/chessqa_phase1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
    "Use them to identify pieces and answer questions about the position."
)

_SUBSETS = {"motifs", "position_judgement", "semantic", "short_tactics", "structural"}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _get_context(fen: str) -> str:
    """Generate board context string (piece arrangement + legal moves) from FEN."""
    try:
        board = chess.Board(fen)
    except Exception:
        return ""
    piece_names = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
    pieces: dict[str, list[str]] = {}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            color = "White" if piece.color == chess.WHITE else "Black"
            key = f"{color} {piece_names[piece.piece_type]}"
            pieces.setdefault(key, []).append(chess.square_name(sq))
    arrangement = ", ".join(f"{k}: {v}" for k, v in pieces.items())
    legal_moves = ", ".join(sorted(m.uci() for m in board.legal_moves))
    return f"Piece arrangement: {arrangement}\nLegal moves: {legal_moves}\n\n"


def _strip_fen_preamble(question: str, fen: str, add_context: bool) -> str:
    """Remove the FEN preamble and fill CONTEXT_PLACEHOLDER.

    When strip_fen=True: removes the FEN text preamble; CONTEXT_PLACEHOLDER is
    replaced with board context (piece arrangement + legal moves) so the task
    instruction is preserved even without the FEN text.
    When strip_fen=False: leaves the FEN preamble; replaces CONTEXT_PLACEHOLDER
    with board context the same way.
    """
    for preamble in (
        f"You are given a chess position in FEN: {fen}.\n",
        f"You are analyzing a chess position in FEN: {fen}.\n",
    ):
        if question.startswith(preamble):
            question = question[len(preamble) :]
            break
    context = _get_context(fen) if add_context else ""
    question = question.replace("CONTEXT_PLACEHOLDER", context)
    return question.lstrip()


def _build_prompt(
    question: str,
    fen: str,
    format_examples: list[str],
    strip_fen: bool,
    task_type: str = "",
) -> str:
    """Build user message with optional FEN stripping and format hint.

    CONTEXT_PLACEHOLDER is replaced with piece arrangement + legal moves for all
    tasks EXCEPT structural ones — structural tasks ask the model to enumerate
    pieces, so injecting the context would give away the answer.
    When strip_fen=True: FEN preamble is stripped (model uses encoder tokens).
    When strip_fen=False: FEN preamble is kept (model sees FEN text).
    """
    add_context = not task_type.startswith("structural")
    question = _strip_fen_preamble(question, fen, add_context=add_context)
    if not strip_fen:
        question = f"You are given a chess position in FEN: {fen}.\n{question}"
    # Replace inline placeholder with the first example (e.g. in short_tactics questions)
    if format_examples and "FORMAT_EXAMPLE_PLACEHOLDER" in question:
        question = question.replace("FORMAT_EXAMPLE_PLACEHOLDER", format_examples[0])
    if format_examples:
        examples_str = "\n".join(f"  {e}" for e in format_examples[:3])
        return f"{question}\n\nAnswer using this format (examples):\n{examples_str}"
    return question


def _make_prompt(
    tokenizer,
    fen: str,
    question: str,
    format_examples: list[str],
    thinking: bool,
    strip_fen: bool,
    task_type: str = "",
) -> str:
    """Build the full prompt string for one example."""
    user_content = (
        _BOARD_BLOCK + "\n\n" + _build_prompt(question, fen, format_examples, strip_fen, task_type)
    )
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    if not thinking and prompt_text.endswith("<think>\n"):
        prompt_text = prompt_text + "</think>\n\n"
    return prompt_text


def infer_batch(
    model,
    tokenizer,
    batch: list[dict],
    max_new_tokens: int,
    thinking: bool,
    strip_fen: bool,
) -> list[str]:
    """Run batched inference for a list of ChessQA examples.

    Each item in batch must have keys: fen, question, format_examples, task_type.
    Returns decoded predictions in the same order.
    """
    device = next(model.cnn.parameters()).device

    # Build board tensors and prompts
    board_tensors: list[torch.Tensor] = []
    prompts: list[str] = []
    valid: list[bool] = []
    for ex in batch:
        try:
            fen = ex["fen"].split("|")[0].strip()
            board = chess.Board(fen)
            board_tensors.append(board_to_tensor(board).to(torch.bfloat16))
            prompts.append(
                _make_prompt(
                    tokenizer,
                    fen,
                    ex["question"],
                    ex["format_examples"],
                    thinking,
                    strip_fen,
                    task_type=ex.get("task_type", ""),
                )
            )
            valid.append(True)
        except Exception:
            _logger.warning("Invalid FEN: %s", ex["fen"])
            prompts.append(".")  # non-empty placeholder so tokenizer produces seq_len > 0
            board_tensors.append(torch.zeros(19, 8, 8, dtype=torch.bfloat16))
            valid.append(False)

    # Tokenize with left-padding
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Stack board tensors: (B, 19, 8, 8) — one per example
    boards_batch = torch.stack(board_tensors).to(device)

    with torch.no_grad():
        # encoder_model.generate receives input_ids for CNN injection, then calls
        # llm.generate with inputs_embeds only (no input_ids). Transformers therefore
        # returns only the newly generated token IDs (no prompt prefix).
        out = model.generate(
            input_ids=input_ids,
            board_tensors_flat=boards_batch,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, ids in enumerate(out):
        if not valid[i]:
            results.append(("", ""))
            continue
        full = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if thinking:
            idx = full.rfind("</think>")
            answer = full[idx + len("</think>") :].strip() if idx != -1 else full
        else:
            answer = full
        # full: complete output including think block (saved to JSON)
        # answer: text after </think> used for matching
        results.append((full, answer))
    return results


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    return s.strip().lower()


def _extract_final_answer(pred: str) -> str:
    """Extract final answer using the same logic as the reference eval script.

    Looks for the last 'FINAL ANSWER: <text>' up to the next newline, then falls
    back to \\boxed{} format, then returns the full prediction as-is.
    """
    matches = list(re.finditer(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", pred, re.IGNORECASE | re.DOTALL))
    if matches:
        answer = matches[-1].group(1).strip()
        answer = re.sub(r"^FINAL ANSWER:\s*", "", answer, flags=re.IGNORECASE).strip()
        answer = re.sub(r"^\*+|\*+$", "", answer).strip()
        return answer

    boxed = list(re.finditer(r"[Tt]he\s+final\s+answer\s+is\s+\$?\\boxed\{([^}]+)\}\$?", pred))
    if boxed:
        return boxed[-1].group(1).strip()

    return pred


def _normalize_lists(s: str) -> str:
    """Sort square lists within bracket notation so order doesn't affect matching.

    e.g. "['a5', 'e3', 'c5']" → "['a5', 'c5', 'e3']"
    Handles answers like structural_piece_arrangement where squares inside each
    [...] list may appear in any order.
    """
    import re

    def sort_bracket(m: re.Match) -> str:
        items = [x.strip().strip("'\"") for x in m.group(1).split(",")]
        return "[" + ", ".join(f"'{x}'" for x in sorted(items)) + "]"

    return re.sub(r"\[([^\]]+)\]", sort_bracket, s)


def match_exact(pred: str, correct: str, answer_type: str = "single") -> bool:
    extracted = _normalize(_extract_final_answer(pred))
    if answer_type == "multi":
        pred_set = {item.strip() for item in extracted.split(",") if item.strip()}
        correct_set = {item.strip() for item in _normalize(correct).split(",") if item.strip()}
        return pred_set == correct_set
    # For answers containing bracket lists, sort contents before comparing
    correct_norm = _normalize(correct)
    if "[" in correct_norm:
        return _normalize_lists(extracted) == _normalize_lists(correct_norm)
    return extracted == correct_norm


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ChessLMWithEncoder on ChessQA-Benchmark")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to encoder model checkpoint directory",
    )
    parser.add_argument(
        "--config",
        default="recipes-train/qwen3.5-4b-encoder-phase1-alignment/config.yaml",
        help="Training config YAML (for architecture params)",
    )
    parser.add_argument(
        "--subset",
        default=None,
        choices=sorted(_SUBSETS) + [None],
        help="Evaluate only this subset (default: all)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit total examples (useful for quick sanity checks)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save full results to this JSON file (e.g. results/chessqa_phase1.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per GPU (default: 1 with thinking, 32 without)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max tokens to generate (default: 512 with thinking, 64 without)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3 thinking mode (<think> block before answer). "
        "Full output (including think block) saved to JSON; think block stripped for answer matching. "
        "Defaults: max-new-tokens=32768, batch-size=1.",
    )
    parser.add_argument(
        "--strip-fen",
        action="store_true",
        default=False,
        help="Strip the FEN preamble from questions so the model must use encoder "
        "tokens (default: off).",
    )
    parser.add_argument(
        "--no-strip-fen",
        dest="strip_fen",
        action="store_false",
        help="Keep the FEN text in the question (model can read position from text).",
    )
    args = parser.parse_args()

    # Defaults depend on thinking mode
    if args.max_new_tokens is None:
        args.max_new_tokens = 32768 if args.thinking else 512
    if args.batch_size is None:
        args.batch_size = 4 if args.thinking else 32

    # DDP: init process group BEFORE any CUDA ops so NCCL can enumerate GPUs correctly
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    import torch
    import torch.distributed as dist

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    # ------------------------------------------------------------------
    # Load dataset in the main process (avoids redundant HF downloads)
    # ------------------------------------------------------------------
    _logger.info("Loading wieeii/ChessQA-Benchmark...")
    from datasets import load_dataset

    subsets_to_load = [args.subset] if args.subset else sorted(_SUBSETS)
    examples: list[dict] = []
    for config_name in subsets_to_load:
        _logger.info("  Loading config: %s", config_name)
        ds = load_dataset("wieeii/ChessQA-Benchmark", config_name)
        for split_data in ds.values():  # type: ignore[union-attr]
            for row in split_data:
                examples.append(dict(row))

    import random

    random.shuffle(examples)

    if args.max_examples:
        examples = examples[: args.max_examples]

    _logger.info(
        "Evaluating %d examples%s...",
        len(examples),
        f" (subset={args.subset})" if args.subset else "",
    )

    # ------------------------------------------------------------------
    # DDP: shard examples across GPUs if launched with torchrun
    # ------------------------------------------------------------------
    # Each rank handles its shard
    shard = examples[local_rank::world_size]

    _logger.info(
        "Rank %d/%d — evaluating %d/%d examples on GPU %d",
        local_rank,
        world_size,
        len(shard),
        len(examples),
        local_rank,
    )

    _logger.info("Loading model from %s...", args.checkpoint)
    model, tokenizer = load_encoder_model(args.checkpoint, args.config, device=local_rank)
    model.eval()

    _dummy = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "question": "Side to move?",
            "format_examples": [],
            "task_type": "",
        }
    ]
    infer_batch(
        model, tokenizer, _dummy, max_new_tokens=4, thinking=args.thinking, strip_fen=args.strip_fen
    )
    _logger.info("Rank %d model ready.", local_rank)

    results: list[dict] = []
    for batch_start in tqdm(
        range(0, len(shard), args.batch_size),
        desc=f"ChessQA eval [rank {local_rank}]",
        disable=local_rank != 0,
    ):
        batch = shard[batch_start : batch_start + args.batch_size]
        batch_inputs = [
            {
                "fen": ex["input"],
                "question": ex["question"],
                "format_examples": ex.get("format_examples") or [],
                "task_type": ex.get("task_type", ""),
            }
            for ex in batch
        ]
        preds = infer_batch(
            model,
            tokenizer,
            batch_inputs,
            args.max_new_tokens,
            thinking=args.thinking,
            strip_fen=args.strip_fen,
        )
        for ex, (full_output, answer) in zip(batch, preds):
            results.append(
                {
                    "task_id": ex.get("task_id", ""),
                    "task_group": ex.get("task_group", "unknown"),
                    "task_type": ex.get("task_type", ""),
                    "fen": ex["input"],
                    "question": ex["question"],
                    "correct_answer": ex["correct_answer"],
                    "answer_type": ex.get("answer_type", "single"),
                    "predicted": full_output,
                    "predicted_answer": _extract_final_answer(answer),
                    "exact": match_exact(
                        answer, ex["correct_answer"], ex.get("answer_type", "single")
                    ),
                }
            )

    # ------------------------------------------------------------------
    # Gather results from all ranks onto rank 0
    # ------------------------------------------------------------------
    if world_size > 1:
        import pickle

        payload_bytes = pickle.dumps(results)
        all_bytes = [None] * world_size
        dist.all_gather_object(all_bytes, payload_bytes)
        if local_rank == 0:
            results = []
            for b in all_bytes:
                assert b is not None
                results.extend(pickle.loads(b))
        dist.destroy_process_group()

    if local_rank != 0:
        return

    # ------------------------------------------------------------------
    # Aggregate per-subset stats
    # ------------------------------------------------------------------
    per_subset: dict[str, dict[str, int]] = defaultdict(lambda: {"exact": 0, "total": 0})
    for r in results:
        subset = r.get("task_group", "unknown")
        per_subset[subset]["exact"] += int(r["exact"])
        per_subset[subset]["total"] += 1

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    total = len(results)
    total_exact = sum(r["exact"] for r in results)

    SEP = "=" * 72
    fen_label = "strip-fen=ON" if args.strip_fen else "strip-fen=OFF"
    print(f"\n{SEP}")
    print(f"  ChessQA-Benchmark — {Path(args.checkpoint).name}  [{fen_label}]")
    print(SEP)
    print(f"  Overall  exact={total_exact}/{total} ({100 * total_exact / total:.1f}%)")

    samples_by_subset: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        samples_by_subset[r["task_group"]].append(r)

    for subset_name in sorted(samples_by_subset):
        subset_results = samples_by_subset[subset_name]
        n = len(subset_results)
        n_exact = sum(r["exact"] for r in subset_results)
        print(f"\n  [{subset_name}]  exact={n_exact}/{n} ({100 * n_exact / n:.1f}%)")
        for r in subset_results[:10]:
            mark = "✓" if r["exact"] else "✗"
            print(f"\n    {mark} [{r['task_type']}]")
            print(f"       Q: {r['question']}")
            print(f"       correct: {r['correct_answer']}")
            print(f"       got:     {r['predicted_answer']}")

    print(f"\n{SEP}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "thinking": args.thinking,
            "strip_fen": args.strip_fen,
            "max_new_tokens": args.max_new_tokens,
            "subset_filter": args.subset,
            "total": total,
            "exact_accuracy": total_exact / total if total else 0.0,
            "per_subset": {
                k: {**v, "exact_acc": v["exact"] / v["total"]} for k, v in per_subset.items()
            },
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        _logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
