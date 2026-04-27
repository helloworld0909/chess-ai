"""Generate GRPO training data from Lichess tactical puzzles.

Each Lichess puzzle has a FEN (position before the opponent's move) and a Moves list
(space-separated long algebraic). This script expands each puzzle into one sub-question
per player move:

  Moves[0] — opponent's move (applied silently to reach the tactic)
  Moves[1] — player's first winning move  ← model must find this
  Moves[2] — opponent's forced reply       (applied silently)
  Moves[3] — player's second winning move ← model must find this
  ...

Output JSONL records (flat metadata only — prompts are built dynamically at training time):
  {
    "fen": str,
    "solution_uci": str,
    "color": str,        # "White" or "Black"
    "themes": list[str],
    "rating": int,
    "puzzle_id": str,
    "depth": int
  }

Sorted by depth then rating for natural curriculum (easy single-move puzzles first).

Usage:
    uv run python data/pipeline/generate_puzzle_grpo.py \
        --output data/processed/puzzle_grpo_train.jsonl \
        --eval-output data/processed/puzzle_grpo_eval.jsonl \
        --rating-min 1000 --rating-max 2200 \
        --min-popularity 50 \
        --eval-size 10000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator

import chess
from datasets import load_dataset
from tqdm import tqdm


def expand_puzzle(puzzle: dict) -> Iterator[dict]:
    """Expand one puzzle into N sub-questions (one per player move)."""
    moves_str = puzzle.get("Moves", "")
    moves = moves_str.split()
    if len(moves) < 2:
        return

    try:
        board = chess.Board(puzzle["FEN"])
        # Apply opponent's move silently
        board.push(chess.Move.from_uci(moves[0]))
    except Exception:
        return

    themes = puzzle.get("Themes") or []
    rating = puzzle.get("Rating", 1500)
    puzzle_id = puzzle.get("PuzzleId", "")
    total_moves = len(moves) // 2  # player moves only (excludes opponent moves)

    for i in range(1, len(moves), 2):
        puzzle_fen = board.fen()
        solution_uci = moves[i]
        depth = (i + 1) // 2
        color = "White" if board.turn == chess.WHITE else "Black"

        # Validate the solution move is legal
        try:
            solution_move = chess.Move.from_uci(solution_uci)
            if solution_move not in board.legal_moves:
                break
        except Exception:
            break

        yield {
            "fen": puzzle_fen,
            "solution_uci": solution_uci,
            "color": color,
            "themes": themes,
            "rating": rating,
            "puzzle_id": puzzle_id,
            "depth": depth,
            "total_moves": total_moves,
        }

        # Advance board: apply player's move then opponent's reply
        try:
            board.push(chess.Move.from_uci(moves[i]))
            if i + 1 < len(moves):
                board.push(chess.Move.from_uci(moves[i + 1]))
        except Exception:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate puzzle GRPO dataset")
    parser.add_argument("--output", default="data/processed/puzzle_grpo_train.jsonl")
    parser.add_argument("--eval-output", default="data/processed/puzzle_grpo_eval.jsonl")
    parser.add_argument("--rating-min", type=int, default=1000)
    parser.add_argument("--rating-max", type=int, default=2200)
    parser.add_argument("--min-popularity", type=int, default=50)
    parser.add_argument("--eval-size", type=int, default=10000)
    parser.add_argument("--max-puzzles", type=int, default=None,
                        help="Cap on source puzzles (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading Lichess/chess-puzzles dataset...")
    ds = load_dataset("Lichess/chess-puzzles", split="train")

    records: list[dict] = []
    n_puzzles = 0
    n_skipped = 0

    source = ds
    if args.max_puzzles:
        source = ds.select(range(min(args.max_puzzles * 3, len(ds))))  # over-select for filtering

    for puzzle in tqdm(source, desc="Processing puzzles"):
        rating = puzzle.get("Rating", 0)
        popularity = puzzle.get("Popularity", 0)

        if not (args.rating_min <= rating <= args.rating_max):
            n_skipped += 1
            continue
        if popularity < args.min_popularity:
            n_skipped += 1
            continue

        for record in expand_puzzle(puzzle):
            records.append(record)

        n_puzzles += 1
        if args.max_puzzles and n_puzzles >= args.max_puzzles:
            break

    print(f"Processed {n_puzzles} puzzles → {len(records)} sub-questions ({n_skipped} skipped)")

    # Sort by (total_moves, depth, rating) for natural curriculum:
    # shorter puzzles first (total_moves=difficulty), then shallower moves, then easier rating
    records.sort(key=lambda r: (r["total_moves"], r["depth"], r["rating"]))

    # Split eval set (proportional stratification by total_moves — mirrors train distribution)
    eval_records: list[dict] = []
    from collections import defaultdict
    bucket_idxs: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        bucket_idxs[r["total_moves"]].append(i)
    for idxs in bucket_idxs.values():
        random.shuffle(idxs)

    total = len(records)
    eval_ids: set[int] = set()
    for tm, idxs in sorted(bucket_idxs.items()):
        n_from_bucket = max(1, round(args.eval_size * len(idxs) / total))
        for idx in idxs[:n_from_bucket]:
            eval_ids.add(idx)

    # Trim or top-up to exact eval_size
    eval_ids_list = list(eval_ids)
    if len(eval_ids_list) > args.eval_size:
        eval_ids_list = eval_ids_list[:args.eval_size]
    eval_ids = set(eval_ids_list)

    eval_records = [records[i] for i in sorted(eval_ids)]
    train_records = [r for i, r in enumerate(records) if i not in eval_ids]

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.eval_output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(args.eval_output, "w") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")

    print(f"Train: {len(train_records)} records → {args.output}")
    print(f"Eval:  {len(eval_records)} records → {args.eval_output}")
    tm_dist = {}
    for r in train_records:
        tm = r["total_moves"]
        tm_dist[tm] = tm_dist.get(tm, 0) + 1
    print("Total_moves distribution (train):", dict(sorted(tm_dist.items())))


if __name__ == "__main__":
    main()
