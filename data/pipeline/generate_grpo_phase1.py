"""Generate clean GRPO Phase 1 prompt dataset from lines_30k.jsonl.

Prompts are purpose-built for tool-use RL:
  - No engine key lines (model must call tools)
  - No SF15 pre-computation needed (fast generation)
  - Clean task instruction matching the phase-1 system prompt

Each record: {"messages": [system, user], "metadata": {"fen": ..., "move_san": ...}}
The assistant turn is omitted — GRPOTrainer generates it.

Usage:
    uv run python data/pipeline/generate_grpo_phase1.py
    uv run python data/pipeline/generate_grpo_phase1.py --target 15000 --eval-size 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from tutor.prompts import board_ascii, move_facts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert chess coach giving move-by-move feedback to a student.

You have tools to analyse the position before writing your coaching comment.
Use them — especially sf15_term_diff to verify strategic claims.

Workflow:
1. Think step by step inside <think>...</think> — reason about the position, call tools, interpret results.
2. Write your coaching comment AFTER </think> — direct, second person, no tool syntax.

Coaching rules:
- Write in second person ("You centralise the knight", "You play Nd5 to...").
- Name the key strategic or tactical idea and explain WHY this move achieves it.
- Reference specific moves, squares, or pieces — never vague platitudes.
- OPENING (moves 1-12): be concise (2-3 sentences), identify opening/variation.
- MIDDLEGAME: explain the key idea and top alternatives (3-5 sentences).
- ENDGAME: analyse key lines in depth (4-6 sentences).
- Do NOT parrot engine numbers, generic study advice, or opening recommendations.
- Do NOT include <tool_call> tags in your final comment.
"""


def _build_user_prompt(fen: str, move_san: str) -> str:
    board = chess.Board(fen)
    board_str = board_ascii(board)

    try:
        move = board.parse_san(move_san)
        facts = move_facts(board, move)
        board_after = board.copy()
        board_after.push(move)
        board_after_str = board_ascii(board_after)
        fen_after = board_after.fen()
    except Exception:
        facts = []
        board_after_str = ""
        fen_after = ""

    prompt = f"## Position\n\nBoard before the move:\n{board_str}\nFEN: {fen}\n"
    prompt += f"\n## Move Played\n\nMove: {move_san}\n"
    if board_after_str:
        prompt += f"\nBoard after the move:\n{board_after_str}\nFEN: {fen_after}\n"
    if facts:
        prompt += "\n## Verified Move Facts\n\n" + "\n".join(f"- {f}" for f in facts) + "\n"
    prompt += (
        "\n## Task\n\n"
        "Use your tools to analyse this position, then give a coaching comment "
        "to the student about their move."
    )
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/processed/lines_30k.jsonl")
    parser.add_argument("--output", default="data/processed/grpo_phase1_prompts.jsonl")
    parser.add_argument("--eval-output", default="data/processed/grpo_phase1_prompts_eval.jsonl")
    parser.add_argument("--target", type=int, default=15000)
    parser.add_argument("--eval-size", type=int, default=100)
    args = parser.parse_args()

    rows: list[dict] = []
    skipped = 0

    with open(args.source) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            fen = rec.get("fen", "")
            move_san = rec.get("move_san", "")
            if not fen or not move_san:
                skipped += 1
                continue

            try:
                user_content = _build_user_prompt(fen, move_san)
            except Exception as exc:
                log.debug("Skipping %s %s: %s", fen, move_san, exc)
                skipped += 1
                continue

            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "metadata": {"fen": fen, "move_san": move_san},
                }
            )

            if len(rows) >= args.target + args.eval_size:
                break

    log.info("Generated %d rows (skipped=%d)", len(rows), skipped)

    import random

    random.seed(42)
    random.shuffle(rows)

    eval_rows = rows[: args.eval_size]
    train_rows = rows[args.eval_size : args.eval_size + args.target]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.eval_output, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("Train: %d → %s", len(train_rows), args.output)
    log.info("Eval:  %d → %s", len(eval_rows), args.eval_output)


if __name__ == "__main__":
    main()
