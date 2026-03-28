"""Generate GRPO Phase 1 board-reading dataset.

Four task types with rule-based (exact) answers:
  - legal_moves   : list all legal SAN moves (sorted)
  - captures      : list capture moves only (sorted)
  - in_check      : yes/no — is the side to move in check?
  - piece_at      : what piece is on square X?

Each record: {"messages": [system, user], "metadata": {"fen": ..., "task": ..., "answer": ...}}
The assistant turn is omitted — GRPOTrainer generates it.

Sources: FENs extracted from grpo_joint_prompts_sf15.jsonl + encoder_pretrain_1b.jsonl
(we just need positions, not the coaching context).

Usage:
    uv run python data/pipeline/generate_grpo_phase1_board_reading.py
    uv run python data/pipeline/generate_grpo_phase1_board_reading.py --target 20000 --eval-size 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert chess coach with deep positional understanding.

You will be shown a board image of a chess position. \
Answer the question precisely and concisely.

Answer format:
<think>
Brief reasoning (a few sentences max).
</think>
<answer>
Your answer here.
</answer>

Answer rules:
- Yes/no questions: answer exactly "yes" or "no".
- Piece questions: color and piece type (e.g. "white rook", "black knight", "empty").
- Count questions: answer with a single integer (e.g. "3").
- Material balance: answer as white minus black with sign (e.g. "+3", "-2", "0").
"""

_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

_COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}

_FILE_NAMES = "abcdefgh"
_RANK_NAMES = "12345678"


def _board_ascii(board: chess.Board) -> str:
    """ASCII board with coordinates."""
    lines = ["  a b c d e f g h"]
    for rank in range(7, -1, -1):
        row = f"{rank + 1} "
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row += ". "
            else:
                row += piece.symbol() + " "
        lines.append(row.rstrip())
    side = "White" if board.turn == chess.WHITE else "Black"
    lines.append(f"  ({side} to move)")
    return "\n".join(lines)


def _side(board: chess.Board) -> str:
    return "White" if board.turn == chess.WHITE else "Black"


def _task_legal_moves(board: chess.Board) -> tuple[str, str]:
    """Ask for all legal moves; answer is sorted SAN list."""
    moves = sorted(board.san(m) for m in board.legal_moves)
    if not moves:
        return "", ""
    user = (
        f"The image shows the board from {_side(board)}'s perspective "
        f"({_side(board)} to move).\n\n"
        "List all legal moves available to the side to move. "
        "One move per line, in SAN notation, sorted alphabetically."
    )
    answer = "\n".join(moves)
    return user, answer


def _task_captures(board: chess.Board) -> tuple[str, str]:
    """Ask for capture moves only."""
    captures = sorted(board.san(m) for m in board.legal_moves if board.is_capture(m))
    user = (
        f"The image shows the board from {_side(board)}'s perspective "
        f"({_side(board)} to move).\n\n"
        "List all capture moves available to the side to move. "
        "One move per line, in SAN notation, sorted alphabetically. "
        'If there are no captures, answer "none".'
    )
    answer = "\n".join(captures) if captures else "none"
    return user, answer


def _task_in_check(board: chess.Board) -> tuple[str, str]:
    """Ask if the side to move is in check."""
    user = (
        f"The image shows the board from {_side(board)}'s perspective "
        f"({_side(board)} to move).\n\n"
        "Is the side to move currently in check? Answer yes or no."
    )
    answer = "yes" if board.is_check() else "no"
    return user, answer


def _task_piece_at(board: chess.Board) -> tuple[str, str]:
    """Ask what piece is on a random square (biased toward occupied squares)."""
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    if occupied and random.random() < 0.7:
        sq = random.choice(occupied)
    else:
        sq = random.randint(0, 63)

    sq_name = chess.square_name(sq)
    piece = board.piece_at(sq)
    if piece is None:
        answer = "empty"
    else:
        answer = f"{_COLOR_NAMES[piece.color]} {_PIECE_NAMES[piece.piece_type]}"

    user = (
        f"The image shows the board from {_side(board)}'s perspective "
        f"({_side(board)} to move).\n\n"
        f"What piece is on square {sq_name}? "
        'Answer with color and piece type (e.g. "white rook") or "empty".'
    )
    return user, answer


def _task_count_pieces(board: chess.Board) -> tuple[str, str]:
    """Ask how many pieces of a given color are on the board."""
    color = random.choice([chess.WHITE, chess.BLACK])
    color_name = _COLOR_NAMES[color]
    count = (
        len(board.pieces(chess.PAWN, color))
        + len(board.pieces(chess.KNIGHT, color))
        + len(board.pieces(chess.BISHOP, color))
        + len(board.pieces(chess.ROOK, color))
        + len(board.pieces(chess.QUEEN, color))
        + len(board.pieces(chess.KING, color))
    )
    user = (
        f"The image shows the board from {_side(board)}'s perspective "
        f"({_side(board)} to move).\n\n"
        f"How many {color_name} pieces are on the board (including the king and pawns)? "
        "Answer with a single integer."
    )
    answer = str(count)
    return user, answer


def _task_piece_count_by_type(board: chess.Board) -> tuple[str, str]:
    """Ask how many pieces of a specific type and color are on the board."""
    color = random.choice([chess.WHITE, chess.BLACK])
    piece_type = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    color_name = _COLOR_NAMES[color]
    piece_name = _PIECE_NAMES[piece_type]
    count = len(board.pieces(piece_type, color))
    user = (
        f"The image shows the board from {_side(board)}'s perspective "
        f"({_side(board)} to move).\n\n"
        f"How many {color_name} {piece_name}s are on the board? "
        "Answer with a single integer."
    )
    answer = str(count)
    return user, answer


def _task_attacked_by(board: chess.Board) -> tuple[str, str]:
    """Ask if a random square is attacked by a given color."""
    sq = random.randint(0, 63)
    sq_name = chess.square_name(sq)
    color = random.choice([chess.WHITE, chess.BLACK])
    color_name = _COLOR_NAMES[color]
    attacked = board.is_attacked_by(color, sq)
    user = (
        f"The image shows the board from {_side(board)}'s perspective "
        f"({_side(board)} to move).\n\n"
        f"Is square {sq_name} attacked by {color_name}? Answer yes or no."
    )
    answer = "yes" if attacked else "no"
    return user, answer


def _task_material_count(board: chess.Board) -> tuple[str, str]:
    """Ask for the material balance (white minus black, in pawns)."""
    _PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    white_mat = sum(_PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE)) for pt in _PIECE_VALUES)
    black_mat = sum(_PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK)) for pt in _PIECE_VALUES)
    balance = white_mat - black_mat
    if balance > 0:
        answer = f"+{balance}"
    elif balance < 0:
        answer = str(balance)
    else:
        answer = "0"
    user = (
        f"The image shows the board from {_side(board)}'s perspective "
        f"({_side(board)} to move).\n\n"
        "What is the material balance? Count pawns=1, knights=3, bishops=3, rooks=5, queens=9. "
        'Answer as white minus black (e.g. "+3", "-2", or "0").'
    )
    return user, answer


_TASK_FNS = {
    "legal_moves": _task_legal_moves,
    "captures": _task_captures,
    "in_check": _task_in_check,
    "piece_at": _task_piece_at,
    "count_pieces": _task_count_pieces,
    "piece_count_by_type": _task_piece_count_by_type,
    "attacked_by": _task_attacked_by,
    "material_count": _task_material_count,
}

# Sampling weights — equal mix of 6 tasks (legal_moves/captures deferred to later curriculum)
_TASK_WEIGHTS = {
    "legal_moves": 0.0,
    "captures": 0.0,
    "in_check": 1 / 6,
    "piece_at": 1 / 6,
    "count_pieces": 1 / 6,
    "piece_count_by_type": 1 / 6,
    "attacked_by": 1 / 6,
    "material_count": 1 / 6,
}


def _iter_fens(source_paths: list[str]):
    """Yield FENs from multiple JSONL source files."""
    for path in source_paths:
        p = Path(path)
        if not p.exists():
            log.warning("Source not found: %s", path)
            continue
        log.info("Reading FENs from %s", path)
        with open(p) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                # grpo_joint_prompts format: metadata.fen
                fen = rec.get("metadata", {}).get("fen") or rec.get("fen", "")
                if fen:
                    yield fen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "data/processed/grpo_joint_prompts_sf15.jsonl",
            "data/processed/grpo_phase1_prompts.jsonl",
            "data/processed/encoder_pretrain_1b.jsonl",
        ],
    )
    parser.add_argument("--output", default="data/processed/grpo_phase1c_board_reading.jsonl")
    parser.add_argument(
        "--eval-output", default="data/processed/grpo_phase1c_board_reading_eval.jsonl"
    )
    parser.add_argument("--target", type=int, default=100000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Active tasks (weight > 0), each FEN gets one sample per active task
    active_tasks = [t for t, w in _TASK_WEIGHTS.items() if w > 0]

    total_needed = args.target + args.eval_size
    rows: list[dict] = []
    seen_fens: set[str] = set()
    skipped = 0

    for fen in _iter_fens(args.sources):
        if len(rows) >= total_needed:
            break
        if fen in seen_fens:
            continue
        seen_fens.add(fen)

        try:
            board = chess.Board(fen)
        except ValueError:
            skipped += 1
            continue

        # Skip terminal positions
        if board.is_game_over():
            skipped += 1
            continue

        # One sample per active task — different questions, same position
        for task in active_tasks:
            if len(rows) >= total_needed:
                break
            try:
                user, answer = _TASK_FNS[task](board)
            except Exception as exc:
                log.debug("Task %s failed for %s: %s", task, fen, exc)
                continue

            if not user or not answer:
                continue

            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    "metadata": {"fen": fen, "task": task, "answer": answer},
                }
            )

    log.info("Generated %d rows (skipped=%d, unique_fens=%d)", len(rows), skipped, len(seen_fens))

    if len(rows) < args.eval_size + 100:
        log.error("Not enough rows — check source files. Got %d, need %d", len(rows), total_needed)
        sys.exit(1)

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

    # Print task distribution
    from collections import Counter

    task_dist = Counter(r["metadata"]["task"] for r in train_rows)
    for t, n in sorted(task_dist.items()):
        log.info("  %-15s %5d (%.1f%%)", t, n, 100 * n / len(train_rows))


if __name__ == "__main__":
    main()
