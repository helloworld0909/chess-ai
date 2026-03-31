"""Generate board-description alignment data from FENs.

LLaVA-style alignment: simple, direct board reading tasks with no reasoning.
Each sample: user = <64 sentinels> + short question, assistant = short factual answer.

Task types (diverse descriptions of the same board):
  1.  piece_abbr_at        — "What piece abbreviation is on e4?" → "P" / "empty"
  2.  piece_name_at        — "What piece is on e4?" → "white pawn" / "empty"
  3.  side_to_move         — "Which side is to move?" → "white" / "black"
  4.  piece_list_abbr      — "List all pieces using abbreviations." → "Ra1 Nb1 ..."
  5.  piece_list_full      — "List all pieces by name." → "white rook on a1, ..."
  6.  rank_contents        — "What pieces are on rank 1?" → "Ra1 Nb1 ..."
  7.  file_contents        — "What pieces are on the e file?" → "Pe2 pe7"
  8.  castling_rights      — "What are the castling rights?" → "White: K Q  Black: k q" / none
  9.  en_passant           — "Is there an en passant square?" → "e6" / "none"
  10. move_number          — "What is the current move number?" → "5"
  11. count_piece_type     — "How many white pawns are on the board?" → "5"
  12. count_total_material — "How many pieces does Black have in total?" → "14"
  13. find_piece_type      — "List all squares occupied by white knights." → "b1, g1" / "none"
  14. is_square_occupied   — "Is there a piece on e4?" → "yes" / "no"  (strict 50/50)

Output JSONL fields:
  {"messages": [...], "metadata": {"fen": ..., "task": ...}}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_logger = logging.getLogger(__name__)

_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

_PIECE_ABBR = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def _sq(sq: int) -> str:
    return chess.square_name(sq)


# ---------------------------------------------------------------------------
# Task generators — each returns (question, answer) or None if not applicable
# ---------------------------------------------------------------------------


def _task_piece_abbr_at(board: chess.Board, rng: random.Random) -> tuple[str, str] | None:
    occupied = [s for s in chess.SQUARES if board.piece_at(s) is not None]
    empty = [s for s in chess.SQUARES if board.piece_at(s) is None]
    if occupied and rng.random() < 0.7:
        sq = rng.choice(occupied)
    elif empty:
        sq = rng.choice(empty)
    else:
        sq = rng.randint(0, 63)
    piece = board.piece_at(sq)
    abbr = _PIECE_ABBR[piece.piece_type] if piece else "empty"
    if piece and piece.color == chess.BLACK:
        abbr = abbr.lower()
    q_variants = [
        f"What piece abbreviation is on {_sq(sq)}? Use uppercase for white, lowercase for black.",
        f"Give the piece abbreviation on {_sq(sq)}. Uppercase = white, lowercase = black.",
        f"Name the piece on {_sq(sq)} using standard abbreviation (uppercase white, lowercase black).",
    ]
    return rng.choice(q_variants), abbr


def _task_piece_name_at(board: chess.Board, rng: random.Random) -> tuple[str, str] | None:
    occupied = [s for s in chess.SQUARES if board.piece_at(s) is not None]
    empty = [s for s in chess.SQUARES if board.piece_at(s) is None]
    if occupied and rng.random() < 0.7:
        sq = rng.choice(occupied)
    elif empty:
        sq = rng.choice(empty)
    else:
        sq = rng.randint(0, 63)
    piece = board.piece_at(sq)
    if piece:
        color = "white" if piece.color == chess.WHITE else "black"
        answer = f"{color} {_PIECE_NAMES[piece.piece_type]}"
    else:
        answer = "empty"
    q_variants = [
        f"What piece is on {_sq(sq)}?",
        f"What occupies {_sq(sq)}?",
        f"Name the piece on square {_sq(sq)}.",
    ]
    return rng.choice(q_variants), answer


def _task_side_to_move(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    answer = "white" if board.turn == chess.WHITE else "black"
    q_variants = [
        "Which side is to move?",
        "Whose turn is it?",
        "Which color moves next?",
        "Who has the next move?",
    ]
    return rng.choice(q_variants), answer


def _task_piece_list_abbr(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    parts = []
    for sq in chess.SQUARES:  # rank 1→8, matching CNN output order
        piece = board.piece_at(sq)
        if piece:
            abbr = _PIECE_ABBR[piece.piece_type]
            if piece.color == chess.BLACK:
                abbr = abbr.lower()
            parts.append(f"{abbr}{_sq(sq)}")
    answer = " ".join(parts) if parts else "none"
    q_variants = [
        "List all pieces on the board using abbreviations (e.g. Ra1, nb8). One per space.",
        "List every piece using standard abbreviation and square. Uppercase = white, lowercase = black.",
        "Give all pieces and their squares using abbreviations.",
    ]
    return rng.choice(q_variants), answer


def _task_piece_list_full(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    white_parts, black_parts = [], []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            name = _PIECE_NAMES[piece.piece_type]
            entry = f"{name} on {_sq(sq)}"
            if piece.color == chess.WHITE:
                white_parts.append(entry)
            else:
                black_parts.append(entry)
    lines = []
    if white_parts:
        lines.append("White: " + ", ".join(white_parts))
    if black_parts:
        lines.append("Black: " + ", ".join(black_parts))
    answer = "\n".join(lines) if lines else "none"
    q_variants = [
        "List all pieces by full name and square.",
        "Describe all pieces on the board with their full names and squares.",
        "Name every piece on the board and its square.",
    ]
    return rng.choice(q_variants), answer


def _task_rank_contents(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    rank = rng.randint(0, 7)
    rank_name = str(rank + 1)
    parts = []
    for file in range(8):
        sq = chess.square(file, rank)
        piece = board.piece_at(sq)
        if piece:
            abbr = _PIECE_ABBR[piece.piece_type]
            if piece.color == chess.BLACK:
                abbr = abbr.lower()
            parts.append(f"{abbr}{_sq(sq)}")
    answer = " ".join(parts) if parts else "empty"
    q_variants = [
        f"What pieces are on rank {rank_name}?",
        f"List all pieces on the {rank_name}{'st' if rank_name == '1' else 'nd' if rank_name == '2' else 'rd' if rank_name == '3' else 'th'} rank.",
        f"Name every piece on rank {rank_name} using abbreviations.",
    ]
    return rng.choice(q_variants), answer


def _task_file_contents(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    file = rng.randint(0, 7)
    file_name = chess.FILE_NAMES[file]
    parts = []
    for rank in range(8):
        sq = chess.square(file, rank)
        piece = board.piece_at(sq)
        if piece:
            abbr = _PIECE_ABBR[piece.piece_type]
            if piece.color == chess.BLACK:
                abbr = abbr.lower()
            parts.append(f"{abbr}{_sq(sq)}")
    answer = " ".join(parts) if parts else "empty"
    q_variants = [
        f"What pieces are on the {file_name} file?",
        f"List all pieces on file {file_name.upper()}.",
        f"Name every piece on the {file_name}-file using abbreviations.",
    ]
    return rng.choice(q_variants), answer


def _task_castling_rights(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    parts = []
    if board.has_kingside_castling_rights(chess.WHITE):
        parts.append("White O-O")
    if board.has_queenside_castling_rights(chess.WHITE):
        parts.append("White O-O-O")
    if board.has_kingside_castling_rights(chess.BLACK):
        parts.append("Black O-O")
    if board.has_queenside_castling_rights(chess.BLACK):
        parts.append("Black O-O-O")
    answer = ", ".join(parts) if parts else "none"
    q_variants = [
        "What castling rights remain?",
        "Which castling moves are still available?",
        "List the remaining castling rights.",
    ]
    return rng.choice(q_variants), answer


def _task_en_passant(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    answer = _sq(board.ep_square) if board.ep_square is not None else "none"
    q_variants = [
        "What is the en passant square, if any?",
        "Is there an en passant target square? If so, which?",
        "Give the en passant square or 'none'.",
    ]
    return rng.choice(q_variants), answer


def _task_move_number(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    answer = str(board.fullmove_number)
    q_variants = [
        "What is the current full move number?",
        "Which move number is this?",
        "What move number is the game on?",
    ]
    return rng.choice(q_variants), answer


def _task_count_piece_type(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    color = rng.choice([chess.WHITE, chess.BLACK])
    piece_type = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    count = len(board.pieces(piece_type, color))
    color_str = "white" if color == chess.WHITE else "black"
    name_str = _PIECE_NAMES[piece_type] + ("s" if count != 1 else "")
    q_variants = [
        f"How many {color_str} {name_str} are on the board?",
        f"Count the number of {color_str} {name_str}.",
    ]
    return rng.choice(q_variants), str(count)


def _task_count_total_material(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    color = rng.choice([chess.WHITE, chess.BLACK])
    count = len(
        board.pieces(chess.PAWN, color)
        | board.pieces(chess.KNIGHT, color)
        | board.pieces(chess.BISHOP, color)
        | board.pieces(chess.ROOK, color)
        | board.pieces(chess.QUEEN, color)
        | board.pieces(chess.KING, color)
    )
    color_name = "White" if color == chess.WHITE else "Black"
    answer = str(count)
    q_variants = [
        f"How many pieces does {color_name} have in total?",
        f"Count all {color_name.lower()} pieces on the board.",
        f"What is the total number of {color_name.lower()} pieces?",
    ]
    return rng.choice(q_variants), answer


def _task_find_piece_type(board: chess.Board, rng: random.Random) -> tuple[str, str]:
    color = rng.choice([chess.WHITE, chess.BLACK])
    piece_type = rng.choice(
        [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    )
    squares = sorted(board.pieces(piece_type, color))  # sorted by square index = a1→h8
    color_str = "white" if color == chess.WHITE else "black"
    name_str = _PIECE_NAMES[piece_type] + "s"
    answer = ", ".join([_sq(sq) for sq in squares]) if squares else "none"
    q_variants = [
        f"List all squares occupied by {color_str} {name_str}.",
        f"Where are the {color_str} {name_str} located?",
    ]
    return rng.choice(q_variants), answer


def _task_is_square_occupied(board: chess.Board, rng: random.Random) -> tuple[str, str] | None:
    occupied_squares = list(board.piece_map().keys())
    empty_squares = [sq for sq in chess.SQUARES if sq not in occupied_squares]
    # Strict 50/50 to prevent model from learning to always guess yes/no
    if rng.random() < 0.5 and occupied_squares:
        sq = rng.choice(occupied_squares)
        answer = "yes"
    elif empty_squares:
        sq = rng.choice(empty_squares)
        answer = "no"
    else:
        return None
    q_variants = [
        f"Is square {_sq(sq)} occupied by any piece?",
        f"Is there a piece on {_sq(sq)}?",
    ]
    return rng.choice(q_variants), answer


_TASKS = [
    ("piece_abbr_at", _task_piece_abbr_at),
    ("piece_name_at", _task_piece_name_at),
    ("side_to_move", _task_side_to_move),
    ("piece_list_abbr", _task_piece_list_abbr),
    ("piece_list_full", _task_piece_list_full),
    ("rank_contents", _task_rank_contents),
    ("file_contents", _task_file_contents),
    ("castling_rights", _task_castling_rights),
    ("en_passant", _task_en_passant),
    ("move_number", _task_move_number),
    ("count_piece_type", _task_count_piece_type),
    ("count_total_material", _task_count_total_material),
    ("find_piece_type", _task_find_piece_type),
    ("is_square_occupied", _task_is_square_occupied),
]


def generate_for_fen(fen: str, rng: random.Random, tasks_per_fen: int = 10) -> list[dict]:
    try:
        board = chess.Board(fen)
    except Exception:
        return []

    records = []
    task_pool = list(_TASKS)
    rng.shuffle(task_pool)

    for task_name, task_fn in task_pool[:tasks_per_fen]:
        result = task_fn(board, rng)
        if result is None:
            continue
        question, answer = result
        records.append(
            {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {"fen": fen, "task": task_name},
            }
        )
    return records


def _collect_fens(source: str, max_fens: int | None = None, from_end: bool = False) -> list[str]:
    """Extract unique FENs from a JSONL source file.

    Supports two formats:
      - grpo/board-reading format: {"metadata": {"fen": ...}, ...}
      - encoder pretrain format:   {"fen": ..., "move_uci": ..., ...}

    Args:
        from_end: If True, read from the end of the file (tail) rather than
                  the beginning. Useful for sourcing positions the encoder has
                  never seen (encoder pretrain reads from the start).
                  Reads a 2× oversampling window via seek to avoid loading the
                  entire file into memory.
    """
    fens: list[str] = []
    seen: set[str] = set()

    if from_end and max_fens:
        # Estimate bytes needed: each JSONL line is ~80 bytes on average.
        # Read 4× max_fens lines worth from the tail to safely get max_fens unique FENs.
        chunk_bytes = max_fens * 80 * 4
        with open(source, "rb") as fb:
            fb.seek(0, 2)
            file_size = fb.tell()
            seek_pos = max(0, file_size - chunk_bytes)
            fb.seek(seek_pos)
            if seek_pos > 0:
                fb.readline()  # skip partial first line
            raw = fb.read()
        lines_iter = (line.decode("utf-8", errors="replace") for line in raw.splitlines())
    else:
        lines_iter = open(source)  # type: ignore[assignment]

    try:
        for line in lines_iter:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fen = rec.get("metadata", {}).get("fen", "") or rec.get("fen", "")
                if fen and fen not in seen:
                    seen.add(fen)
                    fens.append(fen)
            except Exception:
                pass
            if max_fens and len(fens) >= max_fens:
                break
    finally:
        if not from_end:
            lines_iter.close()  # type: ignore[union-attr]
    return fens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/processed/encoder_pretrain_1b.jsonl",
        help="Source JSONL with FENs (encoder pretrain or grpo format)",
    )
    parser.add_argument("--output", default="data/processed/alignment_board_description.jsonl")
    parser.add_argument(
        "--eval-output", default="data/processed/alignment_board_description_eval.jsonl"
    )
    parser.add_argument(
        "--tasks-per-fen",
        type=int,
        default=1,
        help="Tasks per FEN (default 1 — with 1M FENs gives ~1M records)",
    )
    parser.add_argument("--eval-fens", type=int, default=10000)
    parser.add_argument(
        "--max-fens",
        type=int,
        default=1000000,
        help="Max unique FENs to sample (default 1M → ~1M records at 1 task/FEN)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--from-end",
        action="store_true",
        default=True,
        help="Read FENs from the tail of the source file (unseen positions). Default: True.",
    )
    parser.add_argument(
        "--from-start",
        dest="from_end",
        action="store_false",
        help="Read FENs from the start of the source file instead.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Collect unique FENs from source
    fens = _collect_fens(args.source, max_fens=args.max_fens, from_end=args.from_end)
    _logger.info(
        "Unique FENs collected: %d (max_fens=%s, from_end=%s)",
        len(fens),
        args.max_fens,
        args.from_end,
    )

    rng.shuffle(fens)

    eval_fens = set(fens[: args.eval_fens])
    train_fens = fens[args.eval_fens :]

    def _write(path: str, fen_list: list[str]) -> int:
        count = 0
        with open(path, "w") as f:
            for fen in fen_list:
                for rec in generate_for_fen(fen, rng, args.tasks_per_fen):
                    f.write(json.dumps(rec) + "\n")
                    count += 1
        return count

    n_train = _write(args.output, train_fens)
    n_eval = _write(args.eval_output, list(eval_fens))
    _logger.info("Train: %d records → %s", n_train, args.output)
    _logger.info("Eval:  %d records → %s", n_eval, args.eval_output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
