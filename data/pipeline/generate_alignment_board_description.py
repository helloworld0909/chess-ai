"""Generate board-description alignment data from FENs.

LLaVA-style alignment: simple, direct board reading tasks with no reasoning.
Each sample: user = <65 sentinels> + short question, assistant = short factual answer.

Two data sources:
  --source     (default: encoder_pretrain_1b.jsonl)  — main FENs for all tasks
  --sf15-source (default: encoder_pretrain_sf15.jsonl) — SF15-annotated FENs for
                eval_tier and sf15_dominant tasks ONLY (read from tail = unseen)

Task types (diverse descriptions of the same board):
  Per-square tasks (probe per-square CNN tokens):
  1.  piece_abbr_at        — "What piece abbreviation is on e4?" → "P" / "empty"
  2.  piece_name_at        — "What piece is on e4?" → "white pawn" / "empty"
  3.  rank_contents        — "What pieces are on rank 1?" → "Ra1 Nb1 ..."
  4.  file_contents        — "What pieces are on the e file?" → "Pe2 pe7"
  5.  count_piece_type     — "How many white pawns are on the board?" → "5"
  6.  count_total_material — "How many pieces does Black have in total?" → "14"
  7.  find_piece_type      — "List all squares occupied by white knights." → "b1, g1" / "none"
  8.  is_square_occupied   — "Is there a piece on e4?" → "yes" / "no"  (strict 50/50)
  9.  attackers_at         — "Which white pieces attack e5?" → "Ne4 Re1" / "none"
  10. is_pinned            — "Is the piece on f3 pinned?" → "yes" / "no"
  11. mobility_at          — "How many squares does the piece on d5 attack?" → "8"

  Global token tasks (probe the 65th summary CNN token):
  12. side_to_move         — "Which side is to move?" → "white" / "black"
  13. castling_rights      — "What are the castling rights?" → "White O-O, Black O-O-O" / "none"
  14. en_passant           — "Is there an en passant square?" → "e6" / "none"
  15. move_number          — "What is the current move number?" → "5"
  16. is_check             — "Is the king in check?" → "yes" / "no"
  17. material_balance     — "What is the material balance?" → "+3" / "0" / "-5" (white minus black)
  18. who_is_better        — "Which side has the material advantage?" → "white" / "black" / "equal"

  SF15-dependent global token tasks (require --sf15-source with eval_score/sf15_terms):
  19. eval_tier            — "Assess the position" → "White has a slight advantage"
  20. sf15_dominant        — "Which factor most favours one side?" → "Mobility favours white"

Output JSONL fields:
  {"messages": [...], "metadata": {"fen": ..., "task": ...}}
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
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


def _task_piece_abbr_at(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
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


def _task_piece_name_at(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
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


def _task_side_to_move(board: chess.Board, rng: random.Random, **_) -> tuple[str, str]:
    answer = "white" if board.turn == chess.WHITE else "black"
    q_variants = [
        "Which side is to move?",
        "Whose turn is it?",
        "Which color moves next?",
        "Who has the next move?",
    ]
    return rng.choice(q_variants), answer


def _task_rank_contents(board: chess.Board, rng: random.Random, **_) -> tuple[str, str]:
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


def _task_file_contents(board: chess.Board, rng: random.Random, **_) -> tuple[str, str]:
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


def _task_castling_rights(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    parts = []
    if board.has_kingside_castling_rights(chess.WHITE):
        parts.append("White O-O")
    if board.has_queenside_castling_rights(chess.WHITE):
        parts.append("White O-O-O")
    if board.has_kingside_castling_rights(chess.BLACK):
        parts.append("Black O-O")
    if board.has_queenside_castling_rights(chess.BLACK):
        parts.append("Black O-O-O")
    # "none" is ~60% of positions — skip half to prevent always-none bias
    if not parts and rng.random() < 0.5:
        return None
    answer = ", ".join(parts) if parts else "none"
    q_variants = [
        "What castling rights remain?",
        "Which castling moves are still available?",
        "List the remaining castling rights.",
    ]
    return rng.choice(q_variants), answer


def _task_en_passant(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    # En passant is available <1% of random positions — skip 50% of "none" cases
    # so the model sees roughly equal yes/no examples and can't cheat by always
    # outputting "none".
    has_ep = board.ep_square is not None
    if not has_ep and rng.random() < 0.5:
        return None  # skip — will be replaced by another task
    answer = _sq(board.ep_square) if has_ep else "none"  # type: ignore[arg-type]
    q_variants = [
        "What is the en passant square, if any?",
        "Is there an en passant target square? If so, which?",
        "Give the en passant square or 'none'.",
    ]
    return rng.choice(q_variants), answer


def _task_move_number(board: chess.Board, rng: random.Random, **_) -> tuple[str, str]:
    answer = str(board.fullmove_number)
    q_variants = [
        "What is the current full move number?",
        "Which move number is this?",
        "What move number is the game on?",
    ]
    return rng.choice(q_variants), answer


def _task_count_piece_type(board: chess.Board, rng: random.Random, **_) -> tuple[str, str]:
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


def _task_count_total_material(board: chess.Board, rng: random.Random, **_) -> tuple[str, str]:
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


def _task_find_piece_type(board: chess.Board, rng: random.Random, **_) -> tuple[str, str]:
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


def _task_is_square_occupied(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
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


# ---------------------------------------------------------------------------
# Global token tasks — probe the 65th summary CNN token
#
# The global token is CLIP-trained against text encoding:
#   - Eval tier (7 levels from SF15 eval_score)
#   - Top-3 SF15 term imbalances (Material, Pawns, Mobility, King safety, etc.)
#   - Board state: side to move, check, castling, en passant, material summary
#
# Tasks here ask about exactly that information.
# SF15-dependent tasks (eval_tier, sf15_dominant) take optional sf15_terms/eval_score;
# they are skipped (return None) when not available.
# ---------------------------------------------------------------------------

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

_SF15_TERMS = [
    "Material",
    "Imbalance",
    "Pawns",
    "Knights",
    "Bishops",
    "Rooks",
    "Queens",
    "Mobility",
    "King safety",
    "Threats",
    "Passed",
    "Space",
    "Winnable",
]

_EVAL_TIERS = [
    (-1e9, -2.5, "Black is winning decisively"),
    (-2.5, -1.0, "Black has a clear advantage"),
    (-1.0, -0.3, "Black has a slight advantage"),
    (-0.3, +0.3, "approximately equal"),
    (+0.3, +1.0, "White has a slight advantage"),
    (+1.0, +2.5, "White has a clear advantage"),
    (+2.5, +1e9, "White is winning decisively"),
]


def _eval_tier(eval_score: float) -> str:
    for lo, hi, label in _EVAL_TIERS:
        if lo <= eval_score < hi:
            return label
    return _EVAL_TIERS[-1][2]


def _task_is_check(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    in_check = board.is_check()
    # Checks are rare (~5% of positions) — skip 50% of non-check cases
    if not in_check and rng.random() < 0.5:
        return None
    answer = "yes" if in_check else "no"
    q_variants = [
        "Is the king in check?",
        "Is the side to move currently in check?",
        "Is there a check on the board?",
    ]
    return rng.choice(q_variants), answer


def _task_material_balance(board: chess.Board, rng: random.Random, **_) -> tuple[str, str]:
    white_mat = sum(
        _PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    black_mat = sum(
        _PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    balance = white_mat - black_mat
    answer = f"+{balance}" if balance > 0 else str(balance)
    q_variants = [
        "What is the material balance? Give the score as white minus black (e.g. +3, 0, -5). Pawns=1, knights/bishops=3, rooks=5, queens=9.",
        "Calculate the material balance (white minus black). Pawns=1, minor pieces=3, rooks=5, queens=9.",
        "What is the material difference? Positive means white is ahead. Pawns=1, knights/bishops=3, rooks=5, queens=9.",
    ]
    return rng.choice(q_variants), answer


def _task_who_is_better(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """Which side has the material advantage? → "white" / "black" / "equal".

    Balanced ~1/3 each: equal positions are ~40% of random games, so skip
    some to prevent the model from learning to guess "equal" by default.
    """
    white_mat = sum(
        _PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    black_mat = sum(
        _PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    diff = white_mat - black_mat
    if diff > 0:
        answer = "white"
    elif diff < 0:
        answer = "black"
    else:
        if rng.random() < 0.5:
            return None
        answer = "equal"
    q_variants = [
        "Which side has the material advantage?",
        "Who has more material on the board?",
        "Which side is ahead in material?",
    ]
    return rng.choice(q_variants), answer


def _task_attackers_at(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """Which pieces of one color attack a given square?

    Directly mirrors the attacker info encoded in describe_square() CLIP anchors.
    Ask about white or black attackers with equal probability.
    """
    sq = rng.choice(chess.SQUARES)
    sq_name = _sq(sq)
    color = rng.choice([chess.WHITE, chess.BLACK])
    color_str = "white" if color == chess.WHITE else "black"
    attackers = sorted(board.attackers(color, sq))

    if not attackers and rng.random() < 0.5:
        return None  # skip uninteresting empty-attacker cases half the time

    if attackers:
        parts = []
        for asq in attackers[:4]:  # cap at 4 for brevity
            piece = board.piece_at(asq)
            if piece:
                abbr = _PIECE_ABBR[piece.piece_type]
                if piece.color == chess.BLACK:
                    abbr = abbr.lower()
                parts.append(f"{abbr}{_sq(asq)}")
        answer = " ".join(parts)
    else:
        answer = "none"

    q_variants = [
        f"Which {color_str} pieces attack {sq_name}?",
        f"List all {color_str} pieces that attack square {sq_name}.",
        f"What {color_str} attackers target {sq_name}? Use abbreviations.",
    ]
    return rng.choice(q_variants), answer


def _task_is_pinned(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """Is the piece on a given square absolutely pinned to its king?

    Pins are rare — skip 70% of non-pinned cases to keep a balanced yes/no ratio.
    """
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    if not occupied:
        return None
    sq = rng.choice(occupied)
    piece = board.piece_at(sq)
    assert piece is not None
    pinned = board.is_pinned(piece.color, sq)
    if not pinned and rng.random() < 0.7:
        return None
    answer = "yes" if pinned else "no"
    q_variants = [
        f"Is the piece on {_sq(sq)} absolutely pinned to its king?",
        f"Is the piece on {_sq(sq)} pinned?",
        f"Can the piece on {_sq(sq)} move freely, or is it pinned to the king?",
    ]
    return rng.choice(q_variants), answer


def _task_mobility_at(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """How many squares does a piece control/attack from its square?

    Directly mirrors the 'controls N squares' info in describe_square() CLIP anchors.
    """
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    if not occupied:
        return None
    sq = rng.choice(occupied)
    mobility = len(board.attacks(sq))
    answer = str(mobility)
    q_variants = [
        f"How many squares does the piece on {_sq(sq)} attack or control?",
        f"What is the mobility of the piece on {_sq(sq)}? Count the squares it attacks.",
        f"How many squares can the piece on {_sq(sq)} attack from its current position?",
    ]
    return rng.choice(q_variants), answer


def _task_eval_tier(
    board: chess.Board,  # noqa: ARG001
    rng: random.Random,
    eval_score: float | None = None,
    **_,
) -> tuple[str, str] | None:
    """Overall position assessment from SF15 eval_score → 7-tier label."""
    if eval_score is None:
        return None
    answer = _eval_tier(eval_score)
    q_variants = [
        "How would you assess this position overall?",
        "What is the overall evaluation of this position?",
        "Assess the position: who stands better and by how much?",
    ]
    return rng.choice(q_variants), answer


def _task_sf15_dominant(
    board: chess.Board, rng: random.Random, sf15_terms: list[float] | None = None, **_
) -> tuple[str, str] | None:
    """Which SF15 factor most favours one side? → e.g. "Mobility favours white"."""
    if sf15_terms is None:
        return None
    ranked = sorted(zip(_SF15_TERMS, sf15_terms), key=lambda x: abs(x[1]), reverse=True)
    name, val = ranked[0]
    if abs(val) < 0.1:
        return None  # no significant imbalance
    side = "white" if val > 0 else "black"
    answer = f"{name} favours {side}"
    q_variants = [
        "Which positional factor most favours one side?",
        "What is the dominant positional imbalance in this position?",
        "Which factor gives one side the biggest advantage?",
    ]
    return rng.choice(q_variants), answer


_TASKS = [
    ("piece_abbr_at", _task_piece_abbr_at),
    ("piece_name_at", _task_piece_name_at),
    ("side_to_move", _task_side_to_move),
    ("rank_contents", _task_rank_contents),
    ("file_contents", _task_file_contents),
    ("castling_rights", _task_castling_rights),
    ("en_passant", _task_en_passant),
    ("move_number", _task_move_number),
    ("count_piece_type", _task_count_piece_type),
    ("count_total_material", _task_count_total_material),
    ("find_piece_type", _task_find_piece_type),
    ("is_square_occupied", _task_is_square_occupied),
    # Per-square tasks — probe structural info encoded in CLIP per-square anchors
    ("attackers_at", _task_attackers_at),
    ("is_pinned", _task_is_pinned),
    ("mobility_at", _task_mobility_at),
    # Global token tasks (FEN-only)
    ("is_check", _task_is_check),
    ("material_balance", _task_material_balance),
    ("who_is_better", _task_who_is_better),
    # Global token tasks (require SF15 data — skipped when not provided)
    ("eval_tier", _task_eval_tier),
    ("sf15_dominant", _task_sf15_dominant),
]

# Tasks that need a top-up pass because their positive examples are rare in random positions.
# eval_tier and sf15_dominant are also rare — they require SF15-annotated FENs from --sf15-source.
_RARE_TASKS = {
    "castling_rights",
    "en_passant",
    "is_check",
    "is_pinned",
    "eval_tier",
    "sf15_dominant",
}


def generate_for_fen(
    fen: str,
    rng: random.Random,
    tasks_per_fen: int = 10,
    sf15_terms: list[float] | None = None,
    eval_score: float | None = None,
) -> list[dict]:
    try:
        board = chess.Board(fen)
    except Exception:
        return []

    records = []
    task_pool = list(_TASKS)
    rng.shuffle(task_pool)

    for task_name, task_fn in task_pool[:tasks_per_fen]:
        result = task_fn(board, rng, sf15_terms=sf15_terms, eval_score=eval_score)
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


def _collect_fens(
    source: str, max_fens: int | None = None, from_end: bool = False
) -> tuple[list[str], dict[str, dict]]:
    """Extract unique FENs (and optional SF15 data) from a JSONL source file.

    Supports two formats:
      - grpo/board-reading format: {"metadata": {"fen": ...}, ...}
      - encoder pretrain format:   {"fen": ..., "sf15_terms": [...], "eval_score": float}

    Returns:
        (fens, sf15_by_fen) where sf15_by_fen maps fen → {"sf15_terms": [...], "eval_score": float}
        sf15_by_fen is empty if the source does not contain SF15 data.

    Args:
        from_end: If True, read from the end of the file (tail) rather than
                  the beginning. Useful for sourcing positions the encoder has
                  never seen (encoder pretrain reads from the start).
    """
    fens: list[str] = []
    sf15_by_fen: dict[str, dict] = {}
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
                    if "sf15_terms" in rec and "eval_score" in rec:
                        sf15_by_fen[fen] = {
                            "sf15_terms": rec["sf15_terms"],
                            "eval_score": rec["eval_score"],
                        }
            except Exception:
                pass
            if max_fens and len(fens) >= max_fens:
                break
    finally:
        if not from_end:
            lines_iter.close()  # type: ignore[union-attr]
    return fens, sf15_by_fen


def _collect_rare_fens(
    source: str,
    predicate,
    max_fens: int,
    exclude: set[str] | None = None,
) -> list[str]:
    """Scan the full source file for FENs matching a predicate.

    Used to collect enough positive examples for rare board states
    (e.g. en passant available, castling rights present) that are
    underrepresented in any fixed tail window.
    """
    fens: list[str] = []
    seen: set[str] = set(exclude or [])
    with open(source) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fen = rec.get("metadata", {}).get("fen", "") or rec.get("fen", "")
                if not fen or fen in seen:
                    continue
                board = chess.Board(fen)
                if predicate(board):
                    seen.add(fen)
                    fens.append(fen)
                    if len(fens) >= max_fens:
                        break
            except Exception:
                pass
    _logger.info("Rare FENs collected: %d (target=%d)", len(fens), max_fens)
    return fens


def _worker(args: tuple) -> list[dict]:
    """Top-level function for multiprocessing (must be picklable)."""
    fen, seed, tasks_per_fen, sf15_terms, eval_score = args
    return generate_for_fen(
        fen, random.Random(seed), tasks_per_fen, sf15_terms=sf15_terms, eval_score=eval_score
    )


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
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Parallel workers for FEN processing (default: all CPUs)",
    )
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
    parser.add_argument(
        "--sf15-source",
        default="data/processed/encoder_pretrain_sf15.jsonl",
        help="Source JSONL with SF15 annotations (eval_score, sf15_terms). Used ONLY for "
        "eval_tier and sf15_dominant tasks. Read from tail (unseen positions).",
    )
    parser.add_argument(
        "--sf15-fens",
        type=int,
        default=200000,
        help="Max SF15-annotated FENs to collect for eval_tier/sf15_dominant tasks.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Collect unique FENs from source (tail = unseen by encoder)
    fens, sf15_by_fen = _collect_fens(args.source, max_fens=args.max_fens, from_end=args.from_end)
    _logger.info(
        "Unique FENs collected: %d (max_fens=%s, from_end=%s, sf15_coverage=%d)",
        len(fens),
        args.max_fens,
        args.from_end,
        len(sf15_by_fen),
    )

    # Collect extra FENs from full dataset for rare board states.
    # Target: enough positives so that after trimming nones to 30%, each rare
    # task has ~5000 examples (5000 * 0.7 positives needed = 3500 each).
    RARE_TARGET_POSITIVES = 3500
    fens_set = set(fens)
    _logger.info("Scanning full dataset for en_passant FENs (target=%d)...", RARE_TARGET_POSITIVES)
    ep_fens = _collect_rare_fens(
        args.source,
        predicate=lambda b: b.ep_square is not None,
        max_fens=RARE_TARGET_POSITIVES,
        exclude=fens_set,
    )
    _logger.info("Scanning full dataset for castling FENs (target=%d)...", RARE_TARGET_POSITIVES)
    castling_fens = _collect_rare_fens(
        args.source,
        predicate=lambda b: b.has_castling_rights(chess.WHITE)
        or b.has_castling_rights(chess.BLACK),
        max_fens=RARE_TARGET_POSITIVES,
        exclude=fens_set | set(ep_fens),
    )
    _logger.info("Scanning full dataset for is_check FENs (target=%d)...", RARE_TARGET_POSITIVES)
    check_fens = _collect_rare_fens(
        args.source,
        predicate=lambda b: b.is_check(),
        max_fens=RARE_TARGET_POSITIVES,
        exclude=fens_set | set(ep_fens) | set(castling_fens),
    )
    _logger.info("Scanning full dataset for is_pinned FENs (target=%d)...", RARE_TARGET_POSITIVES)
    pinned_fens = _collect_rare_fens(
        args.source,
        predicate=lambda b: any(b.is_pinned(b.piece_at(sq).color, sq) for sq in b.piece_map()),
        max_fens=RARE_TARGET_POSITIVES,
        exclude=fens_set | set(ep_fens) | set(castling_fens) | set(check_fens),
    )

    # Collect SF15-annotated FENs for eval_tier and sf15_dominant.
    # These come ONLY from --sf15-source (not the 1b source), read from the tail
    # so the encoder hasn't seen them during pretraining.
    _logger.info(
        "Collecting SF15 FENs from %s (max=%d, from_end=True)...",
        args.sf15_source,
        args.sf15_fens,
    )
    sf15_fens, sf15_extra = _collect_fens(args.sf15_source, max_fens=args.sf15_fens, from_end=True)
    # Merge SF15 metadata so generate_for_fen can access eval_score/sf15_terms
    sf15_by_fen.update(sf15_extra)
    _logger.info(
        "SF15 FENs collected: %d (with SF15 data: %d)",
        len(sf15_fens),
        len(sf15_extra),
    )

    # Rare FENs are kept separate — only used for their respective tasks
    rare_fens_by_task = {
        "en_passant": ep_fens,
        "castling_rights": castling_fens,
        "is_check": check_fens,
        "is_pinned": pinned_fens,
        # SF15 tasks: use only SF15-source FENs (have eval_score + sf15_terms)
        "eval_tier": [f for f in sf15_fens if f in sf15_extra],
        "sf15_dominant": [f for f in sf15_fens if f in sf15_extra],
    }

    rng.shuffle(fens)

    def _write_split(train_path: str, eval_path: str, fen_list: list[str], rare_pool: dict[str, list[str]], target_eval_records: int) -> tuple[int, int]:
        """Generate data and split into train/eval to match target eval size using RNG."""
        # Estimate eval_ratio needed to get target eval records
        # With N FENs * 1 task/fen ≈ N records, we want eval_ratio = target_eval_records / N
        estimated_records = len(fen_list)
        eval_ratio = min(0.5, max(0.001, target_eval_records / (estimated_records + target_eval_records)))
        
        task_counts_train: dict[str, int] = {name: 0 for name, _ in _TASKS}
        task_counts_eval: dict[str, int] = {name: 0 for name, _ in _TASKS}
        records_train: list[dict] = []
        records_eval: list[dict] = []
        
        split_rng = random.Random(rng.randint(0, 2**32))

        # Main pass: parallel FEN processing
        worker_args = [
            (
                fen,
                rng.randint(0, 2**32),
                args.tasks_per_fen,
                sf15_by_fen.get(fen, {}).get("sf15_terms"),
                sf15_by_fen.get(fen, {}).get("eval_score"),
            )
            for fen in fen_list
        ]
        with multiprocessing.Pool(processes=args.workers) as pool:
            for records in pool.imap_unordered(_worker, worker_args, chunksize=2000):
                for rec in records:
                    t = rec["metadata"]["task"]
                    # Use RNG to decide train vs eval
                    if split_rng.random() < eval_ratio:
                        records_eval.append(rec)
                        task_counts_eval[t] += 1
                    else:
                        records_train.append(rec)
                        task_counts_train[t] += 1

        # Rare task pass: generate from dedicated rare FEN pools (positives guaranteed)
        MAX_NONE_RATIO = 0.30
        topup_rng = random.Random(rng.randint(0, 2**32))
        rare_fns = {name: fn for name, fn in _TASKS if name in _RARE_TASKS}

        for name, fn in rare_fns.items():
            pool = rare_pool.get(name, [])
            for fen in pool:
                try:
                    board = chess.Board(fen)
                except Exception:
                    continue
                sf15 = sf15_by_fen.get(fen, {})
                result = fn(
                    board,
                    topup_rng,
                    sf15_terms=sf15.get("sf15_terms"),
                    eval_score=sf15.get("eval_score"),
                )
                if result is None:
                    continue
                question, answer = result
                rec = {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {"fen": fen, "task": name},
                }
                # Split rare records 90/10
                if split_rng.random() < eval_ratio:
                    records_eval.append(rec)
                    task_counts_eval[name] += 1
                else:
                    records_train.append(rec)
                    task_counts_train[name] += 1

            # Trim nones to MAX_NONE_RATIO in each set
            train_positives = [r for r in records_train if r["metadata"]["task"] == name and r["messages"][1]["content"] != "none"]
            train_nones = [r for r in records_train if r["metadata"]["task"] == name and r["messages"][1]["content"] == "none"]
            eval_positives = [r for r in records_eval if r["metadata"]["task"] == name and r["messages"][1]["content"] != "none"]
            eval_nones = [r for r in records_eval if r["metadata"]["task"] == name and r["messages"][1]["content"] == "none"]
            
            max_train_none = int(len(train_positives) * MAX_NONE_RATIO / (1.0 - MAX_NONE_RATIO))
            max_eval_none = int(len(eval_positives) * MAX_NONE_RATIO / (1.0 - MAX_NONE_RATIO))
            topup_rng.shuffle(train_nones)
            topup_rng.shuffle(eval_nones)
            
            # Rebuild records lists without this task, then add trimmed version
            records_train = [r for r in records_train if r["metadata"]["task"] != name]
            records_eval = [r for r in records_eval if r["metadata"]["task"] != name]
            
            records_train.extend(train_positives + train_nones[:max_train_none])
            records_eval.extend(eval_positives + eval_nones[:max_eval_none])
            
            task_counts_train[name] = len([r for r in records_train if r["metadata"]["task"] == name])
            task_counts_eval[name] = len([r for r in records_eval if r["metadata"]["task"] == name])
            
            _logger.info(
                "  rare task %-20s train: pos=%d total=%d | eval: pos=%d total=%d",
                name,
                len(train_positives),
                task_counts_train[name],
                len(eval_positives),
                task_counts_eval[name],
            )

        _logger.info("Train task counts:")
        for name, cnt in sorted(task_counts_train.items()):
            _logger.info("  task %-25s %d", name, cnt)

        _logger.info("Eval task counts:")
        for name, cnt in sorted(task_counts_eval.items()):
            _logger.info("  task %-25s %d", name, cnt)

        # Write both files
        split_rng.shuffle(records_train)
        split_rng.shuffle(records_eval)
        
        with open(train_path, "w") as f:
            for rec in records_train:
                f.write(json.dumps(rec) + "\n")
        
        with open(eval_path, "w") as f:
            for rec in records_eval:
                f.write(json.dumps(rec) + "\n")
        
        return len(records_train), len(records_eval)

    n_train, n_eval = _write_split(args.output, args.eval_output, fens, rare_fens_by_task, target_eval_records=args.eval_fens * args.tasks_per_fen)
    _logger.info("Train: %d records → %s", n_train, args.output)
    _logger.info("Eval:  %d records → %s", n_eval, args.eval_output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
