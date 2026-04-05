"""Generate board-description alignment data from FENs.

Two difficulty tiers:

  EASY tasks — direct board reading, no reasoning required.
  Each answer is a single fact read directly from the board state or FEN.

  MEDIUM tasks — multi-step reasoning over the encoded position.
  Answers often in SAN notation (e.g. "Bxd5", "Nc6+") or derived pawn structure facts.
  Require combining information across multiple squares or computing legality.

Two data sources:
  --source     (default: encoder_pretrain_1b.jsonl)  — main FENs for all tasks
  --sf15-source (default: encoder_pretrain_sf15.jsonl) — SF15-annotated FENs for
                eval_tier and sf15_dominant tasks ONLY (read from tail = unseen)

Easy tasks:
  Per-square (probe per-square CNN tokens):
  1.  piece_abbr_at        — "What piece abbreviation is on e4?" → "P" / "empty"
  2.  piece_name_at        — "What piece is on e4?" → "white pawn" / "empty"
  3.  rank_contents        — "What pieces are on rank 1?" → "white rook on a1, ..." (3 rotating formats, prompt-instructed)
  4.  file_contents        — "What pieces are on the e file?" → same 3 formats as rank_contents
  5.  count_piece_type     — "How many white pawns are on the board?" → "5"
  6.  count_total_material — "How many pieces does Black have in total?" → "14"
  7.  find_piece_type      — "List all squares occupied by white knights." → "b1, g1" / "none"
  8.  is_square_occupied   — "Is there a piece on e4?" → "yes" / "no"  (strict 50/50)
  9.  attackers_at         — "Which white pieces attack e5?" → "knight on e4, rook on e1" / "none"
  10. is_pinned            — "Is the piece on f3 pinned?" → "yes" / "no"
  11. mobility_at          — "How many squares does the piece on d5 attack?" → "8"

  Global token (probe the 65th summary CNN token):
  12. side_to_move         — "Which side is to move?" → "white" / "black"
  13. castling_rights      — "What are the castling rights?" → "White O-O, Black O-O-O" / "none"
  14. en_passant           — "Is there an en passant square?" → "e6" / "none"
  15. move_number          — "What is the current move number?" → "5"
  16. is_check             — "Is the king in check?" → "yes" / "no"
  17. material_balance     — "What is the material balance?" → "+3" / "0" / "-5" (white minus black)
  18. who_is_better        — "Which side has the material advantage?" → "white" / "black" / "equal"

  SF15-dependent (require --sf15-source with eval_score/sf15_terms):
  19. eval_tier            — "Assess the position" → "White has a slight advantage"
  20. sf15_dominant        — "Which factor most favours one side?" → "Mobility favours white"

Medium tasks (multi-step reasoning, SAN or UCI notation, prompt-instructed format):
  21. hanging_pieces       — "Which pieces are hanging?" → "white pawn on d4, black knight on e6" / "none"
  22. capture_on_square    — "How can white capture on e5?" → SAN or UCI (50/50, prompt-instructed)
  23. give_check           — "Name a checking move." → SAN or UCI / "none"
  24. threaten_piece_with  — "How can the white rook on f1 threaten ...?" → SAN or UCI
  25. fork_move            — "Can white fork two pieces?" → SAN or UCI / "none"
  26. doubled_pawns        — "Does white have doubled pawns?" → "yes" / "no"  (50/50)
  27. isolated_pawn_at     — "Is the pawn on c3 isolated?" → "yes" / "no"  (50/50)
  28. passed_pawn          — "Does white have a passed pawn?" → "d5" / "none"
  29. checkmate_in_one     — "Is there a checkmate in one?" → SAN or UCI / "none"
  30. board_inventory      — "List all pieces on the board." → 3 rotating formats (prompt-instructed)

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
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
        "Which side is to move? (white/black)",
        "Whose turn is it? Answer 'white' or 'black'.",
        "Which color moves next?",
        "Who has the next move? Say 'white' or 'black'.",
        "Is it white's or black's turn?",
    ]
    return rng.choice(q_variants), answer


def _task_rank_contents(board: chess.Board, rng: random.Random, **_) -> tuple[str, str]:
    rank = rng.randint(0, 7)
    rank_name = str(rank + 1)
    ordinal = (
        "st"
        if rank_name == "1"
        else "nd"
        if rank_name == "2"
        else "rd"
        if rank_name == "3"
        else "th"
    )
    fmt = rng.randint(0, 2)
    parts = []
    for file in range(8):
        sq = chess.square(file, rank)
        piece = board.piece_at(sq)
        if piece:
            color_str = "white" if piece.color == chess.WHITE else "black"
            name = _PIECE_NAMES[piece.piece_type]
            sq_str = _sq(sq)
            if fmt == 0:
                # "white pawn on a2"
                parts.append(f"{color_str} {name} on {sq_str}")
            elif fmt == 1:
                # "Pa2" style — but instruct in question
                abbr = _PIECE_ABBR[piece.piece_type]
                if piece.color == chess.BLACK:
                    abbr = abbr.lower()
                parts.append(f"{abbr}:{sq_str}")
            else:
                # "a2(pawn,white)"
                parts.append(f"{sq_str}({name},{color_str})")
    answer = ", ".join(parts) if parts else "empty"
    if fmt == 0:
        q_variants = [
            f"What pieces are on rank {rank_name}? List as 'color piece on square', e.g. 'white pawn on a2, black rook on a8'.",
            f"List all pieces on the {rank_name}{ordinal} rank. Format each as 'color piece on square'.",
        ]
    elif fmt == 1:
        q_variants = [
            f"What pieces are on rank {rank_name}? Use format 'A:sq' where A is piece letter (uppercase=white, lowercase=black), e.g. 'P:a2, r:a8'.",
            f"List all pieces on the {rank_name}{ordinal} rank as 'Letter:square' pairs (uppercase=white, lowercase=black). E.g. 'P:a2, r:a8'.",
        ]
    else:
        q_variants = [
            f"What pieces are on rank {rank_name}? Format each as 'square(piece,color)', e.g. 'a2(pawn,white)'.",
            f"List all pieces on the {rank_name}{ordinal} rank in 'square(piece,color)' format.",
        ]
    return rng.choice(q_variants), answer


def _task_file_contents(board: chess.Board, rng: random.Random, **_) -> tuple[str, str]:
    file = rng.randint(0, 7)
    file_name = chess.FILE_NAMES[file]
    fmt = rng.randint(0, 2)
    parts = []
    for rank in range(8):
        sq = chess.square(file, rank)
        piece = board.piece_at(sq)
        if piece:
            color_str = "white" if piece.color == chess.WHITE else "black"
            name = _PIECE_NAMES[piece.piece_type]
            sq_str = _sq(sq)
            if fmt == 0:
                parts.append(f"{color_str} {name} on {sq_str}")
            elif fmt == 1:
                abbr = _PIECE_ABBR[piece.piece_type]
                if piece.color == chess.BLACK:
                    abbr = abbr.lower()
                parts.append(f"{abbr}:{sq_str}")
            else:
                parts.append(f"{sq_str}({name},{color_str})")
    answer = ", ".join(parts) if parts else "empty"
    if fmt == 0:
        q_variants = [
            f"What pieces are on the {file_name} file? List as 'color piece on square', e.g. 'white pawn on a2'.",
            f"List all pieces on file {file_name.upper()}. Format each as 'color piece on square'.",
        ]
    elif fmt == 1:
        q_variants = [
            f"What pieces are on the {file_name} file? Use format 'A:sq' (uppercase=white, lowercase=black), e.g. 'P:a2, r:a8'.",
            f"List all pieces on file {file_name.upper()} as 'Letter:square' pairs (uppercase=white, lowercase=black).",
        ]
    else:
        q_variants = [
            f"What pieces are on the {file_name} file? Format each as 'square(piece,color)', e.g. 'a2(pawn,white)'.",
            f"List all pieces on file {file_name.upper()} in 'square(piece,color)' format.",
        ]
    return rng.choice(q_variants), answer


def _task_castling_rights(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    wk = board.has_kingside_castling_rights(chess.WHITE)
    wq = board.has_queenside_castling_rights(chess.WHITE)
    bk = board.has_kingside_castling_rights(chess.BLACK)
    bq = board.has_queenside_castling_rights(chess.BLACK)
    # "none" is ~60% of positions — skip 80% to approach 50/50 balance
    if not any([wk, wq, bk, bq]) and rng.random() < 0.80:
        return None

    # Randomly vary answer format so the model follows the prompt rather than memorising one style
    fmt = rng.randint(0, 2)
    if fmt == 0:
        # Full names: "White O-O, Black O-O-O"
        parts = []
        if wk:
            parts.append("White O-O")
        if wq:
            parts.append("White O-O-O")
        if bk:
            parts.append("Black O-O")
        if bq:
            parts.append("Black O-O-O")
        answer = ", ".join(parts) if parts else "none"
        q_variants = [
            "What castling rights remain? List as e.g. 'White O-O, Black O-O-O' or 'none'.",
            "Which castling moves are still available? Use format 'White O-O' / 'White O-O-O' / 'Black O-O' / 'Black O-O-O'.",
            "List the remaining castling rights (e.g. White O-O, Black O-O).",
        ]
    elif fmt == 1:
        # FEN-style: "KQkq", "Kq", "-"
        fen_str = (
            ("K" if wk else "") + ("Q" if wq else "") + ("k" if bk else "") + ("q" if bq else "")
        )
        answer = fen_str if fen_str else "-"
        q_variants = [
            "What are the castling rights in FEN notation? (e.g. KQkq, Kq, -)",
            "Give the castling availability as in FEN (K=white kingside, Q=white queenside, k=black kingside, q=black queenside, - if none).",
            "FEN castling field: what is it? (KQkq format)",
        ]
    else:
        # Boolean breakdown
        parts = []
        if wk:
            parts.append("white can castle kingside")
        if wq:
            parts.append("white can castle queenside")
        if bk:
            parts.append("black can castle kingside")
        if bq:
            parts.append("black can castle queenside")
        answer = ", ".join(parts) if parts else "none"
        q_variants = [
            "Describe the remaining castling rights in plain English.",
            "Which sides can still castle, and on which flank?",
            "List who can still castle: e.g. 'white can castle kingside, black can castle queenside'.",
        ]
    return rng.choice(q_variants), answer


def _task_en_passant(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    # En passant is available <1% of random positions — skip 99% of "none" cases.
    # Positives come from the rare FEN pool, so this function mainly produces "none"
    # in the main pass; skip almost all to keep the model from seeing a flood of nones.
    has_ep = board.ep_square is not None
    if not has_ep and rng.random() < 0.99:
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
        f"Is square {_sq(sq)} occupied by any piece? (yes/no)",
        f"Is there a piece on {_sq(sq)}? Answer yes or no.",
        f"Is {_sq(sq)} occupied?",
        f"Answer 'yes' or 'no': is there a piece on {_sq(sq)}?",
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
    # Checks are rare (~5% of positions) — skip 95% of non-check cases for near-50/50 balance.
    # Positive (check) examples come from the is_check rare FEN pool.
    if not in_check and rng.random() < 0.95:
        return None
    answer = "yes" if in_check else "no"
    q_variants = [
        "Is the king in check? (yes/no)",
        "Is the side to move currently in check? Answer yes or no.",
        "Is there a check on the board?",
        "Answer 'yes' or 'no': is the king in check?",
        "Check status: is the side to move in check?",
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
        "Which side has the material advantage? (white/black/equal)",
        "Who has more material on the board? Answer 'white', 'black', or 'equal'.",
        "Which side is ahead in material?",
        "Material count: who is better off? Say 'white', 'black', or 'equal'.",
        "Is white, black, or neither ahead in material?",
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

    if not attackers and rng.random() < 0.75:
        return None  # skip most empty-attacker cases to reduce "none" bias

    if attackers:
        parts = []
        for asq in attackers[:4]:  # cap at 4 for brevity
            piece = board.piece_at(asq)
            if piece:
                parts.append(f"{_PIECE_NAMES[piece.piece_type]} on {_sq(asq)}")
        answer = ", ".join(parts)
    else:
        answer = "none"

    q_variants = [
        f"Which {color_str} pieces attack {sq_name}? List as 'piece on square', e.g. 'rook on a1, bishop on c3'.",
        f"List all {color_str} pieces that attack square {sq_name}. Format: 'piece on square'.",
        f"What {color_str} attackers target {sq_name}? Answer as 'piece on square' or 'none'.",
    ]
    return rng.choice(q_variants), answer


def _task_is_pinned(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """Is the piece on a given square absolutely pinned to its king?

    Pins are rare — skip 95% of non-pinned cases for near-50/50 balance.
    Positive examples come from the is_pinned rare FEN pool.
    """
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    if not occupied:
        return None
    sq = rng.choice(occupied)
    piece = board.piece_at(sq)
    assert piece is not None
    pinned = board.is_pinned(piece.color, sq)
    if not pinned and rng.random() < 0.95:
        return None
    answer = "yes" if pinned else "no"
    q_variants = [
        f"Is the piece on {_sq(sq)} absolutely pinned to its king? (yes/no)",
        f"Is the piece on {_sq(sq)} pinned? Answer yes or no.",
        f"Can the piece on {_sq(sq)} move freely, or is it pinned to the king?",
        f"Answer 'yes' or 'no': is the piece on {_sq(sq)} pinned?",
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
    """Overall position assessment from SF15 eval_score → text label or cp score.

    Two modes chosen randomly (50/50):
    - Text label: answer is one of 7 tier strings; question lists options in varied formats.
    - CP score: answer is eval_score rounded to nearest 50cp (e.g. +150, -250, 0);
                question asks for centipawn score with example range shown.
    """
    if eval_score is None:
        return None

    if rng.random() < 0.5:
        # --- CP score mode ---
        # Round to nearest 100cp, clamp to ±500 for extreme positions
        cp = int(round(eval_score * 100 / 100) * 100)
        cp = max(-500, min(500, cp))
        answer = f"+{cp}" if cp > 0 else str(cp)
        _cp_options = "-500 / -400 / -300 / -200 / -100 / 0 / +100 / +200 / +300 / +400 / +500"
        q_variants = [
            f"What is the evaluation in centipawns (rounded to nearest 100)? Options: {_cp_options}.",
            f"Give the position's centipawn score rounded to the nearest 100. Choose from: {_cp_options}.",
            f"Estimate the centipawn evaluation (nearest 100cp). Pick one: {_cp_options}.",
        ]
    else:
        # --- Text label mode ---
        answer = _eval_tier(eval_score)
        _option_styles = [
            # slash-separated
            (
                "Black is winning decisively / Black has a clear advantage / "
                "Black has a slight advantage / approximately equal / "
                "White has a slight advantage / White has a clear advantage / "
                "White is winning decisively"
            ),
            # comma-separated
            (
                "Black is winning decisively, Black has a clear advantage, "
                "Black has a slight advantage, approximately equal, "
                "White has a slight advantage, White has a clear advantage, "
                "White is winning decisively"
            ),
            # abbreviated
            "decisive black / clear black / slight black / equal / slight white / clear white / decisive white",
            # lettered
            (
                "(a) Black is winning decisively (b) Black has a clear advantage "
                "(c) Black has a slight advantage (d) approximately equal "
                "(e) White has a slight advantage (f) White has a clear advantage "
                "(g) White is winning decisively"
            ),
            # no options — model must recall labels from training
            None,
        ]
        options = rng.choice(_option_styles)
        if options:
            q_variants = [
                f"How would you assess this position overall? Options: {options}.",
                f"What is the overall evaluation of this position? Choose one: {options}.",
                f"Assess the position: who stands better and by how much? Pick from: {options}.",
                f"Rate the position. Choose: {options}.",
            ]
        else:
            q_variants = [
                "How would you assess this position overall?",
                "What is the overall evaluation of this position?",
                "Assess the position: who stands better and by how much?",
                "Give the position evaluation.",
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

    terms_list = ", ".join(_SF15_TERMS)
    example = "'Mobility favours white' or 'King safety favours black'"
    q_variants = [
        f"Which positional factor most favours one side? Terms: {terms_list}. Answer as {example}.",
        f"What is the dominant positional imbalance? Choose from: {terms_list}. Format: '<Term> favours <white/black>'.",
        f"Which factor gives one side the biggest advantage? ({example})",
        f"Which SF15 term is most imbalanced? Terms: {terms_list}. Answer: '<Term> favours white/black'.",
        f"Identify the strongest positional factor for either side. ({example})",
    ]
    return rng.choice(q_variants), answer


# ---------------------------------------------------------------------------
# Tactical tasks — probe relational/derived structure over encoder tokens
#
# These tasks require multi-step reasoning: identify pieces, compute attacks,
# find legal moves, and output answers in SAN notation. They bridge alignment
# (direct read) toward phase1 SFT (line generation).
#
# Move answers are randomly in SAN (e.g. Bxe5+) or UCI (e.g. f1e5) notation,
# chosen 50/50 per record. The question explicitly names the required notation so
# the model learns to follow the instruction rather than pick a format arbitrarily.
# ---------------------------------------------------------------------------


def _move_notation(board: chess.Board, move: chess.Move, rng: random.Random) -> tuple[str, str]:
    """Return (notation_label, move_string) chosen 50/50 between SAN and UCI.

    notation_label — "SAN (e.g. Bxe5)" or "UCI (e.g. f1e5)" — for embedding in the question.
    move_string    — the actual answer in that notation.
    """
    if rng.random() < 0.5:
        return "SAN (e.g. Bxe5)", board.san(move)
    else:
        return "UCI (e.g. f1e5)", move.uci()


def _task_hanging_pieces(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """Which pieces are hanging (undefended and under attack)?

    A piece is hanging if attacked by the opponent and not defended by any
    friendly piece. Kings are excluded (they can't be captured). Both sides'
    hanging pieces are listed. Skip 60% of "none" positions for balance.
    """
    hanging = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue
        opp_color = not piece.color
        if not board.attackers(opp_color, sq):
            continue
        if board.attackers(piece.color, sq):
            continue
        color_str = "white" if piece.color == chess.WHITE else "black"
        hanging.append(f"{color_str} {_PIECE_NAMES[piece.piece_type]} on {_sq(sq)}")

    if not hanging and rng.random() < 0.80:
        return None

    answer = ", ".join(hanging) if hanging else "none"
    q_variants = [
        "Which pieces are hanging (undefended and under attack)? List as 'color piece on square', e.g. 'white pawn on d4'.",
        "List all hanging pieces — attacked by the opponent and not defended. Format: 'color piece on square'.",
        "Which pieces can be captured for free? Answer as 'color piece on square' or 'none'.",
    ]
    return rng.choice(q_variants), answer


def _task_capture_on_square(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """How can the side to move capture a specific enemy piece?

    Picks a random capturable enemy square, returns SAN of the cheapest
    legal capture (lowest-value attacker first). Teaches Pxe5 / Bxd4 notation.
    """
    color = board.turn

    # Group legal captures by destination square
    captures_by_sq: dict[int, list[chess.Move]] = {}
    for move in board.legal_moves:
        if board.is_capture(move):
            captures_by_sq.setdefault(move.to_square, []).append(move)

    if not captures_by_sq:
        return None

    sq = rng.choice(list(captures_by_sq.keys()))
    target = board.piece_at(sq)
    if target is None:
        # En passant: target square is empty but capture is legal
        target_name = "pawn"
    else:
        target_name = _PIECE_NAMES[target.piece_type]

    captures = sorted(
        captures_by_sq[sq],
        key=lambda m: _PIECE_VALUES.get(
            p.piece_type if (p := board.piece_at(m.from_square)) else chess.PAWN, 0
        ),
    )
    move = captures[0]
    notation, answer = _move_notation(board, move, rng)

    color_str = "white" if color == chess.WHITE else "black"
    sq_name = _sq(sq)
    q_variants = [
        f"How can {color_str} capture the {target_name} on {sq_name}? Use {notation}.",
        f"Name a {color_str} move that takes on {sq_name}. Answer in {notation}.",
        f"Which {color_str} move captures the piece on {sq_name}? Give the move in {notation}.",
    ]
    return rng.choice(q_variants), answer


def _task_give_check(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """Name a move that delivers check to the opponent.

    Filters legal moves by board.gives_check(). Prefer non-capture checks
    (more instructive). Skip 80% of "none" positions — checks are ~25%.
    """
    if board.is_checkmate() or board.is_stalemate():
        return None

    all_checks = [m for m in board.legal_moves if board.gives_check(m)]
    quiet_checks = [m for m in all_checks if not board.is_capture(m)]

    if not all_checks:
        if rng.random() < 0.80:
            return None
        answer = "none"
        notation = rng.choice(["SAN (e.g. Bxe5)", "UCI (e.g. f1e5)"])
    else:
        pool = quiet_checks if quiet_checks and rng.random() < 0.6 else all_checks
        move = rng.choice(pool)
        notation, answer = _move_notation(board, move, rng)

    color_str = "white" if board.turn == chess.WHITE else "black"
    q_variants = [
        f"Name a move that puts the opponent in check. Use {notation}.",
        f"Is there a checking move for {color_str}? If so, name one in {notation}.",
        f"Which {color_str} move delivers check? Answer in {notation}, or 'none'.",
        f"Give a checking move in {notation}, or 'none' if no check is possible.",
    ]
    return rng.choice(q_variants), answer


def _task_threaten_piece_with(
    board: chess.Board, rng: random.Random, **_
) -> tuple[str, str] | None:
    """How can piece X on S1 directly threaten enemy piece Y on S2?

    Searches legal moves for the side to move: find a move from S1 whose
    destination attacks S2. Tests up to 4 attackers × 4 targets.
    """
    color = board.turn
    opp_color = not color

    own_pieces = [
        (sq, p)
        for sq in chess.SQUARES
        if (p := board.piece_at(sq)) is not None and p.color == color and p.piece_type != chess.KING
    ]
    target_pieces = [
        (sq, p)
        for sq in chess.SQUARES
        if (p := board.piece_at(sq)) is not None
        and p.color == opp_color
        and p.piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    ]

    if not own_pieces or not target_pieces:
        return None

    rng.shuffle(own_pieces)
    rng.shuffle(target_pieces)

    for att_sq, attacker in own_pieces[:4]:
        # Legal moves by this attacker
        att_moves = [m for m in board.legal_moves if m.from_square == att_sq]
        if not att_moves:
            continue
        for tgt_sq, target in target_pieces[:4]:
            threat_moves = []
            for move in att_moves:
                board.push(move)
                attacks_target = bool(board.attacks(move.to_square) & chess.BB_SQUARES[tgt_sq])
                board.pop()
                if attacks_target:
                    threat_moves.append(move)
            if threat_moves:
                move = rng.choice(threat_moves)
                notation, answer = _move_notation(board, move, rng)

                color_str = "white" if color == chess.WHITE else "black"
                opp_str = "black" if color == chess.WHITE else "white"
                att_name = _PIECE_NAMES[attacker.piece_type]
                tgt_name = _PIECE_NAMES[target.piece_type]
                q_variants = [
                    f"How can {color_str}'s {att_name} on {_sq(att_sq)} directly threaten {opp_str}'s {tgt_name} on {_sq(tgt_sq)}? Use {notation}.",
                    f"Name a move in {notation} where the {color_str} {att_name} on {_sq(att_sq)} attacks the {opp_str} {tgt_name} on {_sq(tgt_sq)}.",
                    f"How can the {color_str} {att_name} on {_sq(att_sq)} create a direct threat against the {opp_str} {tgt_name} on {_sq(tgt_sq)}? Answer in {notation}.",
                ]
                return rng.choice(q_variants), answer

    return None


def _task_fork_move(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """Find a move that forks (simultaneously attacks) two or more opponent pieces.

    Iterates legal moves; after each push, counts opponent pieces attacked by
    the moved piece's new square. Skip 85% of "none" positions.
    """
    color = board.turn
    opp_color = not color

    fork_moves: list[chess.Move] = []
    for move in board.legal_moves:
        board.push(move)
        # Opponent pieces still on board after the move
        opp_squares = chess.SquareSet(
            sq
            for sq in chess.SQUARES
            if (p := board.piece_at(sq)) is not None
            and p.color == opp_color
            and p.piece_type != chess.PAWN
        )
        attacked = board.attacks(move.to_square) & opp_squares
        board.pop()
        if len(attacked) >= 2:
            fork_moves.append(move)

    notation = rng.choice(["SAN (e.g. Bxe5)", "UCI (e.g. f1e5)"])
    if not fork_moves:
        if rng.random() < 0.85:
            return None
        answer = "none"
    else:
        move = rng.choice(fork_moves)
        notation, answer = _move_notation(board, move, rng)
    color_str = "white" if color == chess.WHITE else "black"
    opp_str = "black" if color == chess.WHITE else "white"
    q_variants = [
        f"Can {color_str} fork two {opp_str} pieces in one move? If so, name the move in {notation}, or 'none'.",
        f"Name a {color_str} move in {notation} that simultaneously attacks two {opp_str} pieces, or 'none'.",
        f"Is there a fork available for {color_str}? Answer in {notation} or 'none'.",
    ]
    return rng.choice(q_variants), answer


def _task_doubled_pawns(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """Does a side have doubled pawns (two pawns on the same file)?

    Skip 80% of "no" positions to approach 50/50 label balance.
    """
    color = rng.choice([chess.WHITE, chess.BLACK])
    has_doubled = any(
        sum(
            1
            for rank in range(8)
            if board.piece_at(chess.square(file, rank)) == chess.Piece(chess.PAWN, color)
        )
        >= 2
        for file in range(8)
    )
    if not has_doubled and rng.random() < 0.80:
        return None

    answer = "yes" if has_doubled else "no"
    color_str = "white" if color == chess.WHITE else "black"
    q_variants = [
        f"Does {color_str} have any doubled pawns? (yes/no)",
        f"Are there doubled pawns in {color_str}'s pawn structure? Answer yes or no.",
        f"Does {color_str} have two pawns on the same file?",
        f"Answer 'yes' or 'no': does {color_str} have doubled pawns?",
    ]
    return rng.choice(q_variants), answer


def _task_isolated_pawn_at(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """Is a specific pawn isolated (no friendly pawns on neighboring files)?

    Skip 85% of non-isolated cases to approach 50/50 label balance.
    """
    pawns = [
        sq
        for sq in chess.SQUARES
        if (p := board.piece_at(sq)) is not None and p.piece_type == chess.PAWN
    ]
    if not pawns:
        return None

    sq = rng.choice(pawns)
    piece = board.piece_at(sq)
    assert piece is not None
    color = piece.color
    file = chess.square_file(sq)

    adj_files = [f for f in (file - 1, file + 1) if 0 <= f <= 7]
    is_isolated = not any(
        board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color)
        for f in adj_files
        for r in range(8)
    )
    if not is_isolated and rng.random() < 0.85:
        return None

    answer = "yes" if is_isolated else "no"
    color_str = "white" if color == chess.WHITE else "black"
    q_variants = [
        f"Is the {color_str} pawn on {_sq(sq)} isolated? (yes/no)",
        f"Does the pawn on {_sq(sq)} have any friendly pawns on adjacent files? Answer yes or no.",
        f"Is the pawn on {_sq(sq)} isolated (no friendly pawns on neighboring files)?",
        f"Answer 'yes' or 'no': is the {color_str} pawn on {_sq(sq)} isolated?",
    ]
    return rng.choice(q_variants), answer


def _task_passed_pawn(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """Does a side have a passed pawn? If so, return its square.

    A pawn is passed if no opposing pawn is on its file or adjacent files
    ahead of it. Returns the most advanced passed pawn. Skip 60% of "none".
    """
    color = rng.choice([chess.WHITE, chess.BLACK])
    opp_color = not color

    passed: list[int] = []
    for sq in board.pieces(chess.PAWN, color):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        ranks_ahead = range(rank + 1, 8) if color == chess.WHITE else range(rank - 1, -1, -1)
        adj_files = [f for f in (file - 1, file, file + 1) if 0 <= f <= 7]
        if not any(
            board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, opp_color)
            for f in adj_files
            for r in ranks_ahead
        ):
            passed.append(sq)

    if not passed and rng.random() < 0.85:
        return None

    color_str = "white" if color == chess.WHITE else "black"
    if passed:
        best = (max if color == chess.WHITE else min)(passed, key=chess.square_rank)
        answer = _sq(best)
        q_variants = [
            f"Does {color_str} have a passed pawn? If so, name the square.",
            f"Name the square of {color_str}'s most advanced passed pawn, or 'none'.",
            f"Which {color_str} pawn has no opposing pawns blocking its path? Give the square.",
        ]
    else:
        answer = "none"
        q_variants = [
            f"Does {color_str} have a passed pawn? If so, name the square.",
            f"Name the square of {color_str}'s most advanced passed pawn, or 'none'.",
        ]
    return rng.choice(q_variants), answer


def _task_board_inventory(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """List all pieces on the board grouped by type and color.

    Answer format varies by question prompt — the model must follow the instruction.
    This is a hard holistic task: the model must read all 64 squares from encoder tokens.
    """
    # Build per-piece-type square lists
    piece_types = [
        (chess.WHITE, chess.KING, "white king"),
        (chess.WHITE, chess.QUEEN, "white queen"),
        (chess.WHITE, chess.ROOK, "white rook"),
        (chess.WHITE, chess.BISHOP, "white bishop"),
        (chess.WHITE, chess.KNIGHT, "white knight"),
        (chess.WHITE, chess.PAWN, "white pawn"),
        (chess.BLACK, chess.KING, "black king"),
        (chess.BLACK, chess.QUEEN, "black queen"),
        (chess.BLACK, chess.ROOK, "black rook"),
        (chess.BLACK, chess.BISHOP, "black bishop"),
        (chess.BLACK, chess.KNIGHT, "black knight"),
        (chess.BLACK, chess.PAWN, "black pawn"),
    ]

    fmt = rng.randint(0, 2)

    if fmt == 0:
        # "white pawn: [a2, b2], white rook: [a1, h1], ..." — skip empty groups
        parts = []
        for color, pt, label in piece_types:
            squares = sorted([_sq(sq) for sq in board.pieces(pt, color)])
            if squares:
                parts.append(f"{label}: [{', '.join(squares)}]")
        if not parts:
            return None
        answer = ", ".join(parts)
        q_variants = [
            "List all pieces on the board grouped by type. Format: 'white pawn: [a2, b2], white rook: [a1, h1], ...'",
            "Enumerate all pieces by type and color. Use format: 'piece type: [square, ...]' for each group present.",
        ]
    elif fmt == 1:
        # Compact: "K:e1 Q:d1 R:a1 R:h1 | k:e8 q:d8 ..." (letter:square style, avoids SAN ambiguity)
        white_parts = []
        black_parts = []
        abbr_map = {
            chess.KING: "K",
            chess.QUEEN: "Q",
            chess.ROOK: "R",
            chess.BISHOP: "B",
            chess.KNIGHT: "N",
            chess.PAWN: "P",
        }
        for pt in [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
            for sq in sorted(board.pieces(pt, chess.WHITE)):
                white_parts.append(f"{abbr_map[pt]}:{_sq(sq)}")
            for sq in sorted(board.pieces(pt, chess.BLACK)):
                black_parts.append(f"{abbr_map[pt].lower()}:{_sq(sq)}")
        answer = " ".join(white_parts) + " | " + " ".join(black_parts)
        q_variants = [
            "List all white pieces then all black pieces. Format: 'K:e1 Q:d1 ... | k:e8 q:d8 ...' (uppercase=white, lowercase=black, Letter:square).",
            "Give a compact piece list using 'Letter:square' format (uppercase=white, lowercase=black), separated by '|'. E.g. 'K:e1 Q:d1 R:a1 | k:e8 q:d8'.",
        ]
    else:
        # Count-only: "white: K=1 Q=1 R=2 B=2 N=2 P=8, black: K=1 Q=1 R=2 B=2 N=2 P=8"
        def _counts(color: chess.Color) -> str:
            return " ".join(
                f"{abbr}={len(board.pieces(pt, color))}"
                for pt, abbr in [
                    (chess.KING, "K"),
                    (chess.QUEEN, "Q"),
                    (chess.ROOK, "R"),
                    (chess.BISHOP, "B"),
                    (chess.KNIGHT, "N"),
                    (chess.PAWN, "P"),
                ]
                if board.pieces(pt, color)
            )

        answer = f"white: {_counts(chess.WHITE)}, black: {_counts(chess.BLACK)}"
        q_variants = [
            "Count all pieces by type for each side. Format: 'white: K=1 Q=1 R=2 B=2 N=2 P=8, black: ...'",
            "How many of each piece type does each side have? Format: 'white: K=1 Q=0 R=1 ..., black: ...'",
        ]

    return rng.choice(q_variants), answer


def _task_checkmate_in_one(board: chess.Board, rng: random.Random, **_) -> tuple[str, str] | None:
    """What move delivers immediate checkmate?

    Iterates legal moves and returns any that result in checkmate. Answer is
    the SAN of the mating move (always ends with '#'). When sourced from the
    Lichess mateIn1 puzzle pool every position has exactly one such move.

    For random positions checkmate-in-one is extremely rare (<0.1%) — skip
    99% of non-checkmate positions to avoid flooding the dataset with 'none'.
    """
    checkmate_moves = []
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            checkmate_moves.append(m)
        board.pop()

    notation = rng.choice(["SAN (e.g. Bxe5)", "UCI (e.g. f1e5)"])
    if not checkmate_moves:
        if rng.random() < 0.99:
            return None
        answer = "none"
    else:
        move = rng.choice(checkmate_moves)
        notation, answer = _move_notation(board, move, rng)

    color_str = "white" if board.turn == chess.WHITE else "black"
    q_variants = [
        f"Is there a checkmate in one move? If so, give the move in {notation}.",
        f"Can {color_str} deliver checkmate in one move? Answer in {notation} or 'none'.",
        f"What move gives immediate checkmate? Use {notation}, or 'none' if none exists.",
        f"Name the mating move in {notation}, or 'none'.",
    ]
    return rng.choice(q_variants), answer


# Each entry: (name, fn, difficulty)  — "easy" or "medium"
# "medium" tasks require multi-step reasoning and output SAN moves or derived pawn structure.
# generate_for_fen uses difficulty to implement the 80/20 split (--difficulty medium).
_TASKS: list[tuple[str, Callable[..., Any], str]] = [
    # Easy tasks — direct board-reading questions
    ("piece_abbr_at", _task_piece_abbr_at, "easy"),
    ("piece_name_at", _task_piece_name_at, "easy"),
    ("side_to_move", _task_side_to_move, "easy"),
    ("rank_contents", _task_rank_contents, "easy"),
    ("file_contents", _task_file_contents, "easy"),
    ("castling_rights", _task_castling_rights, "easy"),
    ("en_passant", _task_en_passant, "easy"),
    ("move_number", _task_move_number, "easy"),
    ("count_piece_type", _task_count_piece_type, "easy"),
    ("count_total_material", _task_count_total_material, "easy"),
    ("find_piece_type", _task_find_piece_type, "easy"),
    ("is_square_occupied", _task_is_square_occupied, "easy"),
    # Per-square tasks — probe structural info encoded in CLIP per-square anchors
    ("attackers_at", _task_attackers_at, "easy"),
    ("is_pinned", _task_is_pinned, "easy"),
    ("mobility_at", _task_mobility_at, "easy"),
    # Global token tasks (FEN-only)
    ("is_check", _task_is_check, "easy"),
    ("material_balance", _task_material_balance, "easy"),
    ("who_is_better", _task_who_is_better, "easy"),
    # Global token tasks (require SF15 data — skipped when not provided)
    ("eval_tier", _task_eval_tier, "easy"),
    ("sf15_dominant", _task_sf15_dominant, "easy"),
    # Medium tasks — multi-step reasoning; answers in SAN notation or derived pawn structure
    ("hanging_pieces", _task_hanging_pieces, "medium"),
    ("capture_on_square", _task_capture_on_square, "medium"),
    ("give_check", _task_give_check, "medium"),
    ("threaten_piece_with", _task_threaten_piece_with, "medium"),
    ("fork_move", _task_fork_move, "medium"),
    ("doubled_pawns", _task_doubled_pawns, "medium"),
    ("isolated_pawn_at", _task_isolated_pawn_at, "medium"),
    ("passed_pawn", _task_passed_pawn, "medium"),
    ("checkmate_in_one", _task_checkmate_in_one, "medium"),
    ("board_inventory", _task_board_inventory, "medium"),
]

# Derived sets — do not edit manually, edit the difficulty field in _TASKS above.
_MEDIUM_TASK_NAMES = {name for name, _, diff in _TASKS if diff == "medium"}

_RARE_TASKS = {
    "castling_rights",
    "en_passant",
    "is_check",
    "is_pinned",
    "eval_tier",
    "sf15_dominant",
    "checkmate_in_one",  # sourced from Lichess mateIn1 puzzle pool via --puzzle-csv
}


def generate_for_fen(
    fen: str,
    rng: random.Random,
    tasks_per_fen: int = 1,
    sf15_terms: list[float] | None = None,
    eval_score: float | None = None,
    medium_ratio: float = 0.0,
) -> list[dict]:
    """Generate task records for one FEN.

    Args:
        medium_ratio: Fraction of tasks drawn from the medium pool (0.0 = easy only).
                      At 0.8, each FEN has an 80% chance of getting a medium task and
                      a 20% chance of getting an easy task (per tasks_per_fen slot).
    """
    try:
        board = chess.Board(fen)
    except Exception:
        return []

    records = []
    if medium_ratio > 0.0:
        easy_pool = [(n, fn) for n, fn, _ in _TASKS if n not in _MEDIUM_TASK_NAMES]
        medium_pool = [(n, fn) for n, fn, _ in _TASKS if n in _MEDIUM_TASK_NAMES]
        task_pool = medium_pool if rng.random() < medium_ratio else easy_pool
    else:
        task_pool = [(n, fn) for n, fn, _ in _TASKS]
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


def _collect_puzzle_fens(max_fens: int) -> list[str]:
    """Stream mateIn1 puzzle positions from Lichess/chess-puzzles on HuggingFace.

    Dataset: https://huggingface.co/datasets/Lichess/chess-puzzles
    Columns used:
      FEN   — position before the opponent's setup move (opponent to move)
      Moves — space-separated UCI moves; Moves[0] is the opponent setup move,
              Moves[1] is the mating move
      Themes — space-separated tags; we filter for "mateIn1"

    We apply Moves[0] to get the puzzle position (solver to move), then verify
    Moves[1] actually delivers checkmate before including the FEN.
    """
    from datasets import load_dataset

    fens: list[str] = []
    seen: set[str] = set()
    skipped = 0

    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    for row in ds:
        rec = dict(row)
        themes = rec.get("Themes") or []
        if "mateIn1" not in themes:
            continue
        fen_raw: str = (rec.get("FEN") or "").strip()
        moves_str: str = (rec.get("Moves") or "").strip()
        moves = moves_str.split()
        if len(moves) < 2 or not fen_raw:
            continue
        try:
            board = chess.Board(fen_raw)
            board.push(chess.Move.from_uci(moves[0]))  # opponent setup move
            puzzle_fen = board.fen()
            if puzzle_fen in seen:
                continue
            # Verify the solution is actually checkmate
            mate_move = chess.Move.from_uci(moves[1])
            if mate_move not in board.legal_moves:
                skipped += 1
                continue
            board.push(mate_move)
            if not board.is_checkmate():
                skipped += 1
                continue
            seen.add(puzzle_fen)
            fens.append(puzzle_fen)
        except Exception:
            skipped += 1
            continue
        if max_fens and len(fens) >= max_fens:
            break

    _logger.info("Puzzle FENs collected: %d (skipped=%d, target=%d)", len(fens), skipped, max_fens)
    return fens


def _worker(args: tuple) -> list[dict]:
    """Top-level function for multiprocessing (must be picklable)."""
    fen, seed, tasks_per_fen, sf15_terms, eval_score, medium_ratio = args
    return generate_for_fen(
        fen,
        random.Random(seed),
        tasks_per_fen,
        sf15_terms=sf15_terms,
        eval_score=eval_score,
        medium_ratio=medium_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/processed/encoder_pretrain_1b.jsonl",
        help="Source JSONL with FENs (encoder pretrain or grpo format)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium"],
        default="easy",
        help="easy = original 20 tasks only (default). "
        "medium = 80%% medium tasks + 20%% easy tasks; output named *_medium.jsonl.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Train output path (default: auto-set from --difficulty).",
    )
    parser.add_argument(
        "--eval-output",
        default=None,
        help="Eval output path (default: auto-set from --difficulty).",
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
    parser.add_argument(
        "--total-records",
        type=int,
        default=None,
        help="Total records to generate — sets --max-fens when tasks-per-fen=1. "
        "Convenience alias; overrides --max-fens when provided.",
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
    parser.add_argument(
        "--puzzle-fens",
        type=int,
        default=5000,
        help="Max mateIn1 puzzle FENs to stream from Lichess/chess-puzzles on HuggingFace "
        "(default 5000). Set to 0 to skip.",
    )
    args = parser.parse_args()

    # --total-records overrides --max-fens when tasks-per-fen=1
    if args.total_records is not None:
        args.max_fens = args.total_records // args.tasks_per_fen

    # Auto-set output paths based on difficulty if not explicitly provided
    _suffix = "_medium" if args.difficulty == "medium" else ""
    if args.output is None:
        args.output = f"data/processed/alignment_board_description{_suffix}.jsonl"
    if args.eval_output is None:
        args.eval_output = f"data/processed/alignment_board_description{_suffix}_eval.jsonl"

    # 0.85 selection probability compensates for medium tasks' lower yield (~53%) vs easy (~74%),
    # resulting in ~80% medium records in the output.
    medium_ratio = 0.85 if args.difficulty == "medium" else 0.0
    _logger.info(
        "Difficulty: %s  medium_ratio=%.1f  output=%s",
        args.difficulty,
        medium_ratio,
        args.output,
    )

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

    # Stream mateIn1 puzzle FENs from Lichess/chess-puzzles on HuggingFace
    if args.puzzle_fens > 0:
        _logger.info(
            "Streaming mateIn1 puzzles from Lichess/chess-puzzles (max=%d)...", args.puzzle_fens
        )
        mate1_fens = _collect_puzzle_fens(max_fens=args.puzzle_fens)
    else:
        mate1_fens = []
        _logger.info("--puzzle-fens=0; skipping checkmate_in_one puzzle pool.")

    # Rare FENs are kept separate — only used for their respective tasks
    rare_fens_by_task = {
        "en_passant": ep_fens,
        "castling_rights": castling_fens,
        "is_check": check_fens,
        "is_pinned": pinned_fens,
        # SF15 tasks: use only SF15-source FENs (have eval_score + sf15_terms)
        "eval_tier": [f for f in sf15_fens if f in sf15_extra],
        "sf15_dominant": [f for f in sf15_fens if f in sf15_extra],
        # Checkmate-in-one: sourced from Lichess mateIn1 puzzle pool
        "checkmate_in_one": mate1_fens,
    }

    rng.shuffle(fens)

    def _write_split(
        train_path: str,
        eval_path: str,
        fen_list: list[str],
        rare_pool: dict[str, list[str]],
        target_eval_records: int,
    ) -> tuple[int, int]:
        """Generate data and split into train/eval to match target eval size using RNG."""
        # Estimate eval_ratio needed to get target eval records
        # With N FENs * 1 task/fen ≈ N records, we want eval_ratio = target_eval_records / N
        estimated_records = len(fen_list)
        eval_ratio = min(
            0.5, max(0.001, target_eval_records / (estimated_records + target_eval_records))
        )

        task_counts_train: dict[str, int] = {name: 0 for name, _, _ in _TASKS}
        task_counts_eval: dict[str, int] = {name: 0 for name, _, _ in _TASKS}
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
                medium_ratio,
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

        # Rare task pass: generate from dedicated rare FEN pools (positives guaranteed).
        # All rare tasks run regardless of difficulty — easy rare tasks (is_check, is_pinned,
        # en_passant, castling_rights) need their pools to achieve near-50/50 label balance.
        MAX_NONE_RATIO = 0.30
        topup_rng = random.Random(rng.randint(0, 2**32))
        rare_fns = {name: fn for name, fn, _ in _TASKS if name in _RARE_TASKS}

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
            train_positives = [
                r
                for r in records_train
                if r["metadata"]["task"] == name and r["messages"][1]["content"] != "none"
            ]
            train_nones = [
                r
                for r in records_train
                if r["metadata"]["task"] == name and r["messages"][1]["content"] == "none"
            ]
            eval_positives = [
                r
                for r in records_eval
                if r["metadata"]["task"] == name and r["messages"][1]["content"] != "none"
            ]
            eval_nones = [
                r
                for r in records_eval
                if r["metadata"]["task"] == name and r["messages"][1]["content"] == "none"
            ]

            max_train_none = int(len(train_positives) * MAX_NONE_RATIO / (1.0 - MAX_NONE_RATIO))
            max_eval_none = int(len(eval_positives) * MAX_NONE_RATIO / (1.0 - MAX_NONE_RATIO))
            topup_rng.shuffle(train_nones)
            topup_rng.shuffle(eval_nones)

            # Rebuild records lists without this task, then add trimmed version
            records_train = [r for r in records_train if r["metadata"]["task"] != name]
            records_eval = [r for r in records_eval if r["metadata"]["task"] != name]

            records_train.extend(train_positives + train_nones[:max_train_none])
            records_eval.extend(eval_positives + eval_nones[:max_eval_none])

            task_counts_train[name] = len(
                [r for r in records_train if r["metadata"]["task"] == name]
            )
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

    n_train, n_eval = _write_split(
        args.output,
        args.eval_output,
        fens,
        rare_fens_by_task,
        target_eval_records=args.eval_fens * args.tasks_per_fen,
    )
    _logger.info("Train: %d records → %s", n_train, args.output)
    _logger.info("Eval:  %d records → %s", n_eval, args.eval_output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
