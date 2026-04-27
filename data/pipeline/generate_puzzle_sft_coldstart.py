"""Generate SFT cold-start data for puzzle GRPO training.

Reads puzzle_grpo JSONL records (from generate_puzzle_grpo.py), runs Stockfish
analysis on each position, and produces SFT examples in the format:

  <think>
  Looking at the position, I can identify the pieces on the board.
  [piece list from python-chess]

  It's {color}'s turn. [tactical explanation from Stockfish + theme]
  The best move is {san} ({uci}).
  </think>
  FINAL ANSWER: {uci}

This cold-start teaches the model:
  1. Vision tokens → piece identification in thinking
  2. Tactical reasoning → UCI move after </think>

Usage:
    uv run python data/pipeline/generate_puzzle_sft_coldstart.py \\
        --input data/processed/puzzle_grpo/train.jsonl \\
        --output data/processed/puzzle_sft_coldstart.jsonl \\
        --max-records 5000 \\
        --stockfish-path ~/.local/bin/stockfish
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from encoder import BOARD_TOKEN

# ---------------------------------------------------------------------------
# Piece description helpers
# ---------------------------------------------------------------------------

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

SQUARE_NAMES = chess.SQUARE_NAMES  # a1..h8


def describe_pieces(board: chess.Board) -> str:
    """Describe all pieces on the board in natural language."""
    white_pieces = []
    black_pieces = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        name = PIECE_NAMES[piece.piece_type]
        sq_name = SQUARE_NAMES[sq]
        if piece.color == chess.WHITE:
            white_pieces.append(f"{name} on {sq_name}")
        else:
            black_pieces.append(f"{name} on {sq_name}")

    lines = []
    if white_pieces:
        lines.append("White: " + ", ".join(white_pieces) + ".")
    if black_pieces:
        lines.append("Black: " + ", ".join(black_pieces) + ".")
    return "\n".join(lines)


def describe_move_tactics(board: chess.Board, move: chess.Move, themes: list[str]) -> str:
    """Generate a short tactical explanation for a move."""
    san = board.san(move)
    uci = move.uci()
    color = "White" if board.turn == chess.WHITE else "Black"
    lines = []

    # Check if move is a capture
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        cap_name = PIECE_NAMES[captured.piece_type] if captured else "piece"
        lines.append(f"{san} captures the {cap_name} on {SQUARE_NAMES[move.to_square]}.")

    # Check if move gives check
    board.push(move)
    if board.is_check():
        lines.append("This puts the opponent's king in check.")
    if board.is_checkmate():
        lines.append("This is checkmate!")
    board.pop()

    # Promotion
    if move.promotion:
        promo_name = PIECE_NAMES[move.promotion]
        lines.append(f"The pawn promotes to a {promo_name}.")

    # Theme hints
    useful_themes = [t for t in themes if t not in ("oneMove", "master", "masterVsMaster", "middlegame", "endgame", "opening")]
    if useful_themes:
        lines.append(f"Key theme: {', '.join(useful_themes[:3])}.")

    if not lines:
        lines.append(f"{color} plays {san}, the best move in this position.")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Thinking trace builder
# ---------------------------------------------------------------------------

def build_thinking_trace(board: chess.Board, solution_uci: str, themes: list[str], color: str) -> str:
    """Build a thinking trace for the puzzle."""
    move = chess.Move.from_uci(solution_uci)
    san = board.san(move)

    piece_desc = describe_pieces(board)
    tactic_desc = describe_move_tactics(board, move, themes)

    # Check what the move does (capture, check, mate)
    board.push(move)
    is_check = board.is_check()
    is_mate = board.is_checkmate()
    board.pop()

    # Build reasoning
    parts = []
    parts.append(f"Looking at the board position, I can identify the pieces:\n{piece_desc}")
    parts.append(f"\nIt's {color}'s turn to move.")
    parts.append(f"\n{tactic_desc}")

    if is_mate:
        parts.append(f"\n{san} ({solution_uci}) delivers checkmate. That's the best move.")
    elif is_check:
        parts.append(f"\nPlaying {san} ({solution_uci}) gives check and maintains the advantage.")
    else:
        parts.append(f"\nThe best move is {san}, played as {solution_uci} in UCI notation.")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_messages(fen: str, themes: list[str], color: str, solution_uci: str) -> list[dict]:
    """Build the full messages list with thinking trace."""
    board = chess.Board(fen)
    thinking = build_thinking_trace(board, solution_uci, themes, color)

    is_mate = "mateIn1" in themes
    puzzle_type = "mate in 1 puzzle" if is_mate else "chess puzzle"

    system = (
        "You are a chess assistant. The board position is encoded as a sequence of vision tokens "
        "wrapped in <board> </board> tags. Use them to identify pieces and answer questions about the position."
    )
    user = (
        f"<board>{BOARD_TOKEN}</board>\n\n"
        f"This is a {puzzle_type}. Find the single best move for {color}. "
        f"Every puzzle has a forcing solution — check for checks, captures, and threats.\n\n"
        f"Think briefly, then immediately output your answer as a JSON object on the last line.\n"
        f'Use SAN (e.g. Nf3, Qxd5+) or UCI (e.g. g1f3, d1d5) notation. Example: {{"move": "Nf3"}}'
    )
    assistant = f'<think>\n{thinking}\n</think>\n{{"move": "{solution_uci}"}}'

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/puzzle_grpo/train.jsonl")
    parser.add_argument("--output", default="data/processed/puzzle_sft_coldstart.jsonl")
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Reading puzzles from {args.input}...")
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(records) >= args.max_records:
                break

    print(f"Building SFT traces for {len(records)} records...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    with open(args.output, "w") as out:
        for rec in records:
            try:
                fen = rec["fen"]
                solution_uci = rec["solution_uci"]
                themes = rec.get("themes", [])
                color = rec.get("color", "White")

                # Validate move is legal
                board = chess.Board(fen)
                move = chess.Move.from_uci(solution_uci)
                if move not in board.legal_moves:
                    n_fail += 1
                    continue

                messages = build_messages(fen, themes, color, solution_uci)
                out.write(json.dumps({
                    "messages": messages,
                    "metadata": {
                        "fen": fen,
                        "solution_uci": solution_uci,
                        "themes": themes,
                        "color": color,
                        "rating": rec.get("rating", 0),
                    },
                }) + "\n")
                n_ok += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"  Failed: {e}")

    print(f"Done: {n_ok} ok, {n_fail} failed → {args.output}")

    # Print a sample
    with open(args.output) as f:
        sample = json.loads(f.readline())
    print("\n--- Sample assistant turn ---")
    print(sample["messages"][2]["content"])


if __name__ == "__main__":
    main()
