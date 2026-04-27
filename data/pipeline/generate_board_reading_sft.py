"""Generate SFT data for board reading only.

Task: Given <board> vision tokens, list all pieces with their squares.
No puzzle solving, no move prediction — pure board perception.

Assistant turn (no thinking block):
  White: king on e1, rook on h1, pawn on a2...
  Black: king on e8, rook on a8, pawn on a7...

This trains CNN embeddings → piece identity/square mapping without
leaking any move answer that the model can shortcut through.

Usage:
    python data/pipeline/generate_board_reading_sft.py \\
        --input data/processed/puzzle_grpo/train.jsonl \\
        --output data/processed/board_reading_sft.jsonl \\
        --max-records 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from encoder import BOARD_TOKEN

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

SYSTEM = (
    "You are a chess assistant. The board position is encoded as a sequence of vision tokens "
    "wrapped in <board> </board> tags. Use them to identify pieces and answer questions about the position."
)


def describe_pieces(board: chess.Board) -> str:
    """List all pieces by color and square, sorted a1→h8."""
    white_pieces = []
    black_pieces = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        name = PIECE_NAMES[piece.piece_type]
        sq_name = chess.SQUARE_NAMES[sq]
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


def build_messages(fen: str) -> list[dict]:
    board = chess.Board(fen)
    color = "White" if board.turn == chess.WHITE else "Black"

    user = (
        f"<board>{BOARD_TOKEN}</board>\n\n"
        f"It's {color}'s turn. List all pieces on the board with their squares.\n"
        f"Use this format:\n"
        f"White: <piece> on <square>, <piece> on <square>, ...\n"
        f"Black: <piece> on <square>, <piece> on <square>, ..."
    )
    assistant = describe_pieces(board)

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/puzzle_grpo/train.jsonl")
    parser.add_argument("--output", default="data/processed/board_reading_sft.jsonl")
    parser.add_argument("--max-records", type=int, default=100000)
    args = parser.parse_args()

    print(f"Reading from {args.input}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if n_ok >= args.max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fen = rec["fen"]
                messages = build_messages(fen)
                fout.write(json.dumps({
                    "messages": messages,
                    "metadata": {"fen": fen},
                }) + "\n")
                n_ok += 1
                if n_ok % 10000 == 0:
                    print(f"  {n_ok}...")
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"  Failed: {e}")

    print(f"Done: {n_ok} ok, {n_fail} failed → {args.output}")

    # Print sample
    with open(args.output) as f:
        sample = json.loads(f.readline())
    print("\n--- Sample ---")
    print("USER:", sample["messages"][1]["content"][:100])
    print("ASSISTANT:", sample["messages"][2]["content"])


if __name__ == "__main__":
    main()
