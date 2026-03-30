"""Convert annotated Lichess study PGN into clean text with inline FEN tokens.

Each chapter becomes a section of prose. Move annotations ({ text }) are
extracted as natural text. The FEN at each annotated position is inlined as
[Position: FEN] so the encoder can later inject board tokens there.

Output: plain .txt file — one section per chapter, positions as FEN tokens.

Usage:
    uv run python data/pipeline/parse_pgn_study.py \\
        --input data/raw/textbooks/capablanca_fundamentals_part1.pgn \\
        --output data/processed/textbook_capablanca_part1.txt
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import chess
import chess.pgn


def _clean(text: str) -> str:
    """Strip PGN annotation markers and normalize whitespace."""
    # Remove clock annotations [%clk ...], eval annotations [%eval ...]
    text = re.sub(r"\[%\w+[^\]]*\]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pgn_to_text(pgn_text: str) -> str:
    """Convert a full PGN file (possibly multi-game) to clean annotated text."""
    out_sections: list[str] = []
    pgn_io = io.StringIO(pgn_text)

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break

        chapter = game.headers.get("ChapterName", game.headers.get("Event", ""))
        study = game.headers.get("StudyName", "")

        lines: list[str] = []
        if study:
            lines.append(f"## {study}")
        if chapter:
            lines.append(f"### {chapter}")
        lines.append("")

        # Walk all nodes collecting annotations and positions
        node = game
        board = game.board()

        def _walk(node: chess.pgn.GameNode, board: chess.Board, depth: int = 0) -> None:
            # Emit comment before move
            if node.comment:
                comment = _clean(node.comment)
                if comment:
                    if depth == 0:
                        lines.append(comment)
                    else:
                        lines.append(f"({comment})")

            for child in node.variations:
                child_board = board.copy()
                san = child_board.san(child.move)
                child_board.push(child.move)

                is_main = child == node.variations[0]

                if is_main:
                    # Inline FEN only when this node has a substantive comment
                    if child.comment and len(_clean(child.comment)) > 20:
                        fen = child_board.fen()
                        lines.append(f"[Position: {fen}]")

                    _walk(child, child_board, depth)
                else:
                    # Variation — wrap in parens
                    lines.append(f"(Variation: {san}")
                    _walk(child, child_board, depth + 1)
                    lines.append(")")

        _walk(node, board)

        section = "\n".join(lines).strip()
        if section:
            out_sections.append(section)

    return "\n\n---\n\n".join(out_sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pgn_text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    result = pgn_to_text(pgn_text)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(result, encoding="utf-8")

    n_positions = result.count("[Position:")
    n_sections = result.count("###")
    print(f"Written {n_sections} chapters, {n_positions} positions → {args.output}")


if __name__ == "__main__":
    main()
