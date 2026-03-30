"""Parse chess textbook plain text (Gutenberg format) into SFT training records.

Extracts (FEN, surrounding_text) pairs from books with ASCII board diagrams.

Supported formats:
  - Lasker format (Chess Strategy #5614, Chess and Checkers #4913):
      Piece notation: ^P ^Kt ^B ^R ^Q ^K (White), #P #Kt #B #R #Q #K (Black)
      Board border:   -------...------- (top/bottom), |----...-| (row dividers)
      Rank labels:    left of each piece row (1-8 or 8-1)
      File labels:    A-H or a-h below the board
      Caption:        "Diag. N." or "DIAGRAM N." on the line after the board

  - Staunton format (Blue Book of Chess #16377):
      Piece notation: R, N, B, Q, K, P (White), R*, N*, B*, Q*, K*, P* (Black)
      Board border:   +---+---+---+---+---+---+---+---+
      No rank labels, BLACK./WHITE. header/footer outside the grid
      Caption:        "No. N." or "Diagram N." on the line before the BLACK. header

Output JSONL:
  {"fen": "...", "diagram_id": "Diag. 4", "text_before": "...", "text_after": "..."}

Usage:
    uv run python data/pipeline/parse_textbook_pgn.py \\
        --input data/raw/textbooks/lasker_chess_strategy.txt \\
        --output data/processed/textbook_lasker.jsonl

    uv run python data/pipeline/parse_textbook_pgn.py \\
        --input data/raw/textbooks/staunton_blue_book.txt \\
        --output data/processed/textbooks/textbook_staunton.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Piece mapping: Lasker format → FEN character
# ---------------------------------------------------------------------------

_WHITE = {
    "^P": "P",
    "^Kt": "N",
    "^B": "B",
    "^R": "R",
    "^Q": "Q",
    "^K": "K",
    "^N": "N",  # alternate spelling
}
_BLACK = {
    "#P": "p",
    "#Kt": "n",
    "#B": "b",
    "#R": "r",
    "#Q": "q",
    "#K": "k",
    "#N": "n",
    # Lasker sometimes omits the space before the piece token inside a cell:
    # |#Kt | or |^Kt | — handled by stripping
}

# ---------------------------------------------------------------------------
# Board detection
# ---------------------------------------------------------------------------

# A board top/bottom border line (39+ dashes, optionally leading spaces and +)
_BORDER_RE = re.compile(r"^\s*\+?-{30,}\+?\s*$")

# A rank row: optional spaces, rank digit, spaces, | cells |
_RANK_ROW_RE = re.compile(r"^\s*\d\s*\|")

# A separator row between ranks
_SEP_ROW_RE = re.compile(r"^\s*\|[-]+\|")

# Caption line after the board
_CAPTION_RE = re.compile(r"(?i)(diag(?:ram)?\.?\s*\d+\.?)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Staunton format (+---+---+ grid, * = black pieces)
# ---------------------------------------------------------------------------

# Staunton horizontal border: +---+---+---+---+---+---+---+---+
_STAUNTON_BORDER_RE = re.compile(r"^\s*\+(?:---\+){8}\s*$")

# Staunton rank row: | R*| N |  | ... | (8 cells)
_STAUNTON_RANK_ROW_RE = re.compile(r"^\s*\|(?:[^|]+\|){8}\s*$")

# Staunton caption line: "No. 21." or "Diagram 5." (appears before BLACK. header)
_STAUNTON_CAPTION_RE = re.compile(r"(?i)(?:No\.\s*(\d+)\.?|Diagram\s*(\d+)\.?)")


def _parse_staunton_cell(cell: str) -> str | None:
    """Parse a Staunton cell like ' R*' or ' P ' or '   ' into FEN char or None."""
    stripped = cell.strip()
    if not stripped:
        return None
    # Black piece ends with *
    if stripped.endswith("*"):
        piece = stripped[:-1].strip().upper()
        mapping = {"R": "r", "N": "n", "B": "b", "Q": "q", "K": "k", "P": "p"}
        return mapping.get(piece)
    # White piece
    mapping = {"R": "R", "N": "N", "B": "B", "Q": "Q", "K": "K", "P": "P"}
    return mapping.get(stripped.upper())


def _parse_staunton_rank_row(line: str) -> list[str | None] | None:
    """Parse a Staunton rank row into 8 cells."""
    # Split on |, skip first (empty before leading |) and last (empty/trailing)
    parts = line.strip().split("|")
    # parts[0] should be empty before first |
    cells = [p for p in parts if p != "" or True][1:-1]  # strip leading/trailing empty
    # Re-split properly
    inner = line.strip()
    if not inner.startswith("|") or not inner.endswith("|"):
        return None
    inner = inner[1:-1]  # strip outer |
    cells = inner.split("|")
    if len(cells) != 8:
        return None
    return [_parse_staunton_cell(c) for c in cells]


def _parse_staunton_board_block(lines: list[str], i: int) -> tuple[str | None, str, int]:
    """Try to parse a Staunton +---+ board block starting at line i.

    Line i should be the first +---+ border. Returns (fen, caption, next_i).
    """
    n = len(lines)
    rank_rows: list[list[str | None]] = []
    j = i + 1  # skip opening border

    while j < n and len(rank_rows) < 8:
        line = lines[j]
        if _STAUNTON_BORDER_RE.match(line):
            j += 1  # separator between ranks
        elif _STAUNTON_RANK_ROW_RE.match(line):
            parsed = _parse_staunton_rank_row(line)
            if parsed is not None:
                rank_rows.append(parsed)
            j += 1
        else:
            break

    # Skip trailing border if present
    if j < n and _STAUNTON_BORDER_RE.match(lines[j]):
        j += 1

    # Skip WHITE. footer line
    if j < n and re.match(r"^\s*WHITE\.\s*$", lines[j]):
        j += 1

    # Look for caption in the lines before i (scan backwards for "No. N." or "Diagram N.")
    caption = ""
    for k in range(i - 1, max(i - 5, -1), -1):
        m = _STAUNTON_CAPTION_RE.search(lines[k])
        if m:
            num = m.group(1) or m.group(2)
            caption = f"No. {num}" if m.group(1) else f"Diagram {num}"
            break
        if lines[k].strip() and not re.match(r"^\s*BLACK\.\s*$", lines[k]):
            break

    if len(rank_rows) != 8:
        return None, caption, j

    placement = _ranks_to_fen(rank_rows)
    if placement is None:
        return None, caption, j

    return _fen_placement_to_fen(placement), caption, j


def _parse_cell(cell: str) -> str | None:
    """Parse a 4-char board cell into a FEN piece char or None (empty)."""
    cell = cell.strip()
    if not cell:
        return None
    # Try white pieces (longest match first to handle ^Kt before ^K)
    for token, piece in sorted(_WHITE.items(), key=lambda x: -len(x[0])):
        if cell == token or cell.startswith(token):
            return piece
    for token, piece in sorted(_BLACK.items(), key=lambda x: -len(x[0])):
        if cell == token or cell.startswith(token):
            return piece
    return None  # empty or unrecognised (e.g. "*" for attack-square diagrams)


def _parse_rank_row(line: str) -> list[str | None] | None:
    """Parse a rank row into a list of 8 piece chars (None = empty square).

    Returns None if the row is malformed or not a valid rank row.
    """
    # Strip rank number prefix: "   8 | ..."  → "| ..."
    m = re.match(r"^\s*\d\s*(\|.*)", line)
    if not m:
        return None
    row_content = m.group(1)

    # Split on | — produces empty strings at the ends
    parts = row_content.split("|")
    # parts[0] is empty (before first |), parts[-1] may be empty or trailing spaces
    cells = parts[1:-1] if parts[-1].strip() == "" else parts[1:]

    if len(cells) != 8:
        return None

    return [_parse_cell(c) for c in cells]


def _ranks_to_fen(ranks: list[list[str | None]]) -> str | None:
    """Convert 8 rank rows (rank8 first) to a FEN piece placement string."""
    if len(ranks) != 8:
        return None

    fen_rows = []
    for rank in ranks:
        if len(rank) != 8:
            return None
        row_str = ""
        empty = 0
        for piece in rank:
            if piece is None:
                empty += 1
            else:
                if empty:
                    row_str += str(empty)
                    empty = 0
                row_str += piece
        if empty:
            row_str += str(empty)
        fen_rows.append(row_str)

    return "/".join(fen_rows)


def _fen_placement_to_fen(placement: str) -> str:
    """Add minimal FEN suffix (unknown side to move, no castling, etc.)."""
    # We don't know whose turn it is from a static diagram, use "w" as default
    return f"{placement} w - - 0 1"


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------


def _parse_board_block(lines: list[str], i: int) -> tuple[str | None, str, int]:
    """Try to parse an ASCII board block starting at line i.

    Returns (fen_or_none, caption, next_i).
    If the block is not a valid position, fen_or_none is None but next_i still
    advances past the block so caller can emit it as-is.
    """
    n = len(lines)
    rank_rows: list[list[str | None]] = []
    j = i + 1

    while j < n and len(rank_rows) < 8:
        line = lines[j]
        if _RANK_ROW_RE.match(line):
            parsed = _parse_rank_row(line)
            if parsed is not None:
                rank_rows.append(parsed)
        elif _SEP_ROW_RE.match(line) or _BORDER_RE.match(line):
            pass
        else:
            break
        j += 1

    # Skip bottom border
    if j < n and _BORDER_RE.match(lines[j]):
        j += 1
    # Skip file labels
    if j < n and re.match(r"^\s+[A-Ha-h](\s+[A-Ha-h]){7}\s*$", lines[j]):
        j += 1
    # Skip blank lines before caption
    while j < n and lines[j].strip() == "":
        j += 1

    caption = ""
    if j < n:
        m = _CAPTION_RE.search(lines[j])
        if m:
            caption = m.group(1).strip()
            j += 1

    raw = "\n".join(lines[i:j])
    if " *  " in raw or "| * " in raw or len(rank_rows) != 8:
        return None, caption, j

    placement = _ranks_to_fen(rank_rows)
    if placement is None:
        return None, caption, j

    return _fen_placement_to_fen(placement), caption, j


def replace_boards_with_fen(text: str) -> str:
    """Replace every ASCII board block in the text with '[Position (caption): FEN]'.

    Non-board lines pass through unchanged. This produces clean prose where
    diagrams are represented as inline FEN tokens.
    """
    lines = text.splitlines()
    n = len(lines)
    out: list[str] = []
    i = 0

    while i < n:
        # Staunton +---+ format (skip BLACK. header to get to the first border)
        if re.match(r"^\s*BLACK\.\s*$", lines[i]):
            # Peek ahead for the board border
            k = i + 1
            while k < n and lines[k].strip() == "":
                k += 1
            if k < n and _STAUNTON_BORDER_RE.match(lines[k]):
                fen, caption, next_i = _parse_staunton_board_block(lines, k)
                if fen is not None:
                    label = f" ({caption})" if caption else ""
                    out.append(f"[Position{label}: {fen}]")
                    i = next_i
                    continue
            out.append(lines[i])
            i += 1
            continue

        if not _BORDER_RE.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue

        fen, caption, next_i = _parse_board_block(lines, i)
        if fen is None:
            # Not a valid position — keep lines as-is
            out.append(lines[i])
            i += 1
        else:
            label = f" ({caption})" if caption else ""
            out.append(f"[Position{label}: {fen}]")
            i = next_i

    return "\n".join(out)


def extract_diagrams(text: str, context_chars: int = 1500) -> list[dict]:
    """Extract all diagrams with surrounding text.

    First replaces all ASCII boards with '[Position: FEN]' tokens so that
    text_before and text_after contain only clean prose (no board art).
    """
    clean_text = replace_boards_with_fen(text)
    clean_lines = clean_text.splitlines()

    records = []
    for idx, line in enumerate(clean_lines):
        m = re.match(r"^\[Position(?:\s*\(([^)]*)\))?: (.+)\]$", line)
        if not m:
            continue

        caption = m.group(1) or ""
        fen = m.group(2).strip()

        before_lines = []
        chars = 0
        k = idx - 1
        while k >= 0 and chars < context_chars:
            before_lines.append(clean_lines[k])
            chars += len(clean_lines[k]) + 1
            k -= 1
        text_before = "\n".join(reversed(before_lines)).strip()

        after_lines = []
        chars = 0
        k = idx + 1
        while k < len(clean_lines) and chars < context_chars:
            after_lines.append(clean_lines[k])
            chars += len(clean_lines[k]) + 1
            k += 1
        text_after = "\n".join(after_lines).strip()

        records.append(
            {
                "fen": fen,
                "diagram_id": caption,
                "text_before": text_before,
                "text_after": text_after,
            }
        )

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--context-chars",
        type=int,
        default=1500,
        help="Max chars of text to capture before/after each diagram",
    )
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="replace")

    # Output mode: if .txt, emit the full clean text with boards replaced by FEN tokens.
    # Otherwise emit JSONL with per-diagram records.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.output.endswith(".txt"):
        clean = replace_boards_with_fen(text)
        Path(args.output).write_text(clean, encoding="utf-8")
        n = clean.count("[Position")
        log.info("Written %d positions to %s", n, args.output)
    else:
        records = extract_diagrams(text, context_chars=args.context_chars)
        log.info("Extracted %d diagrams from %s", len(records), args.input)
        with open(args.output, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("Written to %s", args.output)


if __name__ == "__main__":
    main()
