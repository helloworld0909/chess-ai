"""Stockfish 15 classical eval wrapper.

Extracts named evaluation terms (Material, Mobility, King safety, etc.) that
were removed in Stockfish 16 when NNUE became the sole evaluator.

Uses a persistent subprocess per process (module-level singleton) to avoid the
overhead of spawning a new process for every FEN.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

# Configurable via env var; falls back to a compiled sf15 binary on PATH
_SF15_PATH = os.environ.get("SF15_PATH", "/usr/local/bin/stockfish-15")

# Module-level persistent process (one per worker process in a pool)
_proc: subprocess.Popen | None = None


def _get_proc() -> subprocess.Popen:
    global _proc
    if _proc is None or _proc.poll() is not None:
        _proc = subprocess.Popen(
            [_SF15_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Handshake
        assert _proc.stdin and _proc.stdout
        _proc.stdin.write("uci\n")
        _proc.stdin.flush()
        for line in _proc.stdout:
            if line.strip() == "uciok":
                break
        _proc.stdin.write("isready\n")
        _proc.stdin.flush()
        for line in _proc.stdout:
            if line.strip() == "readyok":
                break
    return _proc


def _send(proc: subprocess.Popen, cmd: str) -> None:
    assert proc.stdin
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def _read_eval_output(proc: subprocess.Popen) -> str:
    """Read until SF15 eval output sentinel line."""
    assert proc.stdout
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line)
        # SF15 ends with "Final evaluation" after the classical+NNUE breakdown
        if "Final evaluation" in line:
            break
    return "\n".join(lines)


# Term names as printed by SF15 trace (left-stripped)
_TERMS = [
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

_ROW_RE = re.compile(
    r"^\|\s*(?P<term>[A-Za-z ]+?)\s*\|"
    r"\s*(?P<wm>[-0-9.]+|-+)\s+(?P<we>[-0-9.]+|-+)\s*\|"
    r"\s*(?P<bm>[-0-9.]+|-+)\s+(?P<be>[-0-9.]+|-+)\s*\|",
)


def _parse_float(s: str) -> float:
    s = s.strip()
    if "---" in s or s == "":
        return 0.0
    return float(s)


def get_sf15_eval(fen: str) -> dict[str, dict[str, float]]:
    """Return per-term MG/EG averages for White and Black.

    Returns:
        {term_name: {"White": float, "Black": float}}
        Values are in pawn units (cp/100), averaged over MG and EG.
    """
    proc = _get_proc()
    _send(proc, f"position fen {fen}")
    _send(proc, "eval")
    raw = _read_eval_output(proc)

    terms: dict[str, dict[str, float]] = {}
    for line in raw.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        term = m.group("term").strip()
        if term not in _TERMS:
            continue
        wm = _parse_float(m.group("wm"))
        we = _parse_float(m.group("we"))
        bm = _parse_float(m.group("bm"))
        be = _parse_float(m.group("be"))
        terms[term] = {
            "White": (wm + we) / 2.0,
            "Black": (bm + be) / 2.0,
        }
    return terms


def get_eval_diff(
    fen_before: str,
    fen_after: str,
    white_moved: bool,
) -> dict[str, float]:
    """Compute per-term delta from the perspective of the side that just moved.

    A positive delta means the moving side improved on that dimension.
    Uses total-board perspective (White − Black advantage) so that both sides
    are measured on the same scale regardless of who moved.

    Args:
        fen_before: Position before the move.
        fen_after:  Position after the move.
        white_moved: True if White just played the move.

    Returns:
        {term_name: delta}  — positive = improved for the moving side.
    """
    before = get_sf15_eval(fen_before)
    after = get_sf15_eval(fen_after)

    diffs: dict[str, float] = {}
    for term in before:
        if term not in after:
            continue
        # Total advantage = White score − Black score (from White's perspective)
        adv_before = before[term]["White"] - before[term]["Black"]
        adv_after = after[term]["White"] - after[term]["Black"]
        delta = adv_after - adv_before
        # Flip for Black: improvement for Black means advantage decreased for White
        if not white_moved:
            delta = -delta
        diffs[term] = round(delta, 2)
    return diffs


if __name__ == "__main__":
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    print(get_eval_diff(start, after_e4, white_moved=True))
