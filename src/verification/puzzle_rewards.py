"""Reward functions for Lichess puzzle GRPO training.

Gated reward structure:
  format missing          → reward = -1.0
  format ok, move wrong   → reward =  0.0 + length_penalty
  format ok, move correct → reward =  1.0 + length_penalty

Length penalty (soft, threshold-based):
  - No penalty for completions ≤ THRESHOLD tokens
  - Linear from 0.0 at THRESHOLD to -0.2 at MAX_TOKENS
"""

from __future__ import annotations

import re

import chess

# Accepts any non-empty move string (UCI or SAN) inside {"move": "..."}
_MOVE_RE = re.compile(
    r'\{[^}]*"move"\s*:\s*"([^"]{1,10})"[^}]*\}',
    re.IGNORECASE,
)

THRESHOLD = 100
MAX_TOKENS = 2048


def _extract_uci(text: str) -> str | None:
    """Extract move from {"move": "..."} only if it appears at the end of the completion.

    Rules:
    - Only search after </think> to avoid prompt examples echoed in thinking.
    - The JSON object must be the last non-whitespace content in the text.
    """
    think_end = text.rfind("</think>")
    search_text = text[think_end:] if think_end != -1 else text

    # The JSON must be at the tail — strip trailing whitespace and check
    stripped = search_text.rstrip()
    if not stripped.endswith("}"):
        return None

    # Find the last JSON object and verify it is at the end
    matches = list(_MOVE_RE.finditer(search_text))
    if not matches:
        return None

    last = matches[-1]
    # Everything after the match (in stripped text) must be empty
    after = search_text[last.end():].strip()
    if after:
        return None

    return last.group(1).strip()


def _length_penalty(completion_tokens: int) -> float:
    """Soft length penalty: 0.0 for ≤THRESHOLD, linear to -0.2 at MAX_TOKENS."""
    if completion_tokens <= THRESHOLD:
        return 0.0
    excess = (completion_tokens - THRESHOLD) / (MAX_TOKENS - THRESHOLD)
    return -0.2 * min(excess, 1.0)


def reward_format(completion: str) -> float:
    """Return -1.0 if FINAL ANSWER: <uci> is missing, else 0.0."""
    return 0.0 if _extract_uci(completion) is not None else -1.0


def reward_correct(
    completion: str,
    solution_uci: str,
    completion_tokens: int = 0,
    fen: str = "",
) -> float:
    """Gated reward: -1.0 if no format, 0.0/1.0 based on correctness + length_penalty.

    Accepts both UCI (e2e4) and SAN (Nf3, Qxd5+) move formats.

    Args:
        completion: Model completion text.
        solution_uci: The correct UCI move (e.g. "e2e4").
        completion_tokens: Number of tokens in the completion (for length penalty).
        fen: Board FEN for SAN→UCI conversion (required for SAN moves).

    Returns:
        -1.0 if format missing, else (0.0 or 1.0) + length_penalty.
    """
    predicted_raw = _extract_uci(completion)
    if predicted_raw is None:
        return -1.0

    penalty = _length_penalty(completion_tokens)
    solution_uci = solution_uci.lower().strip()

    predicted = predicted_raw.strip()

    # Direct UCI match
    if predicted.lower() == solution_uci:
        return 1.0 + penalty

    # Try SAN→UCI conversion via python-chess
    if fen:
        try:
            board = chess.Board(fen)
            move = board.parse_san(predicted)
            if move.uci() == solution_uci:
                return 1.0 + penalty
            # Legal SAN but wrong move
            return 0.0 + penalty
        except Exception:
            pass

    # Illegal / unparseable move
    return 0.0 + penalty


def compute_rewards(
    completions: list[str],
    solution_ucis: list[str],
    completion_tokens: list[int] | None = None,
    fens: list[str] | None = None,
) -> list[float]:
    """Compute per-completion rewards.

    Args:
        completions: List of model completions.
        solution_ucis: List of correct UCI moves, one per completion.
        completion_tokens: Optional list of token counts per completion.
        fens: Optional list of board FENs for SAN→UCI conversion.

    Returns:
        List of reward floats in [-1.0, 1.0].
    """
    if completion_tokens is None:
        completion_tokens = [0] * len(completions)
    if fens is None:
        fens = [""] * len(completions)
    return [
        reward_correct(c, s, t, f)
        for c, s, t, f in zip(completions, solution_ucis, completion_tokens, fens)
    ]
