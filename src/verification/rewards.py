"""GRPO reward functions for the chess line generator.

Each function follows the TRL GRPOTrainer signature:
    reward_fn(prompts, completions, **kwargs) -> list[float]

Two reward groups:

Line rewards — verify the <line>LINE N: ...</line> blocks (Goals 1, 3):
    R0  — Format gate: <line> tags present
    R1  — Move legality (python-chess, free)
    R2  — Final position eval accuracy (Stockfish depth 12)
    R3a — Structural annotation accuracy (python-chess, free)
    R4  — Line depth: encourages ≥6 half-moves
    R5  — Line breadth: unique first moves across lines
    R6  — Line relevance: first move legal from post-move FEN
    R_think — Think block: non-empty, mentions candidate moves

Coaching comment rewards — verify the paragraph after </line> (Goals 2, 3, 4):
    RC_fmt   — Comment present after last </line>
    RC_tone  — Second-person, encouraging, not cold/robotic
    RC_conc  — Conciseness: 2-5 sentences
    RC_educ  — Educational: references moves + chess concepts + causal reasoning

Note: The phase3 GRPO (lines_sft_thinking) output has NO coaching comment — the
system prompt says "Output only the <line> blocks after your thinking."
RC_* rewards are wired in for future use but score 0.0 when no comment is present.
R_think rewards the free-form <think> block that precedes the <line> blocks.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import chess

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

# Matches lines inside <line>...</line> tags OR bare "LINE N: ..." lines.
# Group 1 = move body, Group 2 = eval label.
_LINE_RE = re.compile(
    r"<line>\s*LINE\s+\d+\s*:\s*(.+?)\|\s*eval\s*:\s*(.+?)\s*</line>"
    r"|LINE\s+\d+\s*:\s*(.+?)\|\s*eval\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)
_MOVE_RE = re.compile(r"([A-Za-z][A-Za-z0-9\-+#=!?]+|\bO-O(?:-O)?\b)")
_ANNOT_RE = re.compile(r"\([^)]*\)")

# Maps Stockfish cp (white perspective) to our eval label
_CP_BANDS: list[tuple[int, str]] = [
    (300, "winning for white"),
    (100, "good for white"),
    (-100, "equal"),
    (-300, "good for black"),
]


def _cp_to_label(cp: int) -> str:
    for threshold, label in _CP_BANDS:
        if cp > threshold:
            return label
    return "winning for black"


_LABEL_DISTANCE: dict[str, int] = {
    "winning for white": 4,
    "good for white": 3,
    "equal": 2,
    "good for black": 1,
    "winning for black": 0,
}


def _label_distance(a: str, b: str) -> int:
    """Ordinal distance between two eval labels (0 = same)."""
    return abs(_LABEL_DISTANCE.get(a, 2) - _LABEL_DISTANCE.get(b, 2))


def _groups(m: re.Match) -> tuple[str, str]:
    """Return (body, eval_label) from a _LINE_RE match (handles both alt groups)."""
    if m.group(1) is not None:
        return m.group(1).strip(), m.group(2).strip().lower()
    return m.group(3).strip(), m.group(4).strip().lower()


def parse_lines(text: str) -> list[dict]:
    """Extract structured lines from a model completion.

    Handles both <line>LINE N: ...</line> (new format) and bare LINE N: ...
    Returns a list of dicts: {moves_san: [str, ...], eval_label: str}
    Only lines with at least one move are returned.
    """
    results = []
    for m in _LINE_RE.finditer(text):
        body, eval_label = _groups(m)
        # Strip annotations, then collect move tokens
        bare = _ANNOT_RE.sub("", body)
        # Split on → or whitespace, filter move-like tokens
        raw_tokens = re.split(r"[→\s]+", bare)
        moves = [t.strip() for t in raw_tokens if t.strip() and _MOVE_RE.fullmatch(t.strip())]
        if moves:
            results.append({"moves_san": moves, "eval_label": eval_label})
    return results


# ---------------------------------------------------------------------------
# Prompt → (fen_before, fen_after, move_san) extraction
# ---------------------------------------------------------------------------

# Matches "FEN: <fen>" — there are two in the prompt: before and after the move.
_FEN_RE = re.compile(r"FEN:\s*(\S+(?:\s+\S+){5})")
# Matches "Move: <san>"
_MOVE_PLAYED_RE = re.compile(r"Move(?:\s+played)?:\s*(\S+)")


def _extract_context(prompt: str) -> tuple[str, str, str]:
    """Extract (fen_before, fen_after, move_san) from the user prompt.

    The prompt contains two FENs: the position before and after the student's
    move.  Engine lines start from fen_after (the position the opponent sees).
    Falls back to applying move_san to fen_before if fen_after is not present.
    """
    fens = _FEN_RE.findall(prompt)
    move_m = _MOVE_PLAYED_RE.search(prompt)
    move_san = move_m.group(1).strip() if move_m else ""

    fen_before = fens[0].strip() if fens else ""
    if len(fens) >= 2:
        fen_after = fens[1].strip()
    elif fen_before and move_san:
        # Derive fen_after by applying the move
        try:
            board = chess.Board(fen_before)
            mv = board.parse_san(move_san)
            board.push(mv)
            fen_after = board.fen()
        except Exception:
            fen_after = fen_before
    else:
        fen_after = fen_before

    return fen_before, fen_after, move_san


def _prompt_str(prompt: list[dict] | str) -> str:
    """Flatten a prompt (list of messages or plain string) to text."""
    if isinstance(prompt, str):
        return prompt
    return "\n".join(m.get("content") or "" for m in prompt if m.get("content"))


# ---------------------------------------------------------------------------
# Stockfish engine pool
# ---------------------------------------------------------------------------

_STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
_ENGINE_POOL: list[Any] = []
_ENGINE_LOCK = threading.Lock()
_POOL_SIZE = 16  # concurrent Stockfish processes (match max completions per step)
_SF_DEPTH = 12  # eval depth for reward signal; can be overridden by trainer


def _get_engine() -> Any:
    """Borrow a chess.engine.SimpleEngine from the pool (creates if needed)."""
    import chess.engine

    with _ENGINE_LOCK:
        if _ENGINE_POOL:
            return _ENGINE_POOL.pop()
    return chess.engine.SimpleEngine.popen_uci(_STOCKFISH_PATH)


def _return_engine(engine: Any) -> None:
    with _ENGINE_LOCK:
        if len(_ENGINE_POOL) < _POOL_SIZE:
            _ENGINE_POOL.append(engine)
        else:
            try:
                engine.quit()
            except Exception:
                pass


def _prewarm_engine_pool(n: int = 4) -> None:
    """Spawn N Stockfish processes eagerly so the first reward step has no cold-start lag.

    Called once at module import.  Silently skipped if Stockfish is not installed.
    """
    import chess.engine

    engines = []
    for _ in range(n):
        try:
            engines.append(chess.engine.SimpleEngine.popen_uci(_STOCKFISH_PATH))
        except Exception:
            break
    with _ENGINE_LOCK:
        for e in engines:
            if len(_ENGINE_POOL) < _POOL_SIZE:
                _ENGINE_POOL.append(e)
            else:
                try:
                    e.quit()
                except Exception:
                    pass


# Pre-warm a small pool of engines in a background thread so the first GRPO
# step doesn't pay the cold-start cost (each popen_uci ≈ 200 ms).
import threading as _threading

_threading.Thread(target=_prewarm_engine_pool, args=(4,), daemon=True).start()


def _eval_fen_sync(fen: str, depth: int = 18) -> int | None:
    """Evaluate a FEN with Stockfish (blocking). Returns cp from white's perspective."""
    import chess.engine

    engine = _get_engine()
    try:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()
        if score.is_mate():
            return 30000 if score.mate() > 0 else -30000  # type: ignore[arg-type]
        cp = score.score()
        return cp
    except Exception as exc:
        log.debug("Stockfish eval failed for %s: %s", fen, exc)
        return None
    finally:
        _return_engine(engine)


# Thread pool for parallelising Stockfish calls across all lines in a batch.
# reward_eval_accuracy submits one job per (sample, line) pair here.
# _eval_fen_sync is called directly from inside these jobs — no nested submits.
_SF_EXECUTOR = ThreadPoolExecutor(max_workers=_POOL_SIZE, thread_name_prefix="sf_reward")


def _eval_fen(fen: str, depth: int = 18) -> int | None:
    """Evaluate a FEN synchronously (call only from within _SF_EXECUTOR threads)."""
    return _eval_fen_sync(fen, depth)


# ---------------------------------------------------------------------------
# R0 — Format reward (cold-start signal)
# ---------------------------------------------------------------------------


def reward_format(
    prompts: list,
    completions: list,
    **kwargs: Any,
) -> list[float]:
    """R0: reward presence of <line>...</line> tags in the completion.

    This fires before legality is checked and gives the model a gradient
    signal to produce the correct output structure during cold start.
    Score: +1.0 if ≥1 <line> block found, -1.0 otherwise.
    """
    scores = []
    for comp in completions:
        text = comp[-1]["content"] if isinstance(comp, list) else str(comp)
        has_line = bool(re.search(r"<line>.*?</line>", text, re.DOTALL))
        scores.append(1.0 if has_line else -1.0)
    return scores


# ---------------------------------------------------------------------------
# R_think — Think block quality
# ---------------------------------------------------------------------------

# A useful think block should mention candidate moves, not just be empty.
# We check: (a) think block present, (b) ≥2 chess moves mentioned in think,
# (c) thinks about the position (not just copies the prompt back verbatim).
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# SAN move pattern inside think (rough — must look like a chess move)
_THINK_MOVE_RE = re.compile(r"\b([NBRQK][a-h1-8x+#=]?\w*|[a-h][1-8x]\w*|O-O(?:-O)?)\b")


def reward_think(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R_think: rewards a non-trivial <think> block before the <line> outputs.

    A good think block (Goal 1 — Correctness):
    - Is present and non-empty                             +0.4
    - Mentions ≥2 distinct candidate moves                 +0.3
    - References the position context (not just whitespace)+0.3

    Score range: [-1.0, +1.0], mapped from [0, 1].
    Returns -1.0 if no think block at all.
    """
    scores: list[float] = []
    for completion in completions:
        text = _prompt_str(completion)
        m = _THINK_RE.search(text)
        if not m:
            scores.append(-1.0)
            continue
        think = m.group(1).strip()
        if not think:
            scores.append(-1.0)
            continue

        # (a) present and non-empty — already confirmed above
        # (b) ≥2 candidate moves mentioned
        moves_in_think = set(_THINK_MOVE_RE.findall(think))
        has_candidates = len(moves_in_think) >= 2

        # (c) substantive: more than 50 chars of reasoning
        is_substantive = len(think) >= 50

        raw = (
            0.4  # present
            + 0.3 * (1.0 if has_candidates else 0.0)
            + 0.3 * (1.0 if is_substantive else 0.0)
        )
        scores.append(2.0 * raw - 1.0)
    return scores


# ---------------------------------------------------------------------------
# R1 — Move Legality
# ---------------------------------------------------------------------------


def _play_line(fen: str, moves_san: list[str]) -> tuple[bool, chess.Board]:
    """Attempt to play a SAN move sequence from fen.

    Returns (fully_legal: bool, board_after_last_legal_move).
    """
    try:
        board = chess.Board(fen)
    except Exception:
        return False, chess.Board()
    for san in moves_san:
        try:
            move = board.parse_san(san)
            board.push(move)
        except (
            chess.IllegalMoveError,
            chess.InvalidMoveError,
            chess.AmbiguousMoveError,
            ValueError,
        ):
            return False, board
    return True, board


def _legality_score(fen: str, moves_san: list[str]) -> float:
    """Smooth per-line legality score in [-1.0, +1.0].

    +1.0  — all moves legal
    (-1, +1) — partial credit: 2*(legal_moves/total_moves) - 1
    -1.0  — first move already illegal (zero legal moves)

    This gives a continuous gradient signal: the model is rewarded
    proportionally for each additional legal move it produces.
    """
    if not moves_san:
        return -1.0
    try:
        board = chess.Board(fen)
    except Exception:
        return -1.0
    legal_count = 0
    for san in moves_san:
        try:
            board.push(board.parse_san(san))
            legal_count += 1
        except Exception:
            break
    # Map [0, total] → [-1.0, +1.0]
    return 2.0 * (legal_count / len(moves_san)) - 1.0


def reward_legality(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R1: smooth legality score in [-1.0, +1.0], mean over all parsed lines.

    Lines start from fen_after (the position after the student's move), since
    the engine explores what happens from that point onward.

    +1.0 = every move in every line is legal.
    -1.0 = no legal moves at all (or no lines parsed).
    Partial credit for lines where the first N moves are legal before hitting
    an illegal one — gives the model a gradient to push more moves through.
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_str(prompt)
        _, fen_after, _ = _extract_context(prompt_text)
        if not fen_after:
            scores.append(0.0)
            continue

        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        if not lines:
            scores.append(-1.0)
            continue

        line_scores = [_legality_score(fen_after, line["moves_san"]) for line in lines]
        scores.append(sum(line_scores) / len(line_scores))
    return scores


# ---------------------------------------------------------------------------
# R2 — Final Position Evaluation Accuracy
# ---------------------------------------------------------------------------


def _r2_line_score(fen: str, line: dict) -> float:
    """Score a single line's eval label against Stockfish ground truth.

    fen is fen_after (post-move position — where the engine lines start).
    """
    legal, board = _play_line(fen, line["moves_san"])
    if not legal or not line["moves_san"]:
        return -1.0  # illegal line — can't evaluate final position

    final_fen = board.fen()
    cp = _eval_fen(final_fen, depth=_SF_DEPTH)
    if cp is None:
        return 0.0  # Stockfish unavailable — neutral

    gt_label = _cp_to_label(cp)
    model_label = line["eval_label"]
    dist = _label_distance(gt_label, model_label)

    if dist == 0:
        return 1.0
    elif dist == 1:
        return -0.5
    else:
        return -1.0


def reward_eval_accuracy(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R2: Eval label accuracy vs Stockfish ground truth at depth 12.

    +1.0 exact match, -0.5 off by one band, -1.0 two+ bands off.
    Score is mean over all parsed lines (or -1.0 if no lines found).
    Lines are played out from fen_after (post-move position).
    """
    scores: list[float] = []

    # Run all Stockfish evals in parallel via thread pool
    all_items: list[tuple[int, int, str, dict]] = []  # (sample_idx, line_idx, fen, line)
    sample_lines: list[list[dict]] = []

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        prompt_text = _prompt_str(prompt)
        _, fen_after, _ = _extract_context(prompt_text)
        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        sample_lines.append(lines)
        for j, line in enumerate(lines):
            all_items.append((i, j, fen_after, line))

    # Submit all Stockfish jobs
    futures = {
        (i, j): _SF_EXECUTOR.submit(_r2_line_score, fen, line) for i, j, fen, line in all_items
    }

    # Collect results per sample
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        lines = sample_lines[i]
        if not lines:
            scores.append(-1.0)
            continue
        line_scores = []
        for j in range(len(lines)):
            fut = futures.get((i, j))
            if fut is None:
                line_scores.append(0.0)
            else:
                try:
                    line_scores.append(fut.result(timeout=60))
                except Exception:
                    line_scores.append(0.0)
        scores.append(sum(line_scores) / len(line_scores))
    return scores


# ---------------------------------------------------------------------------
# R3a — Structural Annotation Accuracy
# ---------------------------------------------------------------------------

_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _score_annotation_structural(board: chess.Board, move: chess.Move, annotation: str) -> float:
    """Score a single move annotation against python-chess ground truth.

    Checks:
    - capture vs non-capture correctly identified
    - piece name mentioned
    - check flag matches
    - castling / promotion identified
    Returns +1.0 if all correct, -1.0 if any structural fact wrong.
    """
    ann = annotation.lower()
    ann_tokens = set(ann.split())

    is_capture = board.is_capture(move)
    mentions_capture = "capture" in ann_tokens or "captures" in ann_tokens
    if is_capture != mentions_capture:
        return -1.0

    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0.0
    piece_name = _PIECE_NAMES.get(piece.piece_type, "piece")
    if piece_name not in ann_tokens:
        return -1.0

    gives_check = board.gives_check(move)
    mentions_check = "check" in ann_tokens
    if gives_check != mentions_check:
        return -1.0

    is_castling = board.is_castling(move)
    mentions_castle = "castle" in ann or "castles" in ann or "castling" in ann
    if is_castling != mentions_castle:
        return -1.0

    return 1.0


def _parse_lines_with_annotations(text: str) -> list[dict]:
    """Like parse_lines but keeps (move, annotation) pairs per line."""
    results = []
    for m in _LINE_RE.finditer(text):
        body, eval_label = _groups(m)
        # Split on → to get individual move+annotation chunks
        chunks = re.split(r"→", body)
        move_annots: list[tuple[str, str]] = []
        for chunk in chunks:
            chunk = chunk.strip()
            annot_m = re.search(r"\(([^)]*)\)", chunk)
            annot = annot_m.group(1).strip() if annot_m else ""
            bare = _ANNOT_RE.sub("", chunk).strip()
            tokens = bare.split()
            if tokens:
                move_san = tokens[-1]  # last token after stripping annotation
                move_annots.append((move_san, annot))
        if move_annots:
            results.append({"move_annots": move_annots, "eval_label": eval_label})
    return results


def reward_annotation_structural(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R3a: Structural annotation accuracy using python-chess ground truth.

    For each move in each line, checks:
    - capture/non-capture correctly named
    - piece name present
    - check flag matches
    - castling identified
    Score per move: +1.0 correct, -1.0 wrong. Mean over all moves in all lines.
    Lines are replayed from fen_after (post-move position).
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_str(prompt)
        _, fen_after, _ = _extract_context(prompt_text)
        if not fen_after:
            scores.append(0.0)
            continue

        completion_text = _prompt_str(completion)
        lines = _parse_lines_with_annotations(completion_text)
        if not lines:
            scores.append(-1.0)
            continue

        move_scores: list[float] = []
        for line in lines:
            try:
                board = chess.Board(fen_after)
            except Exception:
                continue
            for move_san, annot in line["move_annots"]:
                if not annot:
                    # No annotation provided — skip (don't penalise missing)
                    try:
                        move = board.parse_san(move_san)
                        board.push(move)
                    except Exception:
                        break
                    continue
                try:
                    move = board.parse_san(move_san)
                    s = _score_annotation_structural(board, move, annot)
                    move_scores.append(s)
                    board.push(move)
                except Exception:
                    move_scores.append(-1.0)
                    break  # illegal move — stop this line

        if not move_scores:
            scores.append(0.0)
        else:
            scores.append(sum(move_scores) / len(move_scores))
    return scores


# ---------------------------------------------------------------------------
# R4 — Line Depth
# ---------------------------------------------------------------------------

# Target half-moves per line — lines shorter than this score less
# Start at 2 for curriculum: reward any line that has ≥2 legal moves
_TARGET_DEPTH = 2


def reward_depth(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R4: Reward lines proportional to their depth, capped at _TARGET_DEPTH.

    Only legal lines (first move legal from fen_after) count toward depth.
    Score per legal line = min(len(moves), _TARGET_DEPTH) / _TARGET_DEPTH.
    Final score = mean over legal lines, or -1.0 if no legal lines found.
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_str(prompt)
        _, fen_after, _ = _extract_context(prompt_text)
        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        if not lines:
            scores.append(-1.0)
            continue
        legal_line_scores = []
        for line in lines:
            if fen_after and line["moves_san"]:
                try:
                    board = chess.Board(fen_after)
                    board.parse_san(line["moves_san"][0])
                except Exception:
                    continue  # first move illegal — skip this line
            legal_line_scores.append(min(len(line["moves_san"]), _TARGET_DEPTH) / _TARGET_DEPTH)
        if not legal_line_scores:
            scores.append(-1.0)
        else:
            scores.append(sum(legal_line_scores) / len(legal_line_scores))
    return scores


# ---------------------------------------------------------------------------
# R5 — Line Breadth
# ---------------------------------------------------------------------------


def reward_breadth(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R5: Reward unique first moves across legal lines only.

    unique_ratio = len(set(legal_first_moves)) / len(legal_first_moves)
    +1.0 if all legal lines start with a different move.
    Returns -1.0 if no legal lines found.
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_str(prompt)
        _, fen_after, _ = _extract_context(prompt_text)
        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        if not lines:
            scores.append(-1.0)
            continue
        legal_first_moves = []
        for line in lines:
            if not line["moves_san"]:
                continue
            if fen_after:
                try:
                    board = chess.Board(fen_after)
                    board.parse_san(line["moves_san"][0])
                except Exception:
                    continue  # illegal first move — skip
            legal_first_moves.append(line["moves_san"][0])
        if not legal_first_moves:
            scores.append(-1.0)
            continue
        unique_ratio = len(set(legal_first_moves)) / len(legal_first_moves)
        scores.append(unique_ratio)
    return scores


# ---------------------------------------------------------------------------
# R6 — Line Relevance
# ---------------------------------------------------------------------------


def reward_relevance(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R6: First move of each line must be legal from fen_after.

    Proxy for relevance: the model analyses the actual post-move position.
    +1.0 if all lines start with a legal move from fen_after.
    -1.0 if any line starts with an illegal move.
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_str(prompt)
        _, fen_after, _ = _extract_context(prompt_text)
        if not fen_after:
            scores.append(0.0)
            continue

        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        if not lines:
            scores.append(-1.0)
            continue

        line_scores: list[float] = []
        for line in lines:
            if not line["moves_san"]:
                line_scores.append(-1.0)
                continue
            try:
                board = chess.Board(fen_after)
                board.parse_san(line["moves_san"][0])
                line_scores.append(1.0)
            except Exception:
                line_scores.append(-1.0)
        scores.append(sum(line_scores) / len(line_scores))
    return scores


# ---------------------------------------------------------------------------
# Coaching comment rewards (free, no LLM judge)
# These reward the paragraph that follows the <line>...</line> blocks.
# Goals: Tone (2nd-person, encouraging) | Conciseness | Educational value
# Note: phase3 GRPO (lines_sft_thinking) has NO comment — these score 0.0.
# ---------------------------------------------------------------------------

_LAST_LINE_END = "</line>"

# Chess concept vocabulary — used by reward_tone (concept token presence)
# and reward_educational (concepts + causal reasoning).
_CONCEPT_TOKENS = {
    # Tactics
    "fork",
    "pin",
    "skewer",
    "discovery",
    "discovered attack",
    "double attack",
    "deflection",
    "decoy",
    "overloaded",
    "overloading",
    "interference",
    "zwischenzug",
    "in-between move",
    "zugzwang",
    "stalemate",
    "sacrifice",
    "exchange sacrifice",
    "combination",
    "tactics",
    "tactical",
    "mating net",
    "checkmate",
    "back rank",
    "trapped",
    "hanging",
    # Strategy
    "tempo",
    "initiative",
    "outpost",
    "centre",
    "center",
    "central",
    "space",
    "development",
    "develop",
    "king safety",
    "pawn structure",
    "pawn chain",
    "isolated pawn",
    "doubled pawn",
    "backward pawn",
    "passed pawn",
    "pawn majority",
    "open file",
    "half-open",
    "semi-open",
    "open diagonal",
    "diagonal",
    "battery",
    "rook on seventh",
    "seventh rank",
    "connected rooks",
    "rook lift",
    "fianchetto",
    "bishop pair",
    "bad bishop",
    "good bishop",
    "knight outpost",
    "strong square",
    "weak square",
    "weakness",
    "structural weakness",
    "prophylaxis",
    "restriction",
    "coordination",
    "activity",
    "piece activity",
    "pressure",
    "attack",
    "counterattack",
    "defend",
    "defense",
    "counterplay",
    "compensation",
    "material",
    "material imbalance",
    "exchange",
    "strategic",
    "positional",
    "endgame",
    "middlegame",
    "opening",
    # Moves / mechanics
    "castling",
    "castle",
    "promotion",
    "en passant",
    "check",
    "recapture",
    "zugzwang",
}


def _extract_comment(text: str) -> str:
    """Return the coaching comment paragraph from a model completion.

    The comment is the plain-text paragraph after the last </line> block.
    Returns an empty string if no </line> is found or nothing follows it.
    """
    idx = text.rfind(_LAST_LINE_END)
    if idx == -1:
        return ""
    comment = text[idx + len(_LAST_LINE_END) :].strip()
    # Strip any residual <think> tags that leaked into comment
    comment = re.sub(r"</?think>", "", comment).strip()
    return comment


def reward_comment_format(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """RC_fmt: Reward presence of a non-empty coaching comment after the line blocks.

    +1.0 if a non-empty comment paragraph is found after </line>.
    -1.0 if no lines at all (can't tell where comment begins).
     0.0 if lines found but comment is empty (phase3 GRPO case — neutral).
    """
    scores: list[float] = []
    for completion in completions:
        text = _prompt_str(completion)
        has_line = bool(re.search(r"</line>", text))
        if not has_line:
            scores.append(-1.0)
            continue
        comment = _extract_comment(text)
        scores.append(1.0 if comment else 0.0)
    return scores


def reward_tone(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """RC_tone: Reward chess concept vocabulary in coaching comments.

    A good coaching comment uses chess terminology to explain *why*, not just
    *what*.  Scores by counting distinct concept tokens from _CONCEPT_TOKENS
    found in the comment:

      0 concepts  → -1.0
      1 concept   →  0.0
      2 concepts  → +0.5
      3+ concepts → +1.0

    Returns 0.0 if no comment found (neutral for phase3 GRPO which has no comment).
    """
    scores: list[float] = []
    for completion in completions:
        text = _prompt_str(completion)
        comment = _extract_comment(text)
        if not comment:
            scores.append(0.0)
            continue

        cl = comment.lower()
        found = sum(1 for tok in _CONCEPT_TOKENS if tok in cl)

        if found == 0:
            scores.append(-1.0)
        elif found == 1:
            scores.append(0.0)
        elif found == 2:
            scores.append(0.5)
        else:
            scores.append(1.0)
    return scores


def reward_conciseness(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """RC_conc: Reward comment conciseness — 2-5 sentences, no padding.

    2-5 sentences → +1.0  (optimal)
    1 sentence    → +0.5  (too brief)
    6 sentences   →  0.0
    7+            → -1.0  (padding/repetition)
    Returns 0.0 if no comment found.
    """
    scores: list[float] = []
    for completion in completions:
        text = _prompt_str(completion)
        comment = _extract_comment(text)
        if not comment:
            scores.append(0.0)
            continue
        sentences = [s.strip() for s in re.split(r"[.!?]+", comment) if s.strip()]
        n = len(sentences)
        if 2 <= n <= 5:
            score = 1.0
        elif n == 1:
            score = 0.5
        elif n == 6:
            score = 0.0
        else:
            score = -1.0
        scores.append(score)
    return scores


def _comment_moves_grounded(comment: str, lines: list[dict]) -> float:
    """Check whether moves mentioned in the comment are grounded in the key lines.

    Extracts move-like tokens from the comment in order, then verifies:
    - If ≥2 moves mentioned: every consecutive pair (A, B) in the comment must
      appear as consecutive moves in at least one parsed line.  A pair that can't
      be found → not grounded.  Score = grounded_pairs / total_pairs in [-1, +1].
    - If 1 move mentioned: it must appear somewhere in any line (+1.0) or not (-1.0).
    - If 0 moves mentioned: -1.0 (comment is generic, not grounded).

    This catches hallucinated move sequences like "after Qd6 ... d4" where those
    two moves are not consecutive in any engine line.
    """
    if not lines:
        return 0.0  # no lines to check against — neutral

    # Extract SAN-like tokens from the comment in reading order
    comment_moves = _MOVE_RE.findall(comment)
    # Filter to only moves that actually appear in any line (ignore noise like "a6" coords)
    all_line_sans = {san for line in lines for san in line["moves_san"]}
    comment_moves = [m for m in comment_moves if m in all_line_sans]

    if not comment_moves:
        return -1.0  # mentions no moves from the lines

    if len(comment_moves) == 1:
        # Single move — just needs to be in any line
        return 1.0  # already confirmed above (filtered to all_line_sans)

    # ≥2 moves: check each consecutive pair is a consecutive subsequence in some line
    line_sequences = [line["moves_san"] for line in lines]
    total_pairs = len(comment_moves) - 1
    grounded = 0
    for a, b in zip(comment_moves, comment_moves[1:]):
        for seq in line_sequences:
            # Find a then b immediately after in this line
            for i in range(len(seq) - 1):
                if seq[i] == a and seq[i + 1] == b:
                    grounded += 1
                    break
            else:
                continue
            break  # found in some line

    ratio = grounded / total_pairs  # [0, 1]
    return 2.0 * ratio - 1.0  # map to [-1, +1]


def reward_educational(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """RC_educ: Reward educational value of the coaching comment.

    Checks (free proxies):
    1. Grounded moves: moves mentioned in comment appear consecutively in a line  +0.4
    2. Uses chess concept vocabulary (from _CONCEPT_TOKENS)                       +0.4
    3. Contains causal reasoning language                                          +0.2

    Groundedness (check 1): if the comment mentions moves A then B, those two moves
    must appear consecutively in the same engine line.  Hallucinated sequences that
    don't exist in the lines score -1.0 on this sub-component.

    Score range: [-1.0, +1.0]. Returns 0.0 if no comment found.
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        text = _prompt_str(completion)
        comment = _extract_comment(text)
        if not comment:
            scores.append(0.0)
            continue

        lines = parse_lines(text)
        cl = comment.lower()

        grounded_score = _comment_moves_grounded(comment, lines)
        uses_concept = bool(_CONCEPT_TOKENS & set(cl.split())) or any(
            concept in cl for concept in _CONCEPT_TOKENS if " " in concept
        )
        has_reasoning = bool(
            re.search(
                r"\b(because|since|as\b|so that|which means|allowing|preventing"
                r"|in order to|this (?:gives|creates|opens|loses|wins|allows)|"
                r"enabling|leads to)\b",
                cl,
            )
        )
        raw = (
            0.4 * ((grounded_score + 1.0) / 2.0)  # rescale [-1,+1] → [0,1]
            + 0.4 * (1.0 if uses_concept else 0.0)
            + 0.2 * (1.0 if has_reasoning else 0.0)
        )
        scores.append(2.0 * raw - 1.0)
    return scores


# ---------------------------------------------------------------------------
# Combined reward (for logging/debugging — GRPOTrainer calls each separately)
# ---------------------------------------------------------------------------

# Phase 1 weights.
# R1 (legality) is a hard gate: illegal → -1.0, downstream comment rewards still apply.
# Line rewards total ~0.95 (incl R0 format gate, R_think).
# Comment rewards total 0.25 — score 0.0 when no comment is present (phase3 GRPO).
_WEIGHTS = {
    "format": 0.15,  # important cold-start signal, not a quality measure
    "think": 0.15,  # reasoning quality drives correctness (bumped from 0.10)
    "eval_accuracy": 0.12,  # gameable: model can copy "Engine assessment" label; downweighted
    "annotation_structural": 0.20,  # strongest non-gameable line signal (bumped from 0.12)
    "depth": 0.12,  # need more encouragement for ≥6 half-move lines (bumped from 0.10)
    "breadth": 0.10,  # genuinely different ideas — keep
    "relevance": 0.05,  # soft nudge — keep
    # Coaching comment rewards (total 0.25; 0.0 when no comment present)
    "comment_format": 0.05,
    "tone": 0.08,
    "conciseness": 0.05,
    "educational": 0.07,
}


def combined_reward(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """Combined reward (for reference / testing).

    R1 (legality) is a hard gate: illegal lines → -1.0 for all line rewards.
    Comment rewards always apply independently of legality.
    """
    r0 = reward_format(prompts, completions, **kwargs)
    r_think = reward_think(prompts, completions, **kwargs)
    r1 = reward_legality(prompts, completions, **kwargs)
    r2 = reward_eval_accuracy(prompts, completions, **kwargs)
    r3a = reward_annotation_structural(prompts, completions, **kwargs)
    r4 = reward_depth(prompts, completions, **kwargs)
    r5 = reward_breadth(prompts, completions, **kwargs)
    r6 = reward_relevance(prompts, completions, **kwargs)
    rc_fmt = reward_comment_format(prompts, completions, **kwargs)
    rc_tone = reward_tone(prompts, completions, **kwargs)
    rc_conc = reward_conciseness(prompts, completions, **kwargs)
    rc_educ = reward_educational(prompts, completions, **kwargs)

    results = []
    for (
        fmt,
        think,
        legal,
        eval_acc,
        annot,
        depth,
        breadth,
        relevance,
        cfmt,
        tone,
        conc,
        educ,
    ) in zip(r0, r_think, r1, r2, r3a, r4, r5, r6, rc_fmt, rc_tone, rc_conc, rc_educ):
        comment_score = (
            _WEIGHTS["comment_format"] * cfmt
            + _WEIGHTS["tone"] * tone
            + _WEIGHTS["conciseness"] * conc
            + _WEIGHTS["educational"] * educ
        )
        if legal < 0:
            # Hard gate: illegal lines; format + think + comment still contribute
            results.append(
                _WEIGHTS["format"] * fmt + _WEIGHTS["think"] * think - 1.0 + comment_score
            )
        else:
            results.append(
                _WEIGHTS["format"] * fmt
                + _WEIGHTS["think"] * think
                + _WEIGHTS["eval_accuracy"] * eval_acc
                + _WEIGHTS["annotation_structural"] * annot
                + _WEIGHTS["depth"] * depth
                + _WEIGHTS["breadth"] * breadth
                + _WEIGHTS["relevance"] * relevance
                + comment_score
            )
    return results
