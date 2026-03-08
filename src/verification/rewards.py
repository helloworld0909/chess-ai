"""GRPO reward functions for the chess coaching task.

Each function follows the TRL GRPOTrainer signature:
    reward_fn(prompts, completions, **kwargs) -> list[float]

Rewards (6 total):

    R0  — Format gate: <line> tags present in completion
    R1  — Move legality (python-chess, free)
    R3b — SF15 annotation accuracy: model's think-block term interpretations
          match the actual SF15 diffs shown in the prompt
    R_think — Think block: non-empty, mentions candidate moves
    RC_tone  — Comment: second-person, encouraging, uses chess concepts
    RC_educ  — Comment: grounded moves + causal reasoning

Removed vs previous version:
    R2  eval_accuracy    — gameable (model copies "Engine assessment" label)
    R3a annotation_structural — no-op (parenthetical format not in SFT data)
    R4  depth            — length bias, not quality signal
    R5  breadth          — minor signal, noise at batch size 4
    R6  relevance        — redundant with R1 legality
    RC_fmt comment_format — subsumed by R0 format gate
    RC_conc conciseness  — minor, easy to game with padding
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

_LINE_RE = re.compile(
    r"<line>\s*LINE\s+\d+\s*:\s*(.+?)\|\s*eval\s*:\s*(.+?)\s*</line>"
    r"|LINE\s+\d+\s*:\s*(.+?)\|\s*eval\s*:\s*(.+?)(?:\n|$)"
    r"|(?:PLAYED|LINE\s+\d+)\s*:\s*(.+?)\|\s*eval\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)
_MOVE_RE = re.compile(r"([A-Za-z][A-Za-z0-9\-+#=!?]+|\bO-O(?:-O)?\b)")
_ANNOT_RE = re.compile(r"\([^)]*\)")
_SENTINEL_RE = re.compile(r"<\|[^|>]*\|>")

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
    return abs(_LABEL_DISTANCE.get(a, 2) - _LABEL_DISTANCE.get(b, 2))


def _groups(m: re.Match) -> tuple[str, str]:
    if m.group(1) is not None:
        return m.group(1).strip(), m.group(2).strip().lower()
    if m.group(3) is not None:
        return m.group(3).strip(), m.group(4).strip().lower()
    return m.group(5).strip(), m.group(6).strip().lower()


def parse_lines(text: str) -> list[dict]:
    """Extract structured lines from a model completion.

    Returns list of {moves_san: [str, ...], eval_label: str}.
    """
    results = []
    for m in _LINE_RE.finditer(text):
        body, eval_label = _groups(m)
        bare = _ANNOT_RE.sub("", body)
        bare = _SENTINEL_RE.sub("", bare)
        raw_tokens = re.split(r"[→\s]+", bare)
        moves = [t.strip() for t in raw_tokens if t.strip() and _MOVE_RE.fullmatch(t.strip())]
        if moves:
            results.append({"moves_san": moves, "eval_label": eval_label})
    return results


# ---------------------------------------------------------------------------
# Prompt context extraction
# ---------------------------------------------------------------------------

_FEN_RE = re.compile(r"FEN:\s*(\S+(?:\s+\S+){5})")
_MOVE_PLAYED_RE = re.compile(r"Move(?:\s+played)?:\s*(?:<\|[^|>]*\|>)?(\S+)")


def _extract_context(prompt: str, move_san_override: str = "") -> tuple[str, str, str]:
    """Extract (fen_before, fen_after, move_san) from the user prompt."""
    fens = _FEN_RE.findall(prompt)
    if move_san_override:
        move_san = move_san_override
    else:
        move_m = _MOVE_PLAYED_RE.search(prompt)
        move_san = _SENTINEL_RE.sub("", move_m.group(1)).strip() if move_m else ""

    fen_before = fens[0].strip() if fens else ""
    if len(fens) >= 2:
        fen_after = fens[1].strip()
    elif fen_before and move_san:
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
    if isinstance(prompt, str):
        return prompt
    return "\n".join(m.get("content") or "" for m in prompt if m.get("content"))


def _lines_start_fen(prompt_text: str, move_san: str = "") -> str:
    fen_before, fen_after, _ = _extract_context(prompt_text, move_san_override=move_san)
    if "## Engine Key Lines" in prompt_text:
        return fen_before or fen_after
    return fen_after


# ---------------------------------------------------------------------------
# Stockfish engine pool
# ---------------------------------------------------------------------------

_STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
_ENGINE_POOL: list[Any] = []
_ENGINE_LOCK = threading.Lock()
_POOL_SIZE = 16
_SF_DEPTH = 12


def _get_engine() -> Any:
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


import threading as _threading

_threading.Thread(target=_prewarm_engine_pool, args=(4,), daemon=True).start()

_SF_CACHE: dict[tuple[str, int], int | None] = {}
_SF_CACHE_LOCK = threading.Lock()


def _eval_fen_sync(fen: str, depth: int = 18) -> int | None:
    """Evaluate a FEN with Stockfish (blocking, cached). Returns cp from white's perspective."""
    import chess.engine

    key = (fen, depth)
    with _SF_CACHE_LOCK:
        if key in _SF_CACHE:
            return _SF_CACHE[key]

    engine = _get_engine()
    try:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()
        if score.is_mate():
            result: int | None = 30000 if score.mate() > 0 else -30000  # type: ignore[arg-type]
        else:
            result = score.score()
    except Exception as exc:
        log.debug("Stockfish eval failed for %s: %s", fen, exc)
        result = None
    finally:
        _return_engine(engine)

    with _SF_CACHE_LOCK:
        _SF_CACHE[key] = result
    return result


_SF_EXECUTOR = ThreadPoolExecutor(max_workers=_POOL_SIZE, thread_name_prefix="sf_reward")


# ---------------------------------------------------------------------------
# R0 — Format reward
# ---------------------------------------------------------------------------


def reward_format(
    prompts: list,
    completions: list,
    **kwargs: Any,
) -> list[float]:
    """R0: +1.0 if completion contains ≥1 <line>...</line> block, -1.0 otherwise."""
    scores = []
    for comp in completions:
        text = comp[-1]["content"] if isinstance(comp, list) else str(comp)
        has_line = bool(re.search(r"<line>.*?</line>", text, re.DOTALL))
        scores.append(1.0 if has_line else -1.0)
    return scores


# ---------------------------------------------------------------------------
# R_think — Think block quality
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>|^(.*?)</think>", re.DOTALL)
_THINK_MOVE_RE = re.compile(r"\b([NBRQK][a-h1-8x+#=]?\w*|[a-h][1-8x]\w*|O-O(?:-O)?)\b")


def reward_think(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R_think: rewards a non-trivial <think> block.

    - Present and non-empty                +0.4
    - Mentions ≥2 distinct moves           +0.3
    - Substantive (≥50 chars)              +0.3

    Score range: [-1.0, +1.0]. -1.0 if no think block.
    """
    scores: list[float] = []
    for completion in completions:
        text = _prompt_str(completion)
        m = _THINK_RE.search(text)
        if not m:
            scores.append(-1.0)
            continue
        think = (m.group(1) or m.group(2) or "").strip()
        if not think:
            scores.append(-1.0)
            continue
        moves_in_think = set(_THINK_MOVE_RE.findall(think))
        raw = (
            0.4
            + 0.3 * (1.0 if len(moves_in_think) >= 2 else 0.0)
            + 0.3 * (1.0 if len(think) >= 50 else 0.0)
        )
        scores.append(2.0 * raw - 1.0)
    return scores


# ---------------------------------------------------------------------------
# R1 — Move Legality
# ---------------------------------------------------------------------------


def _legality_score(fen: str, moves_san: list[str]) -> float:
    """Smooth legality score in [-1.0, +1.0]: 2*(legal/total) - 1."""
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
    return 2.0 * (legal_count / len(moves_san)) - 1.0


def reward_legality(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R1: smooth legality score in [-1.0, +1.0], mean over all parsed lines."""
    move_sans = kwargs.get("move_san", [""] * len(prompts))
    scores: list[float] = []
    for prompt, completion, msn in zip(prompts, completions, move_sans):
        prompt_text = _prompt_str(prompt)
        start_fen = _lines_start_fen(prompt_text, move_san=msn)
        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        if not lines:
            scores.append(-1.0)
            continue
        if not start_fen:
            scores.append(0.0)
            continue
        line_scores = [_legality_score(start_fen, line["moves_san"]) for line in lines]
        scores.append(sum(line_scores) / len(line_scores))
    return scores


# ---------------------------------------------------------------------------
# R3b — SF15 annotation accuracy
# ---------------------------------------------------------------------------

# Maps SF15 term names to vocabulary the model might use
# Each term maps to: (positive_words, negative_words)
# Positive = the term improved for the moving side; negative = it got worse.
_SF15_TERM_VOCAB: dict[str, tuple[set[str], set[str]]] = {
    "mobility": (
        {
            "mobility",
            "active",
            "activity",
            "maneuver",
            "maneuvering",
            "flexible",
            "freedom",
            "improves piece",
            "piece activity",
            "coordinate",
            "coordination",
        },
        {"restrict", "restricted", "passive", "limit", "limited", "cramped", "immobile"},
    ),
    "king safety": (
        {
            "king safety",
            "king shelter",
            "shelter",
            "tuck",
            "tucked",
            "safe",
            "castle",
            "castles",
            "castling",
            "protect king",
            "king cover",
            "cover",
        },
        {
            "king danger",
            "exposed king",
            "weakens king",
            "weaken king",
            "king weakness",
            "attack king",
            "open king",
            "king cover",
            "unsafe king",
        },
    ),
    "threats": (
        {
            "threat",
            "threats",
            "counterplay",
            "attack",
            "attacking",
            "pressure",
            "tactical",
            "dangerous",
            "initiative",
            "fork",
            "pin",
            "skewer",
        },
        {"allows threat", "allows attack", "permits", "allows tactical", "opponent threat"},
    ),
    "material": (
        {
            "material",
            "winning material",
            "gains material",
            "up material",
            "capture",
            "wins piece",
            "wins pawn",
        },
        {"loses material", "down material", "sacrifice", "gives up", "loses piece"},
    ),
    "pawns": (
        {
            "pawn structure",
            "pawn majority",
            "passed pawn",
            "pawn chain",
            "pawn advance",
            "strong pawn",
        },
        {"weak pawn", "isolated pawn", "doubled pawn", "backward pawn", "pawn weakness"},
    ),
    "bishops": (
        {"bishop pair", "good bishop", "active bishop", "open diagonal", "diagonal"},
        {"bad bishop", "buried bishop", "buries bishop", "restricted bishop"},
    ),
    "rooks": (
        {
            "open file",
            "rook activity",
            "rook on seventh",
            "seventh rank",
            "connected rooks",
            "rook lift",
        },
        {"passive rook", "closed file", "inactive rook"},
    ),
    "queens": (
        {"queen activity", "active queen", "centralise queen", "queen pressure"},
        {"misplace queen", "misplaces queen", "poor queen", "passive queen"},
    ),
    "space": (
        {
            "central space",
            "space advantage",
            "territory",
            "gain space",
            "gains space",
            "gain central",
            "more space",
            "spatial",
        },
        {
            "cede space",
            "concede space",
            "cramped",
            "restricted space",
            "restricts space",
            "less space",
        },
    ),
    "passed": (
        {"passed pawn", "passer", "advancing pawn", "queening", "promote"},
        {"stop passer", "blockade", "blockaded"},
    ),
    "initiative": (
        {"initiative", "tempo", "lead development", "active", "pressure"},
        {"lose initiative", "passive", "reactive"},
    ),
}

# SF15 term name → canonical key in _SF15_TERM_VOCAB
_TERM_ALIASES: dict[str, str] = {
    "mobility": "mobility",
    "king safety": "king safety",
    "threats": "threats",
    "material": "material",
    "pawns": "pawns",
    "bishops": "bishops",
    "rooks": "rooks",
    "queens": "queens",
    "space": "space",
    "passed": "passed",
    "passed pawns": "passed",
    "initiative": "initiative",
    "imbalance": "material",
}

# Matches SF15 term annotations in the prompt key lines:
# e.g. "Nd5 [mobility +0.32; king safety −0.15]"
# Also handles "mobility +0.32" without brackets (inside think blocks)
_SF15_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_SF15_TERM_RE = re.compile(
    r"(mobility|king safety|threats|material|pawns|bishops|rooks|queens|space|passed(?:\s+pawns)?|initiative|imbalance)"
    r"\s*([+−\-])\s*(\d+\.?\d*)",
    re.IGNORECASE,
)

# Matches a move+bracket annotation in the Engine Key Lines section:
# e.g. "Nd5 [mobility +0.32; king safety −0.15]"
# or   "Nd5 (no notable term changes)"
_KEY_LINE_MOVE_RE = re.compile(r"(\S+)\s+(?:\[([^\]]+)\]|\(([^)]+)\))")


def _parse_prompt_sf15_terms(prompt_text: str) -> dict[str, list[tuple[str, float]]]:
    """Parse SF15 term diffs from the ## Engine Key Lines section of the prompt.

    Returns: {move_san: [(term_name, signed_delta), ...]}
    Only includes moves with at least one notable term (|delta| >= 0.10).
    Terms are from the moving side's perspective (positive = good for mover).
    """
    result: dict[str, list[tuple[str, float]]] = {}
    m = re.search(r"## Engine Key Lines\n\n(.*?)(?=\n\n##|\Z)", prompt_text, re.DOTALL)
    if not m:
        return result
    section = m.group(1)

    for line in section.strip().split("\n"):
        # Parse each move+bracket on this line
        for move_m in _KEY_LINE_MOVE_RE.finditer(line):
            san = move_m.group(1).strip()
            # Strip sentinel tokens
            san = _SENTINEL_RE.sub("", san).strip()
            bracket_content = move_m.group(2) or ""
            if not bracket_content or not san:
                continue
            terms = []
            for term_m in _SF15_TERM_RE.finditer(bracket_content):
                term_raw = term_m.group(1).lower().strip()
                sign_char = term_m.group(2)
                val = float(term_m.group(3))
                sign = -1.0 if sign_char in ("-", "−") else 1.0
                delta = sign * val
                canonical = _TERM_ALIASES.get(term_raw, term_raw)
                if abs(delta) >= 0.10:
                    terms.append((canonical, delta))
            if terms:
                result[san] = terms
    return result


def _model_correctly_interprets(
    think_text: str,
    san: str,
    terms: list[tuple[str, float]],
) -> float:
    """Score how well the model's think block interprets a move's top SF15 term.

    Looks in the think block for context around the SAN, then checks whether
    the top term's direction is correctly captured in the interpretation.

    Returns:
      +1.0  correct direction mentioned (positive term → positive framing)
      -1.0  wrong direction mentioned (positive term → negative framing)
       0.0  term not mentioned at all (neutral — model skipped it)
    """
    # Find the top term by absolute value
    if not terms:
        return 0.0
    top_term, top_delta = max(terms, key=lambda x: abs(x[1]))
    canonical = _TERM_ALIASES.get(top_term, top_term)
    vocab = _SF15_TERM_VOCAB.get(canonical)
    if vocab is None:
        return 0.0
    positive_words, negative_words = vocab

    # Find mentions of this SAN in the think block, then check surrounding context
    # Look for the SAN followed by interpretation text on the same line
    think_lower = think_text.lower()
    # Find lines in think that mention this SAN
    relevant_lines = []
    for tline in think_text.split("\n"):
        if san.lower() in tline.lower() or san in tline:
            relevant_lines.append(tline.lower())

    if not relevant_lines:
        return 0.0  # model didn't mention this move in think

    context = " ".join(relevant_lines)

    # Check for presence of positive or negative vocabulary for this term
    has_positive = any(w in context for w in positive_words)
    has_negative = any(w in context for w in negative_words)

    if not has_positive and not has_negative:
        return 0.0  # mentioned the move but not this term concept

    # top_delta > 0 means term improved for moving side → expect positive framing
    if top_delta > 0:
        if has_positive and not has_negative:
            return 1.0
        elif has_negative and not has_positive:
            return -1.0
        else:
            return 0.0  # mixed signals
    else:
        if has_negative and not has_positive:
            return 1.0
        elif has_positive and not has_negative:
            return -1.0
        else:
            return 0.0


def reward_sf15_annotation(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R3b: SF15 annotation accuracy in the think block.

    For each move in the prompt's Engine Key Lines that has notable SF15 term
    diffs (|delta| >= 0.10), checks whether the model's <think> block interprets
    the top term in the correct direction.

    Per-move score:
      +1.0  correct direction (e.g. mobility +0.32 → model says "improves activity")
      -1.0  wrong direction   (e.g. mobility +0.32 → model says "restricts pieces")
       0.0  term not mentioned (model skipped it — neutral)

    Final score = mean over all moves with notable terms.
    Returns 0.0 if no notable terms found in the prompt (no signal available).
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_str(prompt)

        # Only applies to joint-format prompts with Engine Key Lines
        if "## Engine Key Lines" not in prompt_text:
            scores.append(0.0)
            continue

        sf15_terms = _parse_prompt_sf15_terms(prompt_text)
        if not sf15_terms:
            scores.append(0.0)
            continue

        completion_text = _prompt_str(completion)
        # Extract think block
        think_m = _THINK_RE.search(completion_text)
        think_text = ""
        if think_m:
            think_text = (think_m.group(1) or think_m.group(2) or "").strip()

        if not think_text:
            # No think block → penalise for each move with notable terms
            scores.append(-1.0)
            continue

        move_scores: list[float] = []
        for san, terms in sf15_terms.items():
            s = _model_correctly_interprets(think_text, san, terms)
            move_scores.append(s)

        if not move_scores:
            scores.append(0.0)
        else:
            scores.append(sum(move_scores) / len(move_scores))
    return scores


# ---------------------------------------------------------------------------
# Coaching comment rewards
# ---------------------------------------------------------------------------

_LAST_LINE_END = "</line>"
_COMMENT_TAG_RE = re.compile(r"<comment>(.*?)</comment>", re.DOTALL)

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
    # SF15 terms (also chess concepts)
    "mobility",
    "threats",
    "bishops",
    "rooks",
    "queens",
    # Mechanics
    "castling",
    "castle",
    "promotion",
    "en passant",
    "check",
    "recapture",
}


def _extract_comment(text: str) -> str:
    m = _COMMENT_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    idx = text.rfind(_LAST_LINE_END)
    if idx == -1:
        return ""
    comment = text[idx + len(_LAST_LINE_END) :].strip()
    comment = re.sub(r"</?think>", "", comment).strip()
    return comment


def reward_tone(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """RC_tone: Chess concept vocabulary in coaching comments.

    0 concepts  → -1.0
    1 concept   →  0.0
    2 concepts  → +0.5
    3+ concepts → +1.0

    Returns 0.0 if no comment (neutral).
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


def _comment_moves_grounded(comment: str, lines: list[dict]) -> float:
    """Check whether moves mentioned in comment appear consecutively in some line.

    Score in [-1.0, +1.0]: 2*(grounded_pairs/total_pairs) - 1.
    -1.0 if no moves from lines are mentioned.
    """
    if not lines:
        return 0.0
    comment_moves = _MOVE_RE.findall(comment)
    all_line_sans = {san for line in lines for san in line["moves_san"]}
    comment_moves = [m for m in comment_moves if m in all_line_sans]

    if not comment_moves:
        return -1.0
    if len(comment_moves) == 1:
        return 1.0

    line_sequences = [line["moves_san"] for line in lines]
    total_pairs = len(comment_moves) - 1
    grounded = 0
    for a, b in zip(comment_moves, comment_moves[1:]):
        for seq in line_sequences:
            for i in range(len(seq) - 1):
                if seq[i] == a and seq[i + 1] == b:
                    grounded += 1
                    break
            else:
                continue
            break
    return 2.0 * (grounded / total_pairs) - 1.0


def reward_educational(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """RC_educ: Educational value of the coaching comment.

    1. Grounded moves (consecutive in a line)  +0.4
    2. Chess concept vocabulary                 +0.4
    3. Causal reasoning language               +0.2

    Score range: [-1.0, +1.0]. Returns 0.0 if no comment.
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
            0.4 * ((grounded_score + 1.0) / 2.0)
            + 0.4 * (1.0 if uses_concept else 0.0)
            + 0.2 * (1.0 if has_reasoning else 0.0)
        )
        scores.append(2.0 * raw - 1.0)
    return scores


# ---------------------------------------------------------------------------
# Weights and combined reward (for reference / testing)
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "format": 0.10,  # cold-start gate
    "think": 0.15,  # reasoning quality
    "legality": 1.0,  # hard gate (unweighted — used as gate in combined)
    "sf15": 0.35,  # SF15 term accuracy — primary quality signal
    "tone": 0.20,  # comment chess vocabulary
    "educational": 0.20,  # comment grounding + causal reasoning
}


def combined_reward(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """Combined reward (for reference / testing).

    R1 legality is a hard gate: illegal lines → −1.0 penalty applied.
    All other rewards contribute independently.
    """
    r0 = reward_format(prompts, completions, **kwargs)
    r_think = reward_think(prompts, completions, **kwargs)
    r1 = reward_legality(prompts, completions, **kwargs)
    r3b = reward_sf15_annotation(prompts, completions, **kwargs)
    rc_tone = reward_tone(prompts, completions, **kwargs)
    rc_educ = reward_educational(prompts, completions, **kwargs)

    results = []
    for fmt, think, legal, sf15, tone, educ in zip(r0, r_think, r1, r3b, rc_tone, rc_educ):
        base = (
            _WEIGHTS["format"] * fmt
            + _WEIGHTS["think"] * think
            + _WEIGHTS["sf15"] * sf15
            + _WEIGHTS["tone"] * tone
            + _WEIGHTS["educational"] * educ
        )
        if legal < 0:
            results.append(base - 1.0)
        else:
            results.append(base)
    return results
