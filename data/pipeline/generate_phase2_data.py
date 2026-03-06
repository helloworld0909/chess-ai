"""Generate Phase 2 Joint Task training data from textbook positions.

Changes vs v1:
  - PLAYED LINE always first (even if played move = engine best)
  - Adaptive line depth (4/6/8/10 half-moves) and breadth (2-5 lines) based on position spread
  - Coaching comment generated from Stockfish lines (NOT from textbook cache)

Usage:
    STOCKFISH_PATH=~/.local/bin/stockfish uv run python data/pipeline/generate_phase2_data.py \
        --train-data data/processed/train.jsonl \
        --output data/processed/lines_joint_sft.jsonl \
        --eval-output data/processed/lines_joint_sft_eval.jsonl \
        --eval-split 0.05 \
        --workers 8 \
        --depth 18
"""

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess
import chess.engine

# Resolve Stockfish paths before importing modules that read them at import time.
# Supports both explicit env vars and tilde expansion.
_DEFAULT_SF_PATH = str(Path("~/.local/bin/stockfish").expanduser())
_DEFAULT_SF15_PATH = str(Path("~/.local/bin/stockfish-15").expanduser())
os.environ.setdefault("STOCKFISH_PATH", _DEFAULT_SF_PATH)
os.environ.setdefault("SF15_PATH", _DEFAULT_SF15_PATH)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from tutor.prompts import (
    JOINT_SYSTEM_PROMPT,
    board_ascii,
    format_joint_user_prompt,
    move_facts,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.pipeline.convert_lines_to_sft_thinking import (
    _PIECE_NAMES,
    _build_thinking,
    _cp_to_label,
    _game_phase,
    _get_engine,
    _move_purpose,
)
from data.pipeline.sf15_eval import get_eval_diff, get_sf15_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive line sampling parameters
# ---------------------------------------------------------------------------


def _sf15_terms(term_diffs: dict[str, float]) -> list[tuple[str, float]]:
    """Return notable (term, delta) pairs sorted by absolute magnitude.

    Only includes terms with |diff| >= 0.10, capped at 3.
    """
    notable = [(abs(d), term, d) for term, d in term_diffs.items() if abs(d) >= 0.10]
    if not notable:
        return []
    notable.sort(reverse=True)
    return [(term, d) for _, term, d in notable[:3]]


def _fmt_terms(terms: list[tuple[str, float]]) -> str:
    """Format term list as 'king safety −0.45; mobility +0.21' for user prompt brackets."""
    parts = []
    for term, d in terms:
        sign = "+" if d >= 0 else "−"
        parts.append(f"{term.lower()} {sign}{abs(d):.2f}")
    return "; ".join(parts)


def _annotate_line_with_sf15(
    moves: list[tuple[str, str]],
    start_fen: str,
    white_to_move_at_start: bool,
) -> list[tuple[str, str, list[tuple[str, float]]]]:
    """Walk a line of (san, purpose) pairs and attach SF15 term annotations.

    Returns list of (san, purpose, terms) triples where terms is a list of
    (term_name, delta) pairs sorted by absolute magnitude (empty if SF15
    unavailable or all diffs below threshold).
    """
    result: list[tuple[str, str, list[tuple[str, float]]]] = []
    board = chess.Board(start_fen)
    try:
        eval_before = get_sf15_eval(start_fen)
    except Exception:
        return [(san, purpose, []) for san, purpose in moves]

    for san, purpose in moves:
        try:
            move = board.parse_san(san)
            white_moved = board.turn == chess.WHITE
            board.push(move)
            fen_after = board.fen()
        except Exception:
            result.append((san, purpose, []))
            continue

        try:
            eval_after = get_sf15_eval(fen_after)
            diffs: dict[str, float] = {}
            for term in eval_before:
                if term not in eval_after:
                    continue
                adv_before = eval_before[term]["White"] - eval_before[term]["Black"]
                adv_after = eval_after[term]["White"] - eval_after[term]["Black"]
                delta = adv_after - adv_before
                if not white_moved:
                    delta = -delta
                diffs[term] = round(delta, 2)
            terms = _sf15_terms(diffs)
        except Exception:
            terms = []

        result.append((san, purpose, terms))
        eval_before = eval_after  # chain: next move's "before" is this move's "after"

    return result


def _line_params(spread_cp: int) -> tuple[int, int]:
    """Return (depth_half_moves, n_engine_lines) based on position spread.

    spread_cp: best_cp - worst_cp among top candidates (side-to-move perspective).
    A large spread means the position is sharp/critical — show more lines.
    Depth is capped at 6 half-moves (the model cannot learn from deeper sequences).
    """
    if spread_cp >= 200:
        depth, n_lines = 6, 5
    elif spread_cp >= 100:
        depth, n_lines = 6, 4
    elif spread_cp >= 50:
        depth, n_lines = 5, 3
    else:
        depth, n_lines = 4, 2
    return depth, n_lines


# ---------------------------------------------------------------------------
# Full-line Stockfish analysis
# ---------------------------------------------------------------------------


def _analyze_full_lines(
    fen: str, depth: int, multipv: int = 6
) -> tuple[
    int | None,
    list[tuple[str, str, int | None]],
    list[tuple[list[tuple[str, str]], str, int | None]],
    int,
]:
    """Run Stockfish multipv and return full PV lines.

    Returns:
        root_cp: centipawn eval of position (White's perspective), or None if mate
        candidates: [(san, purpose, cp), ...] for thinking block
        full_lines: [([( san, purpose), ...], eval_label, cp), ...] ordered by engine rank
        spread_cp: best - worst cp spread (side-to-move), 0 if unavailable
    """
    engine = _get_engine()
    board = chess.Board(fen)
    try:
        result = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    except Exception:
        return None, [], [], 0

    candidates = []
    root_cp = None
    full_lines = []
    cp_values = []  # side-to-move perspective, for spread

    sign = 1 if board.turn == chess.WHITE else -1

    for i, info in enumerate(result):
        pv = info.get("pv", [])
        if not pv:
            continue

        score = info.get("score")
        cp = None
        if score is not None:
            if score.is_mate():
                if i == 0:
                    root_cp = None
            else:
                cp_val = score.white().score()
                assert cp_val is not None  # not mate, so score() is always int
                if i == 0:
                    root_cp = cp_val
                cp = cp_val
                cp_values.append(sign * cp_val)  # side-to-move perspective

        # Candidate for thinking block (first move of PV only)
        first_move = pv[0]
        try:
            cand_san = board.san(first_move)
            cand_purpose = _move_purpose(board, first_move)
            candidates.append((cand_san, cand_purpose, cp))
        except Exception:
            continue

        # Full PV line
        line_board = board.copy()
        current_line = []
        for mv in pv:
            try:
                san = line_board.san(mv)
                purpose = _move_purpose(line_board, mv)
                current_line.append((san, purpose))
                line_board.push(mv)
            except Exception:
                break

        # Eval label (White's perspective) for final position
        eval_label = "equal"
        if cp is not None:
            if cp >= 200:
                eval_label = "winning for white"
            elif cp >= 60:
                eval_label = "good for white"
            elif cp >= -60:
                eval_label = "equal"
            elif cp >= -200:
                eval_label = "good for black"
            else:
                eval_label = "winning for black"
        elif score and score.is_mate():
            mate_val = score.white().mate()
            if mate_val is not None and mate_val > 0:
                eval_label = "winning for white"
            else:
                eval_label = "winning for black"

        full_lines.append((current_line, eval_label, cp))

    spread_cp = 0
    if len(cp_values) >= 2:
        spread_cp = max(cp_values) - min(cp_values)

    return root_cp, candidates, full_lines, spread_cp


# Phrase pools per SF15 term × direction (positive delta, negative delta).
# Each pool has 3-4 distinct English phrases; one is chosen via seeded RNG for diversity.
_TERM_PHRASES: dict[str, tuple[list[str], list[str]]] = {
    "Material": (
        ["wins material", "gains a material edge", "captures for free"],
        ["loses material", "gives up material", "concedes a piece"],
    ),
    "Imbalance": (
        [
            "creates a favourable piece imbalance",
            "reaches a better piece configuration",
            "engineers a beneficial exchange",
        ],
        [
            "accepts an unfavourable imbalance",
            "trades into a worse structure",
            "concedes the exchange quality",
        ],
    ),
    "Pawns": (
        [
            "strengthens the pawn structure",
            "solidifies the pawn chain",
            "creates a solid pawn formation",
        ],
        [
            "creates a pawn weakness",
            "weakens the pawn structure",
            "leaves a backward pawn",
            "isolates a pawn",
        ],
    ),
    "Knights": (
        ["centralises the knight", "places the knight on an outpost", "activates the knight"],
        ["leaves the knight passive", "pushes the knight to the rim", "misplaces the knight"],
    ),
    "Bishops": (
        [
            "opens the bishop's diagonal",
            "activates the bishop pair",
            "improves long-range bishop scope",
        ],
        ["blocks the bishop's diagonal", "concedes the bishop pair", "buries the bishop"],
    ),
    "Rooks": (
        ["opens a file for the rook", "activates the rook", "doubles rooks on an open file"],
        ["leaves the rook passive", "keeps the rook bottled up", "closes the rook's file"],
    ),
    "Queens": (
        ["activates the queen", "brings the queen into the game", "centralises the queen"],
        ["restricts the queen", "misplaces the queen", "drives the queen offside"],
    ),
    "Mobility": (
        ["gains piece activity", "frees the pieces", "improves piece coordination"],
        ["restricts piece freedom", "reduces mobility", "cramps the position"],
    ),
    "King safety": (
        ["improves king shelter", "shelters the king behind pawns", "tucks the king away safely"],
        ["exposes the king", "weakens king cover", "opens lines toward the king"],
    ),
    "Threats": (
        ["creates concrete threats", "sets up tactical threats", "generates dangerous counterplay"],
        ["allows opponent threats", "permits tactical shots", "ignores a key threat"],
    ),
    "Passed": (
        ["creates a passed pawn", "advances the passed pawn", "sets up a dangerous passer"],
        ["neutralises the passed pawn", "blockades the passer", "eliminates the passed pawn"],
    ),
    "Space": (
        ["gains central space", "advances in the centre", "claims more space"],
        ["cedes space", "surrenders central control", "yields territory"],
    ),
    "Winnable": (
        ["makes the position more decisive", "creates winning chances", "sharpens the position"],
        ["steers toward a draw", "reduces winning chances", "simplifies toward equality"],
    ),
}


def _interpret_terms(san: str, terms: list[tuple[str, float]], seed: int = 0) -> str:
    """Turn SF15 term deltas into a one-line English interpretation for the thinking block."""
    if not terms:
        return ""
    rng = random.Random(seed)
    dom_term, dom_val = terms[0]
    pos_phrases, neg_phrases = _TERM_PHRASES.get(
        dom_term,
        ([f"improves {dom_term.lower()}"], [f"reduces {dom_term.lower()}"]),
    )
    phrase_pool = pos_phrases if dom_val >= 0 else neg_phrases
    interp = rng.choice(phrase_pool)
    if len(terms) >= 2:
        sec_term, sec_val = terms[1]
        sp, sn = _TERM_PHRASES.get(
            sec_term,
            ([f"improves {sec_term.lower()}"], [f"reduces {sec_term.lower()}"]),
        )
        interp += ", " + rng.choice(sp if sec_val >= 0 else sn)
    return interp


def _build_thinking_v3(
    fen: str,
    root_cp: int | None,
    board: chess.Board,
    played_line_ann: list[tuple[str, str, list[tuple[str, float]]]],
    played_eval_label: str,
    engine_lines_ann: list[tuple[list[tuple[str, str, list[tuple[str, float]]]], str, int | None]],
    best_cp: int | None,
    student_cp: int | None,
    classification: str,
    cp_loss: int,
    move_san: str,
    candidates: list[tuple[str, str, int | None]],
) -> str:
    """Build structured per-line thinking block for RL reward verification.

    Format is designed so key facts (CLASSIFICATION, VERDICT, per-line eval labels)
    are extractable by regex without re-running the engine.
    """
    phase = _game_phase(board)
    turn_str = "white" if board.turn == chess.WHITE else "black"

    def _cp_to_wb_label(cp: int | None) -> str:
        if cp is None:
            return "forced mate"
        if cp >= 200:
            return "winning for white"
        if cp >= 60:
            return "good for white"
        if cp >= -60:
            return "equal"
        if cp >= -200:
            return "good for black"
        return "winning for black"

    eval_label_str = _cp_to_wb_label(root_cp)
    lines = [
        f"POSITION: {phase} | {turn_str} to move | eval: {eval_label_str}",
        f"CLASSIFICATION: {move_san} is {classification}",
        "",
        "LINE analysis:",
    ]

    # Helper: render one annotated line block
    def _render_line(
        label: str,
        ann_moves: list[tuple[str, str, list[tuple[str, float]]]],
        eval_lbl: str,
    ) -> None:
        moves_str = " → ".join(san for san, _, _ in ann_moves)
        lines.append(f"  {label}: {moves_str} | eval: {eval_lbl}")
        for san, purpose, terms in ann_moves:
            if terms:
                terms_str = ", ".join(
                    f"{t.lower()} {'+' if v >= 0 else '−'}{abs(v):.2f}" for t, v in terms
                )
                interp = _interpret_terms(san, terms, seed=hash(fen + san) & 0xFFFFFFFF)
                lines.append(f"    {san}: {terms_str} → {interp}")
            else:
                lines.append(f"    {san}: (no notable term changes)")
        # Net impact: summarise dominant theme from first move of line
        if ann_moves:
            first_san, _, first_terms = ann_moves[0]
            if first_terms:
                net = _interpret_terms(
                    first_san, first_terms, seed=hash(fen + first_san + "net") & 0xFFFFFFFF
                )
                lines.append(f"    net: {net}")
        lines.append("")

    _render_line("PLAYED", played_line_ann, played_eval_label)

    for idx, (ann_moves, eval_lbl, cp) in enumerate(engine_lines_ann):
        _render_line(f"LINE {idx + 1}", ann_moves, eval_lbl)

    # VERDICT line — extractable by regex
    if best_cp is not None and engine_lines_ann:
        best_ann = engine_lines_ann[0]
        best_san = best_ann[0][0][0] if best_ann[0] else "?"
        if cp_loss == 0:
            verdict = f"{move_san} best."
        else:
            best_eval = (
                _cp_to_wb_label(engine_lines_ann[0][2])
                if engine_lines_ann[0][2] is not None
                else ""
            )
            verdict = f"{move_san} is {classification}. {best_san} was better"
            if best_eval:
                verdict += f" ({best_eval})"
            verdict += "."
    else:
        verdict = f"{move_san} is {classification}."
    lines.append(f"VERDICT: {verdict}")

    # COACHING FOCUS
    if classification in ("best", "great"):
        focus = f"reinforce the idea behind {move_san} — student played well."
    elif classification == "good":
        focus = f"note {move_san} is solid; briefly mention what {engine_lines_ann[0][0][0][0] if engine_lines_ann and engine_lines_ann[0][0] else 'the engine'} achieves differently."
    else:
        best_san_focus = (
            engine_lines_ann[0][0][0][0]
            if engine_lines_ann and engine_lines_ann[0][0]
            else "the engine's suggestion"
        )
        focus = f"explain why {best_san_focus} is stronger and what concept {move_san} misses."
    lines.append(f"COACHING FOCUS: {focus}")

    return "\n".join(lines)


def _build_played_line(
    board: chess.Board, move: chess.Move, depth: int, engine: chess.engine.SimpleEngine
) -> list[tuple[str, str]]:
    """Run Stockfish from the position AFTER the student's move to get engine response line.

    Returns [(san, purpose), ...] starting with the student's move itself.
    """
    move_san = board.san(move)
    move_purpose = _move_purpose(board, move)
    result_line = [(move_san, move_purpose)]

    board_after = board.copy()
    try:
        board_after.push(move)
    except Exception:
        return result_line

    # Get engine continuation from position after student's move
    continuation_depth = max(depth - 2, 4)
    try:
        result = engine.analyse(
            board_after,
            chess.engine.Limit(depth=continuation_depth),
            multipv=1,
        )
    except Exception:
        return result_line

    if not result:
        return result_line

    pv = result[0].get("pv", [])
    line_board = board_after.copy()
    for mv in pv:
        try:
            san = line_board.san(mv)
            purpose = _move_purpose(line_board, mv)
            result_line.append((san, purpose))
            line_board.push(mv)
        except Exception:
            break

    return result_line


def convert_sample(
    args_tuple: tuple[dict[str, Any], int, int, str | None],
) -> dict[str, Any] | None:
    sample, depth, seed_offset, llm_comment = args_tuple
    metadata = sample.get("metadata", {})
    fen = metadata.get("fen")
    move_uci = metadata.get("move_uci")
    if not fen or not move_uci:
        return None

    rng = random.Random(hash(fen + move_uci) + seed_offset)

    try:
        board = chess.Board(fen)
        board_str = board_ascii(board)
        move = chess.Move.from_uci(move_uci)
        move_san = board.san(move)
    except Exception:
        return None

    facts = move_facts(board, move)
    board_after = board.copy()
    try:
        board_after.push(move)
    except Exception:
        return None
    board_after_str = board_ascii(board_after)
    fen_after = board_after.fen()

    # Run Stockfish analysis from the pre-move position
    root_cp, candidates, full_lines, spread_cp = _analyze_full_lines(fen, depth=depth, multipv=6)

    if not candidates or not full_lines:
        return None

    # Use white/black perspective for eval_str (not side-to-move relative)
    def _wb_label(cp: int | None) -> str:
        if cp is None:
            return "a forced mate sequence exists"
        if cp >= 200:
            return "winning for white"
        if cp >= 60:
            return "good for white"
        if cp >= -60:
            return "roughly equal"
        if cp >= -200:
            return "good for black"
        return "winning for black"

    eval_str = _wb_label(root_cp)

    # Determine adaptive depth and number of engine lines
    line_depth, n_engine_lines = _line_params(spread_cp)

    # Get the student's move cp and continuation from the already-computed full_lines PV.
    # This avoids a second Stockfish call in the common case (student's move in top-6).
    student_cp = None
    best_cp = None
    played_line_from_pv: list[tuple[str, str]] | None = None
    sign = 1 if board.turn == chess.WHITE else -1
    if full_lines:
        _, _, best_cp = full_lines[0]
    for moves_info, eval_label, cp in full_lines:
        if moves_info and moves_info[0][0] == move_san:
            student_cp = cp
            played_line_from_pv = moves_info
            break

    # Fallback: shallow analysis only when student's move not in top-6
    if student_cp is None and root_cp is not None:
        engine = _get_engine()
        try:
            res = engine.analyse(board_after.copy(), chess.engine.Limit(depth=8))
            sc = res.get("score")
            if sc and not sc.is_mate():
                student_cp = sc.white().score()
            # Build played line from this shallow PV
            pv = res.get("pv", [])
            move_san_str = board.san(move)
            move_purpose_str = _move_purpose(board, move)
            played_fallback = [(move_san_str, move_purpose_str)]
            lb = board_after.copy()
            for mv in pv:
                try:
                    played_fallback.append((lb.san(mv), _move_purpose(lb, mv)))
                    lb.push(mv)
                except Exception:
                    break
            played_line_from_pv = played_fallback
        except Exception:
            pass

    # If still no played line (very rare), build minimal one from the move itself
    if not played_line_from_pv:
        played_line_from_pv = [(board.san(move), _move_purpose(board, move))]

    played_line_truncated = played_line_from_pv[:line_depth]

    # Select engine alternative lines (skip if first move == student's move)
    engine_lines_selected = []
    for moves_info, eval_label, cp in full_lines:
        if not moves_info:
            continue
        first_san = moves_info[0][0]
        if first_san == move_san:
            continue  # skip — this is the played line, already shown as PLAYED LINE
        engine_lines_selected.append((moves_info[:line_depth], eval_label, cp))
        if len(engine_lines_selected) >= n_engine_lines:
            break

    # Annotate each move in each line with SF15 classical eval term diffs.
    # Terms go into the USER PROMPT as ground truth; model reasons over them in thinking.
    white_to_move = board.turn == chess.WHITE
    played_line_annotated = _annotate_line_with_sf15(played_line_truncated, fen, white_to_move)
    engine_lines_annotated = [
        (_annotate_line_with_sf15(moves_info, fen, white_to_move), eval_label, cp)
        for moves_info, eval_label, cp in engine_lines_selected
    ]

    # Compute played eval label — white/black perspective (consistent with engine lines)
    def _cp_to_wb(cp: int | None) -> str:
        if cp is None:
            return "equal"
        if cp >= 200:
            return "winning for white"
        if cp >= 60:
            return "good for white"
        if cp >= -60:
            return "equal"
        if cp >= -200:
            return "good for black"
        return "winning for black"

    played_eval_label = _cp_to_wb(student_cp)

    # Classify student move
    classification = "good"
    cp_loss = 0
    sign = 1 if board.turn == chess.WHITE else -1
    if student_cp is not None and best_cp is not None:
        cp_loss = max(0, sign * (best_cp - student_cp))
        if cp_loss == 0:
            classification = "best"
        elif cp_loss <= 10:
            classification = "great"
        elif cp_loss <= 30:
            classification = "good"
        elif cp_loss <= 100:
            classification = "inaccuracy"
        elif cp_loss <= 300:
            classification = "mistake"
        else:
            classification = "blunder"

    # Build user prompt with SF15 annotations inline — "san [term1 +val; term2 −val]"
    def _fmt_user_move(san: str, terms: list[tuple[str, float]]) -> str:
        if terms:
            return f"{san} [{_fmt_terms(terms)}]"
        return san

    played_prompt_str = " → ".join(
        _fmt_user_move(san, terms) for san, _, terms in played_line_annotated
    )
    key_lines_for_prompt = [
        " → ".join(_fmt_user_move(san, terms) for san, _, terms in ann_moves)
        for ann_moves, _, _ in engine_lines_annotated
    ]

    # Build thinking block: per-line structured analysis
    thinking = _build_thinking_v3(
        fen=fen,
        root_cp=root_cp,
        board=board,
        played_line_ann=played_line_annotated,
        played_eval_label=played_eval_label,
        engine_lines_ann=engine_lines_annotated,
        best_cp=best_cp,
        student_cp=student_cp,
        classification=classification,
        cp_loss=cp_loss,
        move_san=move_san,
        candidates=candidates,
    )

    # Skip samples without an LLM-generated coaching comment — deterministic fallbacks
    # are low quality and pollute the training set.
    if not llm_comment:
        return None
    comment = llm_comment

    user_content = format_joint_user_prompt(
        board_str,
        fen,
        move_san,
        eval_str,
        facts,
        board_after_str,
        fen_after,
        played_line=played_prompt_str,
        key_lines=key_lines_for_prompt,
    )

    assistant_content = f"<think>\n{thinking}\n</think>\n\n{comment}"

    return {
        "messages": [
            {"role": "system", "content": JOINT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": "textbook_joint_sft",
            "fen": fen,
            "move_san": move_san,
            "classification": classification,
            "cp_loss": cp_loss,
            "spread_cp": spread_cp,
            "n_lines": 1 + len(engine_lines_annotated),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/processed/train.jsonl")
    parser.add_argument(
        "--comments-cache",
        default="data/processed/.phase2_comments_cache.jsonl",
        help="Path to .phase2_comments_cache.jsonl from generate_phase2_comments.py",
    )
    parser.add_argument("--output", default="data/processed/lines_joint_sft.jsonl")
    parser.add_argument("--eval-output", default="data/processed/lines_joint_sft_eval.jsonl")
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-mate", action="store_true", help="Include MATE dataset positions"
    )
    parser.add_argument("--mate-comments-cache", default="data/processed/mate_comments_cache.jsonl")
    args = parser.parse_args()

    train_path = Path(args.train_data)

    if not train_path.exists():
        log.error(f"Input not found: {train_path}")
        sys.exit(1)

    # Load LLM comments cache (keyed by md5("llm10:{fen}:{move_uci}"))
    comments_cache: dict[str, str] = {}
    if args.comments_cache:
        comments_path = Path(args.comments_cache)
        if comments_path.exists():
            with comments_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        k = rec.get("_key", "")
                        v = rec.get("comment", "")
                        if k and v:
                            comments_cache[k] = v
                    except json.JSONDecodeError:
                        continue
            log.info("Loaded %d LLM comments from cache", len(comments_cache))
        else:
            log.warning("Comments cache not found: %s", comments_path)

    all_samples = []
    with train_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = rec.get("metadata", {})
            fen = metadata.get("fen")
            move_uci = metadata.get("move_uci")
            if not fen or not move_uci:
                continue
            all_samples.append(rec)

    log.info("Loaded %d textbook positions", len(all_samples))

    if args.include_mate:
        try:
            import re

            from datasets import load_dataset

            mate_comments = {}
            mate_cache_path = Path(args.mate_comments_cache)
            if mate_cache_path.exists():
                with mate_cache_path.open() as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        k = f"{rec['fen']}_{rec['move_uci']}"
                        mate_comments[k] = rec["comment"]
                log.info("Loaded %d MATE LLM comments from cache", len(mate_comments))

            ds = load_dataset("OutFlankShu/MATE_DATASET", data_files="both.zip", split="train")

            def parse_mate_input(input_text: str, output_text: str):
                fen_match = re.search(r'FEN of the given chess board is "(.*?)".', input_text)
                fen = fen_match.group(1) if fen_match else ""

                move_a_match = re.search(
                    r"MoveA:(\w+),\s*(.*?)\s*TacticA:\s*(.*?)\s*MoveB:", input_text, re.DOTALL
                )
                move_b_match = re.search(
                    r"MoveB:(\w+),\s*(.*?)\s*TacticB:\s*(.*)", input_text, re.DOTALL
                )

                if not move_a_match or not move_b_match:
                    return None

                move_a = move_a_match.group(1)
                tactic_a = move_a_match.group(3).strip()
                move_b = move_b_match.group(1)
                tactic_b = move_b_match.group(3).strip()

                return fen, [
                    {"move": move_a, "tactic": tactic_a},
                    {"move": move_b, "tactic": tactic_b},
                ]

            mate_samples = []
            for i, row in enumerate(ds):
                # Process a reasonable subset if needed, but we limit to cached comments
                parsed = parse_mate_input(row["input"], row["output"])
                if not parsed:
                    continue
                fen, moves = parsed
                for m in moves:
                    k = f"{fen}_{m['move']}"
                    if k in mate_comments:
                        mate_samples.append(
                            {
                                "metadata": {
                                    "source": "mate_dataset",
                                    "fen": fen,
                                    "move_uci": m["move"],
                                    "llm_comment": mate_comments[k],
                                }
                            }
                        )

            log.info("Loaded %d MATE positions with cached comments", len(mate_samples))
            all_samples.extend(mate_samples)
        except Exception as e:
            log.warning("Failed to load MATE dataset: %s", e)

    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    n_eval = max(1, int(len(all_samples) * args.eval_split))
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    log.info("Split: %d train / %d eval", len(train_samples), len(eval_samples))

    def _get_llm_comment(rec: dict[str, Any]) -> str | None:
        source = rec.get("metadata", {}).get("source", "")
        if source == "mate_dataset":
            return rec.get("metadata", {}).get("llm_comment")

        if not comments_cache:
            return None

        meta = rec.get("metadata", {})
        fen = meta.get("fen", "")
        move_uci = meta.get("move_uci", "")
        k = hashlib.md5(f"llm19:{fen}:{move_uci}".encode()).hexdigest()
        return comments_cache.get(k)

    train_tagged = [(s, args.depth, args.seed, _get_llm_comment(s), "train") for s in train_samples]
    eval_tagged = [(s, args.depth, args.seed, _get_llm_comment(s), "eval") for s in eval_samples]
    all_tagged = train_tagged + eval_tagged

    llm_comment_count = sum(1 for s, _, _, c, _ in all_tagged if c)
    log.info("LLM comments available for %d / %d samples", llm_comment_count, len(all_tagged))

    train_out = Path(args.output)
    eval_out = Path(args.eval_output)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.parent.mkdir(parents=True, exist_ok=True)

    written = {"train": 0, "eval": 0}
    failed = {"train": 0, "eval": 0}

    ctx = multiprocessing.get_context("fork")
    with (
        train_out.open("w", encoding="utf-8") as f_train,
        eval_out.open("w", encoding="utf-8") as f_eval,
        ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool,
    ):
        futures = {
            pool.submit(convert_sample, (s, depth, seed, comment)): split
            for s, depth, seed, comment, split in all_tagged
        }

        for i, fut in enumerate(as_completed(futures)):
            split = futures[fut]
            try:
                result = fut.result()
                if result:
                    fout = f_train if split == "train" else f_eval
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    written[split] += 1
                else:
                    failed[split] += 1
            except Exception as e:
                log.debug("Sample failed: %s", e)
                failed[split] += 1

            if (i + 1) % 100 == 0:
                log.info(
                    "  %d / %d done (written: %d, failed: %d)",
                    i + 1,
                    len(all_tagged),
                    sum(written.values()),
                    sum(failed.values()),
                )

    log.info(
        "Done. train=%d eval=%d failed_train=%d failed_eval=%d",
        written["train"],
        written["eval"],
        failed["train"],
        failed["eval"],
    )


if __name__ == "__main__":
    main()
