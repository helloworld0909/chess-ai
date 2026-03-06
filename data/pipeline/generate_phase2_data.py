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
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess
import chess.engine

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


def _sf15_annotation(term_diffs: dict[str, float]) -> str:
    """Convert notable SF15 term diffs into a compact annotation string.

    Only terms with |diff| >= 0.10 are included.  Terms are sorted by
    absolute magnitude so the most impactful appear first.
    Example output: "king safety −0.45, mobility +0.21"
    """
    notable = [(abs(d), term, d) for term, d in term_diffs.items() if abs(d) >= 0.10]
    if not notable:
        return ""
    notable.sort(reverse=True)
    parts = []
    for _, term, d in notable[:3]:  # cap at 3 terms to keep annotations concise
        sign = "+" if d >= 0 else "−"
        parts.append(f"{term.lower()} {sign}{abs(d):.2f}")
    return "; ".join(parts)


def _annotate_line_with_sf15(
    moves: list[tuple[str, str]],
    start_fen: str,
    white_to_move_at_start: bool,
) -> list[tuple[str, str, str]]:
    """Walk a line of (san, purpose) pairs and attach SF15 term annotations.

    Returns list of (san, purpose, sf15_annotation) triples.
    sf15_annotation is empty string when SF15 is unavailable or diff is tiny.
    """
    result: list[tuple[str, str, str]] = []
    board = chess.Board(start_fen)
    try:
        eval_before = get_sf15_eval(start_fen)
    except Exception:
        # SF15 not available — return unannotated
        return [(san, purpose, "") for san, purpose in moves]

    for san, purpose in moves:
        # Push the move to get fen_after
        try:
            move = board.parse_san(san)
            white_moved = board.turn == chess.WHITE
            board.push(move)
            fen_after = board.fen()
        except Exception:
            result.append((san, purpose, ""))
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
            annotation = _sf15_annotation(diffs)
        except Exception:
            annotation = ""

        result.append((san, purpose, annotation))
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


def _build_comment_from_lines(
    board: chess.Board,
    move_san: str,
    played_line_info: list[tuple[str, str]],
    engine_lines: list[tuple[list[tuple[str, str]], str, int | None]],
    root_cp: int | None,
    best_cp: int | None,
    student_cp: int | None,
) -> str:
    """Generate a coaching comment grounded in the Stockfish lines.

    This replaces the textbook cache lookup — the comment is derived deterministically
    from the engine analysis, so it accurately reflects what happens in the lines.
    """
    turn = board.turn
    sign = 1 if turn == chess.WHITE else -1

    # Classify the student's move
    classification = "good"
    cp_loss = 0
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

    phase = _game_phase(board)
    turn_str = "White" if turn == chess.WHITE else "Black"

    # Build comment parts
    parts = []

    # Part 1: quality of the played move
    if classification in ("best", "great"):
        parts.append(
            f"You play {move_san}, which is the engine's top choice — "
            f"a {'strong' if classification == 'best' else 'very good'} move in this {phase}."
        )
    elif classification == "good":
        parts.append(f"You play {move_san}, a solid move that keeps the position balanced.")
    elif classification == "inaccuracy":
        best_san = (
            engine_lines[0][0][0][0]
            if engine_lines and engine_lines[0][0]
            else "the engine's suggestion"
        )
        parts.append(
            f"You play {move_san}, but {best_san} was slightly stronger — "
            f"you lose about {cp_loss} centipawns compared to the best continuation."
        )
    else:  # mistake or blunder
        best_san = (
            engine_lines[0][0][0][0]
            if engine_lines and engine_lines[0][0]
            else "the engine's suggestion"
        )
        parts.append(
            f"You play {move_san}, which is a {classification} — "
            f"{best_san} was the correct idea here, saving approximately {cp_loss} centipawns."
        )

    # Part 2: key idea from the best engine line
    if engine_lines:
        best_line_moves, best_eval_label, _ = engine_lines[0]
        if best_line_moves:
            first_move_san, first_purpose = best_line_moves[0]
            if len(best_line_moves) >= 2:
                second_move_san, second_purpose = best_line_moves[1]
                parts.append(
                    f"The engine's top line starts with {first_move_san} ({first_purpose}) "
                    f"followed by {second_move_san} ({second_purpose}), "
                    f"leaving the position {best_eval_label}."
                )
            else:
                parts.append(
                    f"The engine's top line starts with {first_move_san} ({first_purpose}), "
                    f"leaving the position {best_eval_label}."
                )

    # Part 3: comparison if student's line differs from engine best
    if classification not in ("best", "great") and played_line_info and len(played_line_info) >= 2:
        response_san, response_purpose = played_line_info[1]
        parts.append(
            f"After your {move_san}, the opponent's best response is {response_san} "
            f"({response_purpose})."
        )

    return " ".join(parts)


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

    eval_str = (
        "a forced mate sequence exists" if root_cp is None else _cp_to_label(root_cp, board.turn)
    )

    # Determine adaptive depth and number of engine lines
    line_depth, n_engine_lines = _line_params(spread_cp)

    # Get the student's move cp (to classify it)
    student_cp = None
    best_cp = None
    sign = 1 if board.turn == chess.WHITE else -1
    if full_lines:
        _, _, best_cp = full_lines[0]  # best line cp
    # Find student's move in the engine lines
    for moves_info, eval_label, cp in full_lines:
        if moves_info and moves_info[0][0] == move_san:
            student_cp = cp
            break
    # If not found, run a separate analysis for the student's move eval
    if student_cp is None and root_cp is not None:
        engine = _get_engine()
        board_after_copy = board_after.copy()
        try:
            # Analyse from board_after to get side-to-move eval, then flip
            res = engine.analyse(board_after_copy, chess.engine.Limit(depth=max(depth - 4, 8)))
            sc = res.get("score")
            if sc and not sc.is_mate():
                # After student's move, it's opponent's turn; flip to get White's perspective
                student_cp = sc.white().score()
        except Exception:
            pass

    # Build PLAYED LINE: student's move + engine continuation
    engine = _get_engine()
    played_line_moves = _build_played_line(board, move, depth, engine)
    played_line_truncated = played_line_moves[:line_depth]
    played_line_str = " → ".join(m for m, p in played_line_truncated)

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
    # This is the core RL reward signal: rule-generated, grounded, human-readable.
    white_to_move = board.turn == chess.WHITE
    played_line_annotated = _annotate_line_with_sf15(played_line_truncated, fen, white_to_move)
    engine_lines_annotated = [
        (_annotate_line_with_sf15(moves_info, fen, white_to_move), eval_label, cp)
        for moves_info, eval_label, cp in engine_lines_selected
    ]

    # Build prompt key_lines (plain SAN, no sentinels yet)
    key_lines_for_prompt = []
    assistant_lines_list = []

    # LINE 1 is always the PLAYED LINE
    def _fmt_move(san: str, purpose: str, sf15: str) -> str:
        base = f"{san} ({purpose}"
        return f"{base}; {sf15})" if sf15 else f"{base})"

    played_annotated = " → ".join(
        _fmt_move(san, purpose, sf15) for san, purpose, sf15 in played_line_annotated
    )
    # Eval label for played line: use student_cp
    played_eval_label = "equal"
    if student_cp is not None:
        if student_cp >= 200:
            played_eval_label = "winning for white"
        elif student_cp >= 60:
            played_eval_label = "good for white"
        elif student_cp >= -60:
            played_eval_label = "equal"
        elif student_cp >= -200:
            played_eval_label = "good for black"
        else:
            played_eval_label = "winning for black"

    assistant_lines_list.append(
        f"<line>LINE 1: {played_annotated} | eval: {played_eval_label}</line>"
    )

    # Engine alternative lines
    for idx, (ann_moves, eval_label, cp) in enumerate(engine_lines_annotated):
        sans_only = " → ".join(san for san, _, _ in ann_moves)
        key_lines_for_prompt.append(sans_only)

        annotated_moves = " → ".join(
            _fmt_move(san, purpose, sf15) for san, purpose, sf15 in ann_moves
        )
        assistant_lines_list.append(
            f"<line>LINE {idx + 2}: {annotated_moves} | eval: {eval_label}</line>"
        )

    # Use LLM-grounded comment if available, else fall back to deterministic comment
    if llm_comment:
        comment = llm_comment
    else:
        comment = _build_comment_from_lines(
            board,
            move_san,
            played_line_moves,
            [(moves_info, eval_label, cp) for moves_info, eval_label, cp in engine_lines_selected],
            root_cp,
            best_cp,
            student_cp,
        )

    user_content = format_joint_user_prompt(
        board_str,
        fen,
        move_san,
        eval_str,
        facts,
        board_after_str,
        fen_after,
        played_line=played_line_str,
        key_lines=key_lines_for_prompt,
    )

    thinking = _build_thinking(fen, root_cp, candidates, rng)

    assistant_content = f"<think>\n{thinking}\n</think>\n"
    assistant_content += "\n".join(assistant_lines_list)
    assistant_content += f"\n\n{comment}"

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
            "classification": "best" if student_cp == best_cp else "other",
            "spread_cp": spread_cp,
            "n_lines": len(assistant_lines_list),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/processed/train.jsonl")
    parser.add_argument(
        "--comments-cache",
        default=None,
        help="Path to .phase2_comments_cache.jsonl from generate_phase2_comments.py",
    )
    parser.add_argument("--output", default="data/processed/lines_joint_sft.jsonl")
    parser.add_argument("--eval-output", default="data/processed/lines_joint_sft_eval.jsonl")
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--depth", type=int, default=18)
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
        k = hashlib.md5(f"llm10:{fen}:{move_uci}".encode()).hexdigest()
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

    ctx = multiprocessing.get_context("spawn")
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
