"""Generate GRPO prompt dataset from lines_30k.jsonl (unseen positions).

Converts lines_30k format to lines_joint_sft prompt format:
  - system: JOINT_SYSTEM_PROMPT
  - user: board + FEN + move facts + ## Engine Key Lines (from lines_30k)

Skips FENs already in the joint SFT training/eval sets.
Output: {"messages": [system, user, assistant_placeholder], "metadata": {...}}
  — assistant turn is a placeholder; train.py strips it for GRPO.

Usage:
    uv run python data/pipeline/generate_grpo_joint_prompts.py \\
        --output data/processed/grpo_joint_prompts.jsonl \\
        --target 15000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tutor.prompts import JOINT_SYSTEM_PROMPT, board_ascii, move_facts

_DEFAULT_SF15_PATH = str(Path("~/.local/bin/stockfish-15").expanduser())
os.environ.setdefault("SF15_PATH", _DEFAULT_SF15_PATH)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _load_seen_fens(paths: list[str]) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                    fen = rec.get("metadata", {}).get("fen", "") or rec.get("fen", "")
                    if fen:
                        seen.add(fen)
                except Exception:
                    pass
    log.info("Loaded %d seen FENs", len(seen))
    return seen


def _parse_sans_from_line(moves_part: str) -> list[str]:
    """Extract bare SAN tokens from a lines_30k moves string.

    Input: "O-O (castle kingside) → Bg5 (move bishop) → h6 (move pawn)"
    Output: ["O-O", "Bg5", "h6"]
    """
    sans = []
    for part in moves_part.split("→"):
        # Strip annotation in parens and whitespace
        san = re.sub(r"\s*\([^)]*\)", "", part).strip()
        if san:
            sans.append(san)
    return sans


def _sf15_annotate_sans(fen: str, sans: list[str]) -> list[str]:
    """Run SF15 on each move in sequence, return annotated strings like 'Ne5 [mobility +0.32]'.

    Falls back to bare SAN if SF15 unavailable or move illegal.
    """
    try:
        from data.pipeline.sf15_eval import get_sf15_eval
    except Exception:
        return sans

    result = []
    board = chess.Board(fen)
    for san in sans:
        try:
            eval_before = get_sf15_eval(board.fen())
            mv = board.parse_san(san)
            board.push(mv)
            eval_after = get_sf15_eval(board.fen())
        except Exception:
            result.append(san)
            continue

        if not eval_before or not eval_after:
            result.append(san)
            continue

        white_moved = not board.turn  # board.turn is now the *next* player
        diffs: dict[str, float] = {}
        for term in eval_before:
            if term not in eval_after:
                continue
            # Advantage from the moving side's perspective
            if white_moved:
                delta = (eval_after[term]["White"] - eval_after[term]["Black"]) - (
                    eval_before[term]["White"] - eval_before[term]["Black"]
                )
            else:
                delta = (eval_after[term]["Black"] - eval_after[term]["White"]) - (
                    eval_before[term]["Black"] - eval_before[term]["White"]
                )
            diffs[term] = round(delta, 2)

        notable = sorted(
            [(abs(d), t, d) for t, d in diffs.items() if abs(d) >= 0.10], reverse=True
        )[:3]
        if notable:
            parts = []
            for _, term, d in notable:
                sign = "+" if d >= 0 else "−"
                parts.append(f"{term.lower()} {sign}{abs(d):.2f}")
            result.append(f"{san} [{'; '.join(parts)}]")
        else:
            result.append(san)
    return result


def _stockfish_label_to_joint(label: str) -> str:
    """Normalise lines_30k eval labels to joint_sft format."""
    label = label.strip().lower()
    mapping = {
        "winning for white": "winning for white",
        "good for white": "good for white",
        "equal": "equal",
        "roughly equal": "equal",
        "good for black": "good for black",
        "winning for black": "winning for black",
    }
    return mapping.get(label, "equal")


def _build_user_prompt(fen: str, move_san: str, lines: list[str], eval_str: str) -> str:
    """Build the user message in joint_sft format."""
    board = chess.Board(fen)
    board_str = board_ascii(board)

    try:
        move = board.parse_san(move_san)
        facts = move_facts(board, move)
        board_after = board.copy()
        board_after.push(move)
        board_after_str = board_ascii(board_after)
        fen_after = board_after.fen()
    except Exception:
        facts = []
        board_after_str = ""
        fen_after = ""

    position_section = f"## Position\n\nBoard before the move:\n{board_str}\nFEN: {fen}\n"

    eval_line = f"Engine assessment: {eval_str}\n" if eval_str else ""
    move_section = f"\n## Move Played\n\nMove: {move_san}\n{eval_line}"
    if board_after_str:
        move_section += f"\n\nBoard after the move:\n{board_after_str}\nFEN: {fen_after}\n"

    facts_section = ""
    if facts:
        facts_section = "\n## Verified Move Facts\n\n" + "\n".join(f"- {f}" for f in facts) + "\n"

    # Engine Key Lines — annotate with SF15 term diffs, relabel as PLAYED LINE + Line N
    key_lines_parts = []
    for i, line_str in enumerate(lines):
        # lines_30k format: "LINE N: move (purpose) → move (purpose) | eval: label"
        if "| eval:" in line_str:
            moves_part, eval_part = line_str.rsplit("| eval:", 1)
            eval_label = _stockfish_label_to_joint(eval_part.strip())
        else:
            moves_part = line_str
            eval_label = "equal"

        # Strip "LINE N: " prefix
        moves_part = moves_part.strip()
        moves_part = re.sub(r"^LINE \d+:\s*", "", moves_part)

        # Parse bare SANs then annotate with SF15 term diffs
        sans = _parse_sans_from_line(moves_part)
        annotated = _sf15_annotate_sans(fen, sans)
        annotated_str = " → ".join(annotated)

        if i == 0:
            key_lines_parts.append(f"PLAYED LINE: {annotated_str} | eval: {eval_label}")
        else:
            key_lines_parts.append(f"Line {i}: {annotated_str} | eval: {eval_label}")

    key_lines_section = "\n## Engine Key Lines\n\n" + "\n".join(key_lines_parts) + "\n"

    task_section = (
        "\n## Task\n\n"
        "Analyse each engine key line using the SF15 eval term changes shown, "
        "then give a coaching comment to the student about their move."
    )

    return position_section + move_section + facts_section + key_lines_section + task_section


def _process_record(args_tuple: tuple) -> dict | None:
    """Worker function: build one row. Called in a process pool."""
    fen, move_san, lines = args_tuple
    eval_str = "equal"
    if lines and "| eval:" in lines[0]:
        eval_str = _stockfish_label_to_joint(lines[0].rsplit("| eval:", 1)[1].strip())
    try:
        user_content = _build_user_prompt(fen, move_san, lines, eval_str)
    except Exception:
        return None
    return {
        "messages": [
            {"role": "system", "content": JOINT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "<think>\n</think>\n[placeholder]"},
        ],
        "metadata": {
            "fen": fen,
            "move_san": move_san,
            "source": "grpo_joint_prompts",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/processed/lines_30k.jsonl")
    parser.add_argument(
        "--seen-fens",
        nargs="+",
        default=[
            "data/processed/lines_joint_sft.jsonl",
            "data/processed/lines_joint_sft_eval.jsonl",
        ],
    )
    parser.add_argument("--output", default="data/processed/grpo_joint_prompts.jsonl")
    parser.add_argument("--eval-output", default="data/processed/grpo_joint_prompts_eval.jsonl")
    parser.add_argument("--target", type=int, default=15000)
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    seen_fens = _load_seen_fens(args.seen_fens)

    # Collect candidate records (filter seen FENs first, single-threaded)
    candidates = []
    skipped = 0
    with open(args.source) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            fen = rec.get("fen", "")
            move_san = rec.get("move_san", "")
            lines = rec.get("lines", [])
            if not fen or fen in seen_fens or not lines or not move_san:
                skipped += 1
                continue
            seen_fens.add(fen)
            candidates.append((fen, move_san, lines))
            if len(candidates) >= args.target:
                break

    log.info(
        "Collected %d candidates (skipped=%d), running SF15 with %d workers...",
        len(candidates),
        skipped,
        args.workers,
    )

    from concurrent.futures import ProcessPoolExecutor, as_completed

    rows = []
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_record, c): c for c in candidates}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0:
                log.info("  %d/%d done...", done, len(candidates))
            try:
                result = fut.result()
            except Exception:
                errors += 1
                continue
            if result is None:
                errors += 1
            else:
                rows.append(result)

    log.info("Generated %d rows (errors=%d)", len(rows), errors)

    # Split train/eval
    import random

    random.seed(42)
    random.shuffle(rows)
    n_eval = max(1, int(len(rows) * args.eval_split))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.eval_output, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("Train: %d → %s", len(train_rows), args.output)
    log.info("Eval:  %d → %s", len(eval_rows), args.eval_output)


if __name__ == "__main__":
    main()
