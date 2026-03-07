"""Generate fresh GRPO prompt-only dataset from unseen Lichess positions.

Pulls games from austindavis/lichess_uci (HuggingFace streaming), skips any
FEN already in the SFT training/eval sets, and formats positions with
LINE_GENERATOR_SYSTEM_PROMPT + format_line_generator_prompt.

Output: JSONL with {"prompt": [system_msg, user_msg], "fen": ..., "move_san": ...}
— no assistant turn (GRPOTrainer generates the completions).

Usage:
    uv run python data/pipeline/generate_grpo_prompts.py \\
        --seen-fens data/processed/lines_sft_thinking.jsonl \\
                    data/processed/lines_sft_thinking_eval.jsonl \\
        --output data/processed/grpo_prompts.jsonl \\
        --target 15000 \\
        --workers 8 \\
        --depth 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from tutor.prompts import (
    LINE_GENERATOR_SYSTEM_PROMPT,
    board_ascii,
    format_line_generator_prompt,
    move_facts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", os.path.expanduser("~/.local/bin/stockfish"))

# Game positions to sample: pick one move per game from moves 10-40 (skip opening theory)
MIN_MOVE = 10
MAX_MOVE = 40
# Min pieces to keep (avoid trivial endgames)
MIN_PIECES = 8

# ---------------------------------------------------------------------------
# Stockfish helpers (synchronous, one engine per worker)
# ---------------------------------------------------------------------------

_engine: Optional[chess.engine.SimpleEngine] = None


def _get_engine() -> chess.engine.SimpleEngine:
    global _engine
    if _engine is None:
        _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        return _engine
    try:
        _engine.ping()
    except Exception:
        try:
            _engine.close()
        except Exception:
            pass
        _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    return _engine


def _cp_to_label(cp: int, turn: chess.Color) -> str:
    if turn == chess.BLACK:
        cp = -cp
    if cp >= 200:
        return "winning for me"
    if cp >= 60:
        return "good for me"
    if cp >= -60:
        return "roughly equal"
    if cp >= -200:
        return "slightly worse"
    return "difficult position"


def _eval_fen(fen: str, depth: int = 10) -> str:
    """Return a human-readable eval label for a FEN. Returns 'roughly equal' on error."""
    try:
        engine = _get_engine()
        board = chess.Board(fen)
        result = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = result.get("score")
        if score is None:
            return "roughly equal"
        if score.is_mate():
            return "a forced mate exists"
        cp = score.white().score()
        if cp is None:
            return "roughly equal"
        return _cp_to_label(cp, board.turn)
    except Exception:
        return "roughly equal"


# ---------------------------------------------------------------------------
# Game processing
# ---------------------------------------------------------------------------


def _positions_from_transcript(transcript: str, rng: random.Random) -> list[tuple[str, str, str]]:
    """Replay a game and return candidate (fen, move_san, fen_after) tuples.

    Samples at most 1 position per game from moves MIN_MOVE..MAX_MOVE.
    Returns empty list if no suitable position found.
    """
    moves_uci = transcript.strip().split()
    if len(moves_uci) < MIN_MOVE:
        return []

    board = chess.Board()
    positions: list[tuple[str, str, str]] = []

    for i, uci in enumerate(moves_uci):
        try:
            move = chess.Move.from_uci(uci)
            if not board.is_legal(move):
                break
            if MIN_MOVE <= i < MAX_MOVE:
                fen_before = board.fen()
                # Skip positions with too few pieces
                if len(board.piece_map()) < MIN_PIECES:
                    board.push(move)
                    continue
                san = board.san(move)
                board_copy = board.copy()
                board_copy.push(move)
                fen_after = board_copy.fen()
                positions.append((fen_before, san, fen_after))
            else:
                board.push(move)
                continue
            board.push(move)
        except Exception:
            break

    if not positions:
        return []

    # Pick 1 random position from this game
    chosen = rng.choice(positions)
    return [chosen]


# ---------------------------------------------------------------------------
# Per-sample processing (worker process)
# ---------------------------------------------------------------------------


def process_sample(args_tuple: tuple) -> dict | None:
    """Convert one (fen, move_san, fen_after) to a GRPO prompt row.

    Runs in a worker process — creates its own Stockfish engine.
    """
    fen, move_san, fen_after, depth, seed = args_tuple

    try:
        board = chess.Board(fen)
        board_str = board_ascii(board)
        move = board.parse_san(move_san)
        facts = move_facts(board, move)
        board_after = board.copy()
        board_after.push(move)
        board_after_str = board_ascii(board_after)
    except Exception:
        return None

    eval_str = _eval_fen(fen, depth=depth)

    user_content = format_line_generator_prompt(
        board_str, fen, move_san, eval_str, facts, board_after_str, fen_after
    )

    return {
        "prompt": [
            {"role": "system", "content": LINE_GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "fen": fen,
        "move_san": move_san,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_seen_fens(paths: list[str]) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            log.warning("Seen-FEN file not found: %s", path)
            continue
        with p.open(encoding="utf-8") as f:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seen-fens",
        nargs="+",
        default=[
            "data/processed/lines_sft_thinking.jsonl",
            "data/processed/lines_sft_thinking_eval.jsonl",
        ],
        help="JSONL files whose FENs to exclude",
    )
    parser.add_argument("--output", default="data/processed/grpo_prompts.jsonl")
    parser.add_argument("--target", type=int, default=15000, help="Target number of prompts")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hf-dataset", default="austindavis/lichess_uci")
    parser.add_argument(
        "--max-games",
        type=int,
        default=200_000,
        help="Max games to scan from HF dataset",
    )
    args = parser.parse_args()

    seen_fens = _load_seen_fens(args.seen_fens)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream from HuggingFace
    from datasets import load_dataset

    log.info("Streaming from %s ...", args.hf_dataset)
    ds = load_dataset(args.hf_dataset, split="train", streaming=True)

    rng = random.Random(args.seed)

    # Collect candidate positions
    candidates: list[tuple[str, str, str, int, int]] = []  # (fen, san, fen_after, depth, seed)
    games_scanned = 0
    skipped_seen = 0

    for record in ds:
        if games_scanned >= args.max_games:
            break
        games_scanned += 1

        transcript = record.get("transcript", "")
        positions = _positions_from_transcript(transcript, rng)

        for fen, san, fen_after in positions:
            if fen in seen_fens:
                skipped_seen += 1
                continue
            seen_fens.add(fen)  # deduplicate within run
            candidates.append((fen, san, fen_after, args.depth, args.seed))

        if len(candidates) >= args.target * 3:
            break  # collected enough to select from

        if games_scanned % 10000 == 0:
            log.info(
                "  Scanned %d games — %d candidates (skipped %d seen)",
                games_scanned,
                len(candidates),
                skipped_seen,
            )

    log.info(
        "Scanned %d games — %d candidates total (skipped %d seen)",
        games_scanned,
        len(candidates),
        skipped_seen,
    )

    # Shuffle and trim
    rng.shuffle(candidates)
    candidates = candidates[: args.target * 2]  # process 2x to hit target after failures

    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    written = 0
    failed = 0
    total = len(candidates)

    with (
        out_path.open("w", encoding="utf-8") as fout,
        ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool,
    ):
        futures = {pool.submit(process_sample, c): c for c in candidates}

        for i, fut in enumerate(as_completed(futures)):
            if written >= args.target:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break
            try:
                result = fut.result()
            except Exception as e:
                log.debug("Sample failed: %s", e)
                result = None

            if result is not None:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1
            else:
                failed += 1

            if (i + 1) % 1000 == 0:
                log.info(
                    "  %d / %d processed — written=%d failed=%d", i + 1, total, written, failed
                )

    log.info("Done. Written=%d  Failed=%d  Target=%d", written, failed, args.target)


if __name__ == "__main__":
    main()
