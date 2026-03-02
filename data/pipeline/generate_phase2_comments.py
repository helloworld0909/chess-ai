"""Generate grounded coaching comments for Phase 2 SFT data.

For each textbook position that has a human expert annotation in the coaching cache,
this script calls an LLM to rewrite the expert comment so it is explicitly grounded
in the Stockfish engine lines. The rewritten comment is stored in a new cache.

The new cache is consumed by generate_phase2_data.py via --comments-cache.

Differences from the old coaching cache (llm9:):
  - Comments reference specific moves from the engine key lines
  - The expert annotation is preserved as the primary chess insight source
  - The LLM only *grounds* the comment in lines — it does not invent new analysis
  - Samples with low-quality expert annotations are SKIP'd

Usage:
    STOCKFISH_PATH=~/.local/bin/stockfish uv run python data/pipeline/generate_phase2_comments.py \
        --train-data data/processed/train.jsonl \
        --coaching-cache data/processed/.llm_coaching_cache.jsonl \
        --output data/processed/.phase2_comments_cache.jsonl \
        --llm-url http://localhost:8100/v1 \
        --llm-model Qwen/Qwen3.5-35B-A3B \
        --workers 32 \
        --depth 12
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from tutor.prompts import board_ascii

# Thread-local Stockfish engine — one per thread, avoids asyncio race conditions
_tl = threading.local()
_STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", os.path.expanduser("~/.local/bin/stockfish"))


def _get_thread_engine() -> chess.engine.SimpleEngine:
    if not getattr(_tl, "engine", None):
        _tl.engine = chess.engine.SimpleEngine.popen_uci(_STOCKFISH_PATH)
    return _tl.engine


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chess.engine").setLevel(logging.ERROR)

# Cache version prefix — increment to invalidate old entries
_CACHE_PREFIX = "llm19"
_SKIP = object()  # sentinel: LLM said SKIP — cache it; distinguish from error (None)


# ---------------------------------------------------------------------------
# System prompt — structured think block + coaching comment
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert chess coach. Given a position, engine key lines, and a human expert's "
    "annotation, reverse-engineer the expert's reasoning and produce a structured response.\n\n"
    "You are given:\n"
    "1. The board position and the student's move\n"
    "2. Engine key lines (PLAYED LINE = what follows the student's move; Alt N = alternatives)\n"
    "3. The human expert annotation — the SOLE source of chess truth\n\n"
    "Output format — three sections in this exact order:\n\n"
    "<analysis>\n"
    "Step-by-step reasoning: what chess facts explain why the expert wrote this annotation?\n"
    "Use the engine lines to verify and name specific moves.\n"
    "</analysis>\n\n"
    "<facts>\n"
    "List each atomic chess fact on its own line:\n"
    "  fact_type | key=value | key=value | ...\n\n"
    "Supported fact types:\n"
    "  move_quality | move=<san> | eval=<best|good|inaccuracy|mistake|blunder> | best=<san>\n"
    "               (eval classifies the move; best= is the engine top choice if different)\n"
    "  tactic       | type=<fork|pin|skewer|discovered_attack|double_attack|deflection|"
    "overloading|interference|sacrifice|x_ray|mating_threat|back_rank|zwischenzug>"
    " | move=<san> | targets=<piece@sq,piece@sq>\n"
    "               (targets lists the pieces being attacked or exploited)\n"
    "  material     | move=<san> | wins=<piece>  (or loses=<piece> if losing material)\n"
    "               (name the piece: pawn, knight, bishop, rook, queen)\n"
    "  forced       | sequence=<san→san→san> | reason=<why opponent has no other reply>\n"
    "  threat       | move=<san> | threatens=<san>  (the follow-up that would win)\n"
    "  plan         | idea=<short description> | key_move=<san>\n"
    "</facts>\n\n"
    "<comment>\n"
    "Coaching comment rules:\n"
    "- Rewrite the expert annotation faithfully — preserve every chess idea and claim exactly\n"
    "- Write in second person ('You play Nd5…') and fix any grammatical awkwardness\n"
    "- Use the engine lines ONLY to look up specific move notation when the expert already "
    "refers to a move or variation. Do NOT reference engine lines otherwise.\n"
    "- NEVER invent, infer, or add any chess moves or plans beyond what the expert wrote\n"
    "- ONLY reference moves that also appear in your <facts> section\n"
    "- Each sentence must add NEW information — do NOT rephrase or restate what was already said\n"
    "- No filler: cut any sentence that just says 'this is strong', 'your opponent will struggle', "
    "'this secures an advantage', or similar generic consequences not stated by the expert\n"
    "- Write clearly and simply. 1–3 sentences maximum.\n"
    "</comment>\n\n"
    "SKIP RULES — output exactly SKIP (nothing else) when:\n"
    "- The annotation contains no chess instruction (e.g. 'Forced.', 'Good move.', '0-1')\n"
    "- The annotation is too vague to extract any specific chess fact\n"
)


# ---------------------------------------------------------------------------
# Stockfish helper (synchronous, per-process)
# ---------------------------------------------------------------------------


def _line_params(spread_cp: int) -> tuple[int, int]:
    """Return (half_move_depth, num_lines) based on score spread among candidates."""
    if spread_cp >= 200:
        return 6, 5
    elif spread_cp >= 100:
        return 6, 4
    elif spread_cp >= 50:
        return 5, 3
    else:
        return 4, 2


def _get_key_lines_sync(fen: str, depth: int, multipv: int = 5) -> list[str]:
    """Return plain SAN strings for the top engine lines from this position.

    Uses adaptive depth/breadth based on the score spread among candidates.
    """
    engine = _get_thread_engine()
    board = chess.Board(fen)

    # First pass: get scores to compute spread
    try:
        probe = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    except Exception:
        return []

    scores = []
    for info in probe:
        s = info.get("score")
        if s:
            cp = s.white().score(mate_score=10000)
            if cp is not None:
                scores.append(cp)

    spread = (max(scores) - min(scores)) if len(scores) >= 2 else 0
    move_depth, num_lines = _line_params(spread)

    lines = []
    for info in probe[:num_lines]:
        pv = info.get("pv", [])
        if not pv:
            continue
        b = board.copy()
        sans = []
        for mv in pv[:move_depth]:
            try:
                sans.append(b.san(mv))
                b.push(mv)
            except Exception:
                break
        if sans:
            lines.append(" → ".join(sans))
    return lines


# ---------------------------------------------------------------------------
# LLM call (async)
# ---------------------------------------------------------------------------


async def _rewrite_one(
    client,
    llm_model: str,
    fen: str,
    move_san: str,
    expert_annotation: str,
    played_line: str,
    engine_lines: list[str],
) -> str | object | None:  # str=success, _SKIP=intentional skip, None=error
    """Call the LLM to rewrite the expert comment grounded in engine lines.

    Returns the rewritten comment, or None if SKIP / invalid.
    """
    board = chess.Board(fen)
    board_str = board_ascii(board)

    lines_block = f"PLAYED LINE: {played_line}\n" if played_line else ""
    for i, line in enumerate(engine_lines):
        lines_block += f"Line {i + 1}: {line}\n"

    user_msg = (
        f"## Position\n\n{board_str}\nFEN: {fen}\n\n"
        f"## Move Played\n\nMove: {move_san}\n\n"
        f"## Engine Key Lines\n\n{lines_block}\n"
        f"## Expert Annotation\n\n{expert_annotation}\n\n"
        "Rewrite the expert annotation as a grounded coaching comment that references "
        "specific moves from the key lines above."
    )

    from openai import AsyncOpenAI  # local import to avoid hard dependency at module load

    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=16384,
                temperature=1.0,
                top_p=0.95,
                extra_body={
                    "top_k": 20,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 1.5,
                },
            ),
            timeout=1800.0,
        )
    except Exception as e:
        log.warning("LLM error for %s %s: %s", move_san, fen[:30], e)
        return None

    content = (resp.choices[0].message.content or "").strip()

    # SKIP check — intentional, cache it
    if content.strip().upper() == "SKIP" or not content.strip():
        return _SKIP

    # Must have all three sections
    has_analysis = "<analysis>" in content and "</analysis>" in content
    has_facts = "<facts>" in content and "</facts>" in content
    has_comment = "<comment>" in content and "</comment>" in content

    if not (has_analysis and has_facts and has_comment):
        return None

    # Extract and validate the comment section
    comment_match = re.search(r"<comment>(.*?)</comment>", content, re.DOTALL)
    if not comment_match:
        return None
    comment_text = comment_match.group(1).strip()
    # Strip leading label line ("Coaching comment:" etc.)
    comment_text = re.sub(r"^[^\n]*:\s*\n", "", comment_text).strip()

    if len(comment_text) < 20:
        return None
    if len(content) > 8000:
        return None

    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    train_path = Path(args.train_data)
    cache_path = Path(args.coaching_cache)
    output_path = Path(args.output)

    if not train_path.exists():
        log.error("Train data not found: %s", train_path)
        sys.exit(1)
    if not cache_path.exists():
        log.error("Coaching cache not found: %s", cache_path)
        sys.exit(1)

    # Load old coaching cache (expert annotations)
    old_cache: dict[str, str] = {}
    with cache_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("_key") or rec.get("key") or rec.get("cache_key")
            val = rec.get("coaching") or rec.get("result")
            if not key or not val:
                continue
            # Extract <comment> if present
            if isinstance(val, str):
                if val.strip().upper() == "SKIP" or "<comment>SKIP</comment>" in val:
                    continue
                if "<comment>" in val:
                    c_start = val.find("<comment>") + 9
                    c_end = val.find("</comment>")
                    if c_end > c_start:
                        val = val[c_start:c_end].strip()
                if len(val.strip()) >= 30:
                    old_cache[key] = val

    log.info("Loaded %d usable expert annotations from coaching cache", len(old_cache))

    # Load positions
    all_positions = []
    with train_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = rec.get("metadata", {})
            fen = meta.get("fen")
            move_uci = meta.get("move_uci")
            if not fen or not move_uci:
                continue

            # Look up expert annotation
            key_str = f"llm9:textbook:{fen}:{move_uci}:True"
            k = hashlib.md5(key_str.encode()).hexdigest()
            expert = old_cache.get(k)
            if not expert:
                k2 = hashlib.md5(f"llm9:textbook:{fen}:{move_uci}:False".encode()).hexdigest()
                expert = old_cache.get(k2)
            if not expert:
                continue

            all_positions.append((fen, move_uci, expert))

    log.info("Found %d positions with expert annotations", len(all_positions))

    # Load new cache (resumability)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_cache: dict[str, str] = {}
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    k = rec.get("_key", "")
                    if k and "comment" in rec:
                        new_cache[k] = rec["comment"]  # None = SKIP, str = success
                except json.JSONDecodeError:
                    continue
        log.info("Resuming: %d entries already in output cache", len(new_cache))

    # Filter to only positions not yet processed
    to_process = []
    for fen, move_uci, expert in all_positions:
        cache_key = hashlib.md5(f"{_CACHE_PREFIX}:{fen}:{move_uci}".encode()).hexdigest()
        if cache_key not in new_cache:  # not in cache at all (never processed)
            to_process.append((fen, move_uci, expert, cache_key))

    log.info(
        "%d positions to process (skipping %d cached)",
        len(to_process),
        len(all_positions) - len(to_process),
    )

    if args.limit > 0:
        to_process = to_process[: args.limit]
        log.info("Limiting to %d samples (--limit)", args.limit)

    if not to_process:
        log.info("Nothing to do.")
        return

    client = AsyncOpenAI(base_url=args.llm_url, api_key="dummy")
    semaphore = asyncio.Semaphore(args.workers)

    written = [0]
    skipped = [0]
    failed = [0]
    start_time = time.monotonic()

    cache_file = output_path.open("a", encoding="utf-8")
    cache_lock = asyncio.Lock()

    async def process_one(fen: str, move_uci: str, expert: str, cache_key: str) -> None:
        async with semaphore:
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(move_uci)
                move_san = board.san(move)
            except Exception:
                failed[0] += 1
                return

            # Get engine lines (synchronous Stockfish in thread)
            loop = asyncio.get_event_loop()
            try:
                engine_lines = await loop.run_in_executor(
                    None, _get_key_lines_sync, fen, args.depth, 5
                )
            except Exception:
                engine_lines = []

            # Build PLAYED LINE
            played_line = ""
            if engine_lines:
                # Find the line starting with move_san or just use first move
                for l in engine_lines:
                    if l.startswith(move_san):
                        played_line = l
                        break
            # If not found in engine lines, just use move_san
            if not played_line:
                played_line = move_san

            # Filter engine_lines to exclude played line
            alt_lines = [l for l in engine_lines if not l.startswith(move_san)][:4]

            # Call LLM
            comment = await _rewrite_one(
                client,
                args.llm_model,
                fen,
                move_san,
                expert,
                played_line,
                alt_lines,
            )

            if comment is None:
                # LLM error — do not cache, will retry on next run
                failed[0] += 1
            else:
                # Success (str) or intentional SKIP sentinel — cache both
                cache_comment = None if comment is _SKIP else comment
                entry = {
                    "_key": cache_key,
                    "fen": fen,
                    "move_uci": move_uci,
                    "comment": cache_comment,
                }
                async with cache_lock:
                    cache_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    cache_file.flush()
                if comment is _SKIP:
                    skipped[0] += 1
                else:
                    written[0] += 1

            total = written[0] + skipped[0] + failed[0]
            if total % 100 == 0:
                elapsed = time.monotonic() - start_time
                rate = total / elapsed if elapsed > 0 else 0
                remaining = len(to_process) - total
                eta_s = remaining / rate if rate > 0 else 0
                eta_str = f"{int(eta_s // 3600)}h{int(eta_s % 3600 // 60)}m" if eta_s > 0 else "?"
                log.info(
                    "  %d / %d done (written=%d skipped=%d failed=%d) %.1f/min ETA %s",
                    total,
                    len(to_process),
                    written[0],
                    skipped[0],
                    failed[0],
                    rate * 60,
                    eta_str,
                )

    tasks = [
        process_one(fen, move_uci, expert, cache_key)
        for fen, move_uci, expert, cache_key in to_process
    ]
    await asyncio.gather(*tasks)

    cache_file.close()
    log.info(
        "Done. written=%d skipped=%d failed=%d",
        written[0],
        skipped[0],
        failed[0],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/processed/train.jsonl")
    parser.add_argument("--coaching-cache", default="data/processed/.llm_coaching_cache.jsonl")
    parser.add_argument("--output", default="data/processed/.phase2_comments_cache.jsonl")
    parser.add_argument("--llm-url", default="http://localhost:8100/v1")
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="Stockfish depth for key lines (lighter than training depth)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process only this many samples (0 = all, for testing)"
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
