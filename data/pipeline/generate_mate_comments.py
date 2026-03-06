"""Generate LLM coaching comments for MATE dataset positions.

For each position in the MATE dataset we have two candidate moves (the better
and the worse one) plus expert strategy/tactic rationales.  We ask the LLM to
rewrite those rationales into a unified coaching comment and cache the result.

Cache format (one JSON per line):
    {"fen": "...", "move_uci": "...", "comment": "..." | null}
null comment = LLM said SKIP or position was unsuitable; will not be retried.
Missing key  = error during inference; will be retried on next run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# HuggingFace dataset identifier (MATE NAACL 2025)
_MATE_DATASET_ID = (
    "OutFlankShu/MATE_NAACL2025_Explore-the-Reasoning-Capability-of-LLMs-in-the-Chess-Testbed"
)

_SKIP = object()  # sentinel: LLM said SKIP — cache with comment=null

MATE_PROMPT_TEMPLATE = """\
You are an expert chess coach. You are provided with a game position, \
a move played by the student, and the human expert's strategic and tactical rationale for this move.

Your task is to reformat this expert rationale into a 2-4 sentence coaching comment \
addressed to the student (using second person, e.g., "You play...").
Do NOT mention the words "Strategy" or "Tactic".
Seamlessly blend the positional/strategic idea with the concrete tactical sequence.
If the position or rationale is unclear or trivial, respond with exactly: SKIP

Position FEN: {fen}
Student Move: {move} ({classification})
Expert Strategy: {strategy}
Expert Tactic: {tactic}

Write only the coaching comment (or SKIP).
"""


def parse_mate_input(input_text: str, output_text: str) -> tuple[str, list[dict[str, str]]] | None:
    """Parse a MATE dataset row into (fen, [move_info, ...])."""
    fen_match = re.search(r'FEN of the given chess board is "(.*?)".', input_text)
    if not fen_match:
        return None
    fen = fen_match.group(1)

    move_a_match = re.search(
        r"MoveA:(\w+),\s*(.*?)\s*TacticA:\s*(.*?)\s*MoveB:", input_text, re.DOTALL
    )
    if not move_a_match:
        return None
    move_a = move_a_match.group(1)
    strat_a = move_a_match.group(2).strip()
    tactic_a = move_a_match.group(3).strip()

    move_b_match = re.search(r"MoveB:(\w+),\s*(.*?)\s*TacticB:\s*(.*)", input_text, re.DOTALL)
    if not move_b_match:
        return None
    move_b = move_b_match.group(1)
    strat_b = move_b_match.group(2).strip()
    tactic_b = move_b_match.group(3).strip()

    best = "MoveA" if "MoveA" in output_text else "MoveB"
    return fen, [
        {
            "move": move_a,
            "strategy": strat_a,
            "tactic": tactic_a,
            "classification": "Best Move" if best == "MoveA" else "Mistake",
        },
        {
            "move": move_b,
            "strategy": strat_b,
            "tactic": tactic_b,
            "classification": "Best Move" if best == "MoveB" else "Mistake",
        },
    ]


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from vLLM reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def fetch_comment(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
) -> str | object | None:
    """Return coaching comment string, _SKIP sentinel, or None on error."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    try:
        async with session.post(
            url, headers={"Content-Type": "application/json"}, json=payload
        ) as resp:
            data = await resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            text = _strip_thinking(raw)
            if text.strip().upper() == "SKIP":
                return _SKIP
            return text
    except Exception as e:
        log.debug("Inference error: %s", e)
        return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8100/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B-Instruct")
    parser.add_argument("--output", default="data/processed/mate_comments_cache.jsonl")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    cache: set[str] = set()  # keys already processed (including SKIPs)
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache.add(f"{rec['fen']}||{rec['move_uci']}")
                except Exception:
                    pass
    log.info("Cache: %d entries already done", len(cache))

    ds = load_dataset(_MATE_DATASET_ID, split="train")
    rows = list(ds)
    if args.limit:
        rows = rows[: args.limit]
    log.info("Dataset: %d rows", len(rows))

    # Build work list — skip already-cached keys
    work: list[tuple[str, str, str]] = []  # (fen, move_uci, prompt)
    for row in rows:
        parsed = parse_mate_input(row["input"], row["output"])
        if not parsed:
            continue
        fen, moves = parsed
        for m in moves:
            key = f"{fen}||{m['move']}"
            if key in cache:
                continue
            prompt = MATE_PROMPT_TEMPLATE.format(
                fen=fen,
                move=m["move"],
                classification=m["classification"],
                strategy=m["strategy"],
                tactic=m["tactic"],
            )
            work.append((fen, m["move"], prompt))

    log.info("Work queue: %d positions to process", len(work))
    if not work:
        log.info("Nothing to do.")
        return

    sem = asyncio.Semaphore(args.workers)
    done = [0]
    failed = [0]
    start_time = time.monotonic()

    async def bounded(fen: str, move_uci: str, prompt: str) -> tuple[str, str, object]:
        async with sem:
            result = await fetch_comment(session, args.url, args.model, prompt)
            return fen, move_uci, result

    conn = aiohttp.TCPConnector(limit=args.workers + 4)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [bounded(fen, uci, prompt) for fen, uci, prompt in work]
        with out_path.open("a") as f:
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                fen, move_uci, result = await coro
                if result is None:
                    failed[0] += 1
                else:
                    comment_val = None if result is _SKIP else result
                    rec = {"fen": fen, "move_uci": move_uci, "comment": comment_val}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[0] += 1

                if done[0] % 200 == 0:
                    elapsed = time.monotonic() - start_time
                    rate = done[0] / elapsed * 60
                    remaining = len(work) - done[0]
                    eta_min = remaining / (done[0] / elapsed) / 60
                    log.info(
                        "%d / %d done (failed=%d) %.1f/min ETA %.0fh %02.0fm",
                        done[0],
                        len(work),
                        failed[0],
                        rate,
                        eta_min // 60,
                        eta_min % 60,
                    )

    log.info("Done. %d written, %d failed (will retry next run).", done[0] - failed[0], failed[0])


if __name__ == "__main__":
    asyncio.run(main())
