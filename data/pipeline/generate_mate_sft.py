"""Convert MATE dataset + cached coaching comments into SFT JSONL.

Pipeline:
  1. generate_mate_comments.py  →  mate_comments_cache.jsonl
  2. this script               →  lines_mate_sft.jsonl

Only positions whose coaching comment is already cached are processed.
The cache determines which rows to process — no hardcoded row limit needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess
from datasets import load_dataset

from data.pipeline.generate_mate_comments import parse_mate_input
from data.pipeline.generate_phase2_data import convert_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_MATE_DATASET_ID = (
    "OutFlankShu/MATE_NAACL2025_Explore-the-Reasoning-Capability-of-LLMs-in-the-Chess-Testbed"
)


def _cache_key(fen: str, move_uci: str) -> str:
    return f"{fen}||{move_uci}"


def process_mate_row(
    args_tuple: tuple[dict[str, Any], str, dict[str, str | None], int, int],
) -> list[dict[str, Any]]:
    """Worker: convert one MATE row into SFT samples.

    The comments_cache is passed as a dict so it's only serialized once per
    submitted task (not shared state), keeping things safe for multiprocessing.
    """
    row, cache_path_str, comments_cache, depth, seed_offset = args_tuple
    parsed = parse_mate_input(row["input"], row["output"])
    if not parsed:
        return []

    fen, moves = parsed
    results = []
    for m in moves:
        key = _cache_key(fen, m["move"])
        if key not in comments_cache:
            continue
        llm_comment = comments_cache[key]
        if llm_comment is None:
            # Cached SKIP — still build the sample without a coaching comment
            pass

        sample: dict[str, Any] = {
            "metadata": {"fen": fen, "move_uci": m["move"], "source": "mate_dataset"}
        }
        res = convert_sample((sample, depth, seed_offset, llm_comment))
        if res:
            res["metadata"]["strategy"] = m["strategy"]
            res["metadata"]["tactic"] = m["tactic"]
            results.append(res)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comments-cache", default="data/processed/mate_comments_cache.jsonl")
    parser.add_argument("--output", default="data/processed/lines_mate_sft.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load comments cache
    comments_cache: dict[str, str | None] = {}
    cache_path = Path(args.comments_cache)
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = _cache_key(rec["fen"], rec["move_uci"])
                    comments_cache[key] = rec.get("comment")  # None = SKIP
                except Exception:
                    pass
    log.info("Loaded %d cached comments", len(comments_cache))
    if not comments_cache:
        log.warning("No cached comments found — run generate_mate_comments.py first.")
        return

    # Build set of (fen, move_uci) pairs we actually need from the dataset
    needed_fens: set[str] = set()
    for key in comments_cache:
        fen = key.split("||")[0]
        needed_fens.add(fen)
    log.info("Need %d unique FENs from dataset", len(needed_fens))

    ds = load_dataset(_MATE_DATASET_ID, split="train")
    log.info("Dataset: %d rows total", len(ds))

    # Filter to only rows whose FEN we need (avoid parsing all 1M rows)
    needed_rows: list[dict[str, Any]] = []
    for row in ds:
        parsed = parse_mate_input(row["input"], row["output"])
        if not parsed:
            continue
        fen, _ = parsed
        if fen in needed_fens:
            needed_rows.append(dict(row))
    log.info("Rows to process: %d", len(needed_rows))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    ctx = multiprocessing.get_context("spawn")
    task_args = [
        (row, args.comments_cache, comments_cache, args.depth, args.seed + i)
        for i, row in enumerate(needed_rows)
    ]

    with (
        out_path.open("w", encoding="utf-8") as fout,
        ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool,
    ):
        futures = {pool.submit(process_mate_row, a): i for i, a in enumerate(task_args)}
        for i, fut in enumerate(as_completed(futures)):
            for r in fut.result():
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += 1
            if (i + 1) % 500 == 0:
                log.info(
                    "Processed %d / %d rows, written %d samples", i + 1, len(needed_rows), written
                )

    log.info("Done. Written %d total MATE SFT samples.", written)


if __name__ == "__main__":
    main()
