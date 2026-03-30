"""Generate encoder pre-training dataset with SF15 classical eval term labels.

Reads FENs from existing encoder_pretrain_1b.jsonl (or any JSONL with "fen" field)
and annotates each with SF15 per-term scores. Labels are 13-dimensional vectors:
one value per SF15 term, computed as (White_score - Black_score) averaged over MG+EG.

This is the pre-training signal for the 64-token spatial encoder with cross-attention
readout: 13 learned query vectors attend over 64 per-square tokens to predict
complex positional evaluation terms (King safety, Mobility, Threats, Pawns, etc.).
Each term requires understanding of global board relationships — far richer than
piece classification or single-scalar eval.

SF15 terms (13):
    Material, Imbalance, Pawns, Knights, Bishops, Rooks, Queens,
    Mobility, King safety, Threats, Passed, Space, Winnable

Label format:
    sf15_terms: [white_term - black_term for term in TERMS]  — 13 values
    eval_score: total SF15 classical eval (White - Black), pawn units
    Positive = White advantage. Range roughly (-5, 5) pawn units.

Usage:
    # Annotate from existing encoder pretrain FENs (fast, reuses positions)
    SF15_PATH=$HOME/.local/bin/stockfish-15 \\
    uv run python data/pipeline/generate_encoder_pretrain_sf15.py \\
        --source data/processed/encoder_pretrain_1b.jsonl \\
        --output data/processed/encoder_pretrain_sf15.jsonl \\
        --eval-output data/processed/encoder_pretrain_sf15_eval.jsonl \\
        --limit 5000000 --workers 8

    # Quick test
    SF15_PATH=$HOME/.local/bin/stockfish-15 \\
    uv run python data/pipeline/generate_encoder_pretrain_sf15.py --limit 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# SF15 terms in canonical order — index matches label vector position
SF15_TERMS = [
    "Material",
    "Imbalance",
    "Pawns",
    "Knights",
    "Bishops",
    "Rooks",
    "Queens",
    "Mobility",
    "King safety",
    "Threats",
    "Passed",
    "Space",
    "Winnable",
]
N_TERMS = len(SF15_TERMS)


def _compute_labels(fen: str) -> tuple[list[float], float] | None:
    """Return (SF15 term label vector, total eval score) for a FEN, or None on failure.

    sf15_terms: 13 values, each = White_score - Black_score for that term.
    eval_score: sum of all terms (White - Black total classical eval), pawn units.

    Imports sf15_eval lazily so the module-level subprocess is created in the
    worker process (not the parent), which is required for multiprocessing.
    """
    from data.pipeline.sf15_eval import get_sf15_eval

    try:
        terms = get_sf15_eval(fen)
    except Exception as exc:
        log.debug("SF15 failed for %s: %s", fen, exc)
        return None

    labels: list[float] = []
    total = 0.0
    for term in SF15_TERMS:
        if term in terms:
            white = terms[term].get("White", 0.0)
            black = terms[term].get("Black", 0.0)
            diff = round(white - black, 4)
        else:
            diff = 0.0
        labels.append(diff)
        total += diff
    return labels, round(total, 4)


def _iter_fens(source_path: str, limit: int, offset: int = 0, stride: int = 1):
    """Yield unique FENs from a JSONL file, skipping terminal positions."""
    import chess

    seen: set[str] = set()
    count = 0
    line_idx = 0

    with open(source_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            line_idx += 1
            # Stride-based sharding for multiprocessing
            if (line_idx - 1) % stride != offset:
                continue
            if limit and count >= limit:
                break
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            fen = rec.get("fen", "")
            if not fen or fen in seen:
                continue
            seen.add(fen)
            try:
                board = chess.Board(fen)
                if board.is_game_over():
                    continue
            except ValueError:
                continue
            yield fen
            count += 1


def _worker(
    source_path: str,
    out_train_path: str,
    out_eval_path: str,
    limit: int,
    eval_ratio: float,
    worker_id: int,
    num_workers: int,
    seed: int,
    progress_counter,
) -> None:
    random.seed(seed + worker_id)
    log.info("Worker %d starting (stride=%d, offset=%d)", worker_id, num_workers, worker_id)

    written = 0
    failed = 0
    last_update = 0

    with open(out_train_path, "w") as f_train, open(out_eval_path, "w") as f_eval:
        for fen in _iter_fens(source_path, limit, offset=worker_id, stride=num_workers):
            result = _compute_labels(fen)
            if result is None:
                failed += 1
                continue
            labels, eval_score = result

            record = {"fen": fen, "sf15_terms": labels, "eval_score": eval_score}
            line = json.dumps(record, ensure_ascii=False) + "\n"
            if random.random() < eval_ratio:
                f_eval.write(line)
            else:
                f_train.write(line)
            written += 1

            if progress_counter is not None and written - last_update >= 500:
                with progress_counter.get_lock():
                    progress_counter.value += written - last_update
                last_update = written

    if progress_counter is not None:
        with progress_counter.get_lock():
            progress_counter.value += written - last_update

    log.info("Worker %d done: written=%d failed=%d", worker_id, written, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="data/processed/encoder_pretrain_1b.jsonl",
        help="Source JSONL with 'fen' field",
    )
    parser.add_argument(
        "--output",
        default="data/processed/encoder_pretrain_sf15.jsonl",
    )
    parser.add_argument(
        "--eval-output",
        default="data/processed/encoder_pretrain_sf15_eval.jsonl",
    )
    parser.add_argument("--limit", type=int, default=5_000_000, help="Max FENs to process")
    parser.add_argument("--eval-ratio", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    limit_per_worker = args.limit // args.workers

    if args.workers == 1:
        progress_counter = None
        _worker(
            args.source,
            f"{args.output}.0",
            f"{args.eval_output}.0",
            limit_per_worker,
            args.eval_ratio,
            0,
            1,
            args.seed,
            progress_counter,
        )
    else:
        progress_counter = mp.Value("i", 0)
        processes = []
        for i in range(args.workers):
            p = mp.Process(
                target=_worker,
                args=(
                    args.source,
                    f"{args.output}.{i}",
                    f"{args.eval_output}.{i}",
                    limit_per_worker,
                    args.eval_ratio,
                    i,
                    args.workers,
                    args.seed,
                    progress_counter,
                ),
            )
            p.start()
            processes.append(p)

        try:
            from tqdm import tqdm

            pbar = tqdm(total=args.limit, desc="Annotating with SF15")
        except ImportError:
            pbar = None

        last_val = 0
        while any(p.is_alive() for p in processes):
            if pbar is not None:
                cur = progress_counter.value
                pbar.update(cur - last_val)
                last_val = cur
            time.sleep(1.0)

        if pbar is not None:
            pbar.update(progress_counter.value - last_val)
            pbar.close()

        for p in processes:
            p.join()

    # Concatenate worker outputs
    import shutil

    log.info("Concatenating worker outputs...")
    with open(args.output, "wb") as out:
        for i in range(args.workers):
            part = f"{args.output}.{i}"
            if os.path.exists(part):
                with open(part, "rb") as inp:
                    shutil.copyfileobj(inp, out)
                os.remove(part)

    with open(args.eval_output, "wb") as out:
        for i in range(args.workers):
            part = f"{args.eval_output}.{i}"
            if os.path.exists(part):
                with open(part, "rb") as inp:
                    shutil.copyfileobj(inp, out)
                os.remove(part)

    # Count and report
    n_train = sum(1 for _ in open(args.output))
    n_eval = sum(1 for _ in open(args.eval_output))
    log.info("Done. Train: %d → %s", n_train, args.output)
    log.info("Done. Eval:  %d → %s", n_eval, args.eval_output)
    log.info("Terms (%d): %s", N_TERMS, SF15_TERMS)


if __name__ == "__main__":
    main()
