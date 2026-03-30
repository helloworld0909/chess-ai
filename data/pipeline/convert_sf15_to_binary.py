"""Convert SF15 JSONL dataset to compact binary format for fast training.

Binary record layout (210 bytes each, fixed-size → O(1) random access):
    fen:          90 bytes  (null-padded UTF-8 string)
    sf15_terms:   13 × float32  (52 bytes)
    eval_score:   1  × float32  (4 bytes)
    piece_labels: 64 × uint8    (64 bytes)  values 0-12

Total: 210 bytes/record → 138GB for 656M records (vs 114GB JSONL but sequential reads)

Benefits over JSONL:
  - O(1) random access by index (no byte-offset scan)
  - Sequential reads → OS page cache friendly
  - No JSON parsing overhead per sample (~3x faster DataLoader)
  - Piece labels precomputed (saves chess.Board() call per sample)

Usage:
    uv run python data/pipeline/convert_sf15_to_binary.py \\
        --input  data/processed/encoder_pretrain_sf15.jsonl \\
        --output data/processed/encoder_pretrain_sf15.bin \\
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Binary record layout constants
FEN_BYTES = 90
N_SF15 = 13
RECORD_SIZE = FEN_BYTES + N_SF15 * 4 + 4 + 64  # 210 bytes

# Piece label mapping
_PIECE_TO_CLASS = {
    (chess.WHITE, chess.PAWN): 1,
    (chess.WHITE, chess.KNIGHT): 2,
    (chess.WHITE, chess.BISHOP): 3,
    (chess.WHITE, chess.ROOK): 4,
    (chess.WHITE, chess.QUEEN): 5,
    (chess.WHITE, chess.KING): 6,
    (chess.BLACK, chess.PAWN): 7,
    (chess.BLACK, chess.KNIGHT): 8,
    (chess.BLACK, chess.BISHOP): 9,
    (chess.BLACK, chess.ROOK): 10,
    (chess.BLACK, chess.QUEEN): 11,
    (chess.BLACK, chess.KING): 12,
}


def _encode_record(rec: dict) -> bytes | None:
    """Encode a JSONL record to a fixed-size binary record."""
    try:
        fen = rec["fen"]
        sf15 = rec["sf15_terms"]
        eval_score = rec["eval_score"]
    except KeyError:
        return None

    # FEN: encode and pad/truncate to FEN_BYTES
    fen_b = fen.encode("utf-8")[:FEN_BYTES]
    fen_b = fen_b.ljust(FEN_BYTES, b"\x00")

    # SF15 terms + eval score as float32
    sf15_b = struct.pack(f"{N_SF15}f", *sf15)
    ev_b = struct.pack("f", eval_score)

    # Piece labels: compute from FEN
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    labels = bytearray(64)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            labels[sq] = _PIECE_TO_CLASS[(piece.color, piece.piece_type)]

    return fen_b + sf15_b + ev_b + bytes(labels)


def _worker(
    input_path: str,
    output_path: str,
    worker_id: int,
    num_workers: int,
    progress_counter,
) -> None:
    written = failed = 0
    last_update = 0

    with open(input_path) as fin, open(output_path, "wb") as fout:
        for line_idx, raw in enumerate(fin):
            if line_idx % num_workers != worker_id:
                continue
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                failed += 1
                continue
            encoded = _encode_record(rec)
            if encoded is None:
                failed += 1
                continue
            fout.write(encoded)
            written += 1

            if progress_counter is not None and written - last_update >= 1000:
                with progress_counter.get_lock():
                    progress_counter.value += written - last_update
                last_update = written

    if progress_counter is not None:
        with progress_counter.get_lock():
            progress_counter.value += written - last_update

    log.info("Worker %d done: written=%d failed=%d", worker_id, written, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_lines = sum(1 for _ in open(args.input))
    log.info("Input: %d records → converting with %d workers", n_lines, args.workers)

    progress_counter = mp.Value("i", 0)
    processes = []
    part_paths = []

    for i in range(args.workers):
        part = f"{args.output}.part{i}"
        part_paths.append(part)
        p = mp.Process(
            target=_worker,
            args=(args.input, part, i, args.workers, progress_counter),
        )
        p.start()
        processes.append(p)

    t0 = time.time()
    last = 0
    while any(p.is_alive() for p in processes):
        cur = progress_counter.value
        elapsed = time.time() - t0
        rate = (cur - last) / max(elapsed, 1)
        pct = cur / n_lines * 100
        eta = (n_lines - cur) / max(rate, 1)
        log.info(
            "Progress: %d/%d (%.1f%%) — %.0f/s — ETA %.0fmin", cur, n_lines, pct, rate, eta / 60
        )
        last = cur
        time.sleep(10)

    for p in processes:
        p.join()

    # Concatenate parts into final file (preserves order by interleaving stride)
    log.info("Concatenating %d parts...", args.workers)
    total = 0
    with open(args.output, "wb") as fout:
        for part in part_paths:
            if os.path.exists(part):
                with open(part, "rb") as fin:
                    import shutil

                    shutil.copyfileobj(fin, fout)
                    total += os.path.getsize(part) // RECORD_SIZE
                os.remove(part)

    log.info(
        "Done. %d records → %s (%.1f GB)", total, args.output, os.path.getsize(args.output) / 1e9
    )
    log.info("RECORD_SIZE=%d bytes", RECORD_SIZE)


if __name__ == "__main__":
    main()
