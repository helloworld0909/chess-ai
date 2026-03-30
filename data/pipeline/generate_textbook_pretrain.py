"""Generate continued-pretraining JSONL from processed textbook .txt files.

Each output record is a chunk of text with [Position: FEN] markers replaced by
64 <|vision_pad|> sentinel tokens. Labels equal input_ids (causal LM on all
text tokens; sentinels are masked to -100 by the model's forward pass).

Section-based chunking (PGN-sourced files):
  Files with ## / ### headers are split per section (one game/chapter per record).
  Long sections are sliding-windowed at max_tokens with 25% overlap.

Window-based chunking (Gutenberg full-text files):
  Files without headers are chunked with a sliding window.

Output JSONL fields per record:
  {"text": "...", "fens": ["fen1", "fen2", ...], "source": "filename"}

Usage:
    uv run python data/pipeline/generate_textbook_pretrain.py \\
        --input-dir data/processed/textbooks \\
        --output data/processed/textbook_pretrain.jsonl \\
        --max-tokens 2048 \\
        --eval-split 0.02
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_POSITION_RE = re.compile(r"\[Position(?:[^]]*?):\s*([^\]]+)\]")
_SECTION_RE = re.compile(r"^(?=## |### )", re.MULTILINE)


def _replace_positions_with_sentinels(text: str, sentinel: str, n: int) -> tuple[str, list[str]]:
    """Replace every [Position: FEN] with n sentinel tokens; return (text, fens)."""
    fens: list[str] = []
    block = sentinel * n

    def _replace(m: re.Match) -> str:
        fens.append(m.group(1).strip())
        return block

    replaced = _POSITION_RE.sub(_replace, text)
    return replaced, fens


def _tokenize_length(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _chunk_by_tokens(
    text: str,
    fens: list[str],
    sentinel: str,
    n_sentinels: int,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer,
    source: str,
) -> list[dict]:
    """Sliding-window chunk already-sentinel-replaced text into ≤max_tokens records."""
    # Tokenize full text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [{"text": text, "fens": fens, "source": source}]

    records = []
    step = max_tokens - overlap_tokens
    sentinel_id = tokenizer.convert_tokens_to_ids(sentinel)

    i = 0
    while i < len(token_ids):
        chunk_ids = token_ids[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
        # Count sentinel tokens in this chunk to get FEN slice
        n_sent_in_chunk = chunk_ids.count(sentinel_id)
        # Find which FENs correspond to sentinels in this slice
        # We count sentinels in the prefix up to position i
        prefix_ids = token_ids[:i]
        n_sent_before = prefix_ids.count(sentinel_id)
        # Each board uses n_sentinels tokens; compute board index range
        board_start = n_sent_before // n_sentinels
        board_end = (n_sent_before + n_sent_in_chunk) // n_sentinels
        chunk_fens = fens[board_start:board_end]
        if chunk_text.strip():
            records.append({"text": chunk_text, "fens": chunk_fens, "source": source})
        i += step
        if i + max_tokens > len(token_ids) and i < len(token_ids):
            # Final partial chunk — include if it has content
            chunk_ids = token_ids[i:]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
            n_sent_in_chunk = chunk_ids.count(sentinel_id)
            board_start = (token_ids[:i].count(sentinel_id)) // n_sentinels
            chunk_fens = fens[board_start : board_start + n_sent_in_chunk // n_sentinels]
            if chunk_text.strip():
                records.append({"text": chunk_text, "fens": chunk_fens, "source": source})
            break
    return records


def process_file(
    path: Path,
    sentinel: str,
    n_sentinels: int,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer,
) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source = path.name

    # Check if file has section headers (PGN-sourced)
    has_sections = bool(re.search(r"^## |^### ", text, re.MULTILINE))

    if has_sections:
        raw_sections = _SECTION_RE.split(text)
    else:
        raw_sections = [text]

    records: list[dict] = []
    for section in raw_sections:
        section = section.strip()
        if not section or not _POSITION_RE.search(section):
            continue
        replaced, fens = _replace_positions_with_sentinels(section, sentinel, n_sentinels)
        if not fens:
            continue
        chunk_records = _chunk_by_tokens(
            replaced, fens, sentinel, n_sentinels, max_tokens, overlap_tokens, tokenizer, source
        )
        records.extend(chunk_records)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed/textbooks")
    parser.add_argument("--output", default="data/processed/textbook_pretrain.jsonl")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per record (includes 64 sentinels per position)",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=256,
        help="Overlap between consecutive windows for long sections",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.02,
        help="Fraction of records held out for eval",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from src.encoder import BOARD_TOKEN, BOARD_TOKENS_PER_POSITION

    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)

    sentinel = BOARD_TOKEN
    n_sentinels = BOARD_TOKENS_PER_POSITION

    input_dir = Path(args.input_dir)
    txt_files = sorted(input_dir.glob("*.txt"))
    log.info("Processing %d .txt files from %s", len(txt_files), input_dir)

    all_records: list[dict] = []
    for fpath in txt_files:
        recs = process_file(
            fpath, sentinel, n_sentinels, args.max_tokens, args.overlap_tokens, tokenizer
        )
        log.info("  %s → %d records", fpath.name, len(recs))
        all_records.extend(recs)

    log.info("Total records before dedup: %d", len(all_records))

    # Filter out records with zero FENs (pure text chunks with no board positions)
    # Keep them — they provide chess language context
    # But log the breakdown
    with_pos = sum(1 for r in all_records if r["fens"])
    log.info("  With positions: %d  |  Text-only: %d", with_pos, len(all_records) - with_pos)

    random.seed(args.seed)
    random.shuffle(all_records)

    # Eval split
    n_eval = max(1, int(len(all_records) * args.eval_split))
    eval_records = all_records[:n_eval]
    train_records = all_records[n_eval:]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path = output_path.with_name(output_path.stem + "_eval.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for rec in eval_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(
        "Written %d train + %d eval records → %s / %s",
        len(train_records),
        len(eval_records),
        output_path,
        eval_path,
    )


if __name__ == "__main__":
    main()
