"""ChessQA-Benchmark evaluation via SGLang server (fast inference).

Sends requests to a running ChessQwen3ForCausalLM SGLang server using the
OpenAI-compatible API with board tensors via extra_body.

Usage:
    # Start server first:
    ./sglang/serve.sh /tmp/chess-merged

    # Run eval:
    python scripts/eval_chessqa_sglang.py --output /tmp/chessqa_sglang.json

    # With thinking enabled:
    python scripts/eval_chessqa_sglang.py --thinking --output /tmp/chessqa_thinking.json

    # Specific subset, limit examples:
    python scripts/eval_chessqa_sglang.py --subset short_tactics --max-examples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder import BOARD_TOKEN

_logger = logging.getLogger(__name__)

_BOARD_BLOCK = BOARD_TOKEN  # one placeholder; processor expands to 65 slots

_SYSTEM_PROMPT = (
    "You are a chess assistant. The board position is encoded as a sequence of vision tokens. "
    "Use them to identify pieces and answer questions about the position."
    "Answer chess questions directly and concisely with minimum explanation."
)

_SUBSETS = {"motifs", "position_judgement", "semantic", "short_tactics", "structural"}


# ---------------------------------------------------------------------------
# Prompt building (reused from eval_chessqa.py)
# ---------------------------------------------------------------------------


_PIECE_NAMES = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}


def _get_context(fen: str) -> str:
    """Match official CSSLab eval: full piece names, UCI legal moves."""
    try:
        board = chess.Board(fen)
    except Exception:
        return ""
    pieces: dict[str, list[str]] = {}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            color = "White" if piece.color == chess.WHITE else "Black"
            key = f"{color} {_PIECE_NAMES[piece.piece_type]}"
            pieces.setdefault(key, []).append(chess.square_name(sq))
    arrangement = ", ".join(f"{k}: {v}" for k, v in pieces.items())
    legal_moves = ", ".join(sorted(m.uci() for m in board.legal_moves))
    return f"Piece arrangement: {arrangement}\nLegal moves: {legal_moves}\n\n"


def _strip_fen_preamble(question: str, fen: str, add_context: bool) -> str:
    """Match original eval_chessqa.py: strip FEN preamble, replace CONTEXT_PLACEHOLDER."""
    for preamble in (
        f"You are given a chess position in FEN: {fen}.\n",
        f"You are analyzing a chess position in FEN: {fen}.\n",
    ):
        if question.startswith(preamble):
            question = question[len(preamble):]
            break
    context = _get_context(fen) if add_context else ""
    question = question.replace("CONTEXT_PLACEHOLDER", context)
    return question.lstrip()


def _build_prompt(question: str, fen: str, format_examples: list[str], strip_fen: bool, add_context: bool = False) -> str:
    """Match official CSSLab eval prompt building."""
    question = _strip_fen_preamble(question, fen, add_context=add_context)
    if not strip_fen:
        question = f"You are given a chess position in FEN: {fen}.\n{question}"
    if format_examples and "FORMAT_EXAMPLE_PLACEHOLDER" in question:
        question = question.replace("FORMAT_EXAMPLE_PLACEHOLDER", format_examples[0])
    if format_examples:
        examples_str = "\n".join(f"  {e}" for e in format_examples[:3])
        return f"{question}\n\nAnswer using this format (examples):\n{examples_str}"
    return question


# ---------------------------------------------------------------------------
# Answer matching (identical to eval_chessqa.py)
# ---------------------------------------------------------------------------


def _extract_final_answer(pred: str) -> str:
    """Match official CSSLab eval: last FINAL ANSWER: match, strip markdown bold."""
    # Last FINAL ANSWER: occurrence
    matches = list(re.finditer(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", pred, re.IGNORECASE | re.DOTALL))
    if matches:
        answer = matches[-1].group(1).strip()
        answer = re.sub(r"^FINAL ANSWER:\s*", "", answer, flags=re.IGNORECASE).strip()
        answer = re.sub(r"^\*+|\*+$", "", answer).strip()
        return answer
    # Fallback: \boxed{...}
    boxed = list(re.finditer(r"[Tt]he\s+final\s+answer\s+is\s+\$?\\boxed\{([^}]+)\}\$?", pred))
    if boxed:
        return boxed[-1].group(1).strip()
    return ""


def match_exact(pred: str, correct: str, answer_type: str = "single") -> bool:
    """Match official CSSLab eval: simple lower/strip for single, set for multi."""
    extracted = _extract_final_answer(pred).lower().strip()
    if answer_type == "multi":
        pred_set = {item.strip() for item in extracted.split(",") if item.strip()}
        correct_set = {item.strip() for item in correct.lower().split(",") if item.strip()}
        return pred_set == correct_set
    return extracted == correct.lower().strip()


# ---------------------------------------------------------------------------
# SGLang inference — raw HTTP so image_data passes through untouched
# ---------------------------------------------------------------------------


def _post_json(base_url: str, path: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def infer_one(base_url: str, model_name: str, ex: dict, thinking: bool, strip_fen: bool, max_new_tokens: int, temperature: float = 0.0, board_tokens: bool = True, add_context: bool = False) -> tuple[str, str, int]:
    """Send one example to SGLang server via /generate, return (full_output, answer).

    Uses /generate (not /v1/chat/completions) so image_data passes directly to
    the multimodal processor without being lost in OpenAI message parsing.
    The chat template is applied manually here.
    """
    fen = ex["fen"].split("|")[0].strip()
    try:
        chess.Board(fen)
    except Exception:
        return ("", "", 0)

    board_prefix = _BOARD_BLOCK + "\n\n" if board_tokens else ""
    user_content = board_prefix + _build_prompt(
        ex["question"], fen, ex.get("format_examples") or [], strip_fen, add_context
    )

    # Build prompt text using Qwen3 chat template manually
    # enable_thinking=False injects <think>\n\n</think>\n\n per chat_template.jinja
    think_prefix = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    # When not thinking, append concise-answer instruction (Qwen3 standard practice)
    system = _SYSTEM_PROMPT
    if not thinking:
        system += " Respond concisely and provide only the final answer without lengthy reasoning."
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{think_prefix}"
    )

    payload: dict = {
        "text": prompt,
        **({"image_data": [{"format": "board_tensor", "fen": fen}]} if board_tokens else {}),
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": False,
        },
    }

    resp = _post_json(base_url, "/generate", payload)
    full = resp.get("text", "")
    completion_tokens = resp.get("meta_info", {}).get("completion_tokens", 0)
    if thinking:
        idx = full.rfind("</think>")
        answer = full[idx + len("</think>"):].strip() if idx != -1 else full
    else:
        answer = re.sub(r"</?think>", "", full).strip()
    return (full, answer, completion_tokens)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ChessQA via SGLang server")
    parser.add_argument("--base-url", default="http://localhost:8300", help="SGLang server URL")
    parser.add_argument("--model", default="/tmp/chess-merged", help="Model name for API")
    parser.add_argument("--subset", default=None, choices=sorted(_SUBSETS) + [None])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output", default=None, help="Save results JSON here")
    parser.add_argument("--thinking", action="store_true", default=False,
                        help="Enable thinking mode (chat_template_kwargs enable_thinking=True)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Max tokens (default: 8192 thinking, 512 no-thinking)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent requests (default: 32 no-thinking, 4 thinking)")
    parser.add_argument("--no-strip-fen", dest="strip_fen", action="store_false")
    parser.set_defaults(strip_fen=True)
    parser.add_argument("--no-board-tokens", dest="board_tokens", action="store_false")
    parser.set_defaults(board_tokens=True)
    parser.add_argument("--add-context", action="store_true", default=False,
                        help="Inject piece arrangement + UCI legal moves (CONTEXT_PLACEHOLDER)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system-prompt", default=None, help="Override system prompt")
    args = parser.parse_args()

    if args.system_prompt:
        global _SYSTEM_PROMPT
        _SYSTEM_PROMPT = args.system_prompt

    if args.max_new_tokens is None:
        args.max_new_tokens = 32768 if args.thinking else 512
    if args.workers is None:
        # With 32k max_new_tokens and 313k KV budget per replica (DP=2 → 626k total),
        # safe concurrency ≈ 626k / 32k ≈ 19. Use 16 to leave headroom.
        args.workers = 16 if args.thinking else 128

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Health check
    try:
        urllib.request.urlopen(f"{args.base_url}/health", timeout=5)
    except Exception as e:
        _logger.error("Server not reachable at %s: %s", args.base_url, e)
        _logger.error("Start with: ./sglang/serve.sh /tmp/chess-merged")
        sys.exit(1)

    # Load dataset
    _logger.info("Loading wieeii/ChessQA-Benchmark...")
    from datasets import load_dataset
    from datasets.dataset_dict import DatasetDict

    subsets_to_load = [args.subset] if args.subset else sorted(_SUBSETS)
    per_subset: dict[str, list[dict]] = {}
    for config_name in subsets_to_load:
        _logger.info("  Loading config: %s", config_name)
        ds = load_dataset("wieeii/ChessQA-Benchmark", config_name)
        rows = []
        if isinstance(ds, DatasetDict):
            for split_data in ds.values():
                for row in split_data:
                    rows.append(dict(row))
        else:
            for row in ds:
                rows.append(dict(row))
        # Sort by task_id for deterministic ordering regardless of HF cache order
        rows.sort(key=lambda r: r.get("task_id", ""))
        per_subset[config_name] = rows

    if args.max_examples:
        # Sample evenly across subsets using fixed stride for coverage within each subset
        per_n = max(1, args.max_examples // len(subsets_to_load))
        examples = []
        for config_name in subsets_to_load:
            rows = per_subset[config_name]
            stride = max(1, len(rows) // per_n)
            sampled = rows[::stride][:per_n]
            examples.extend(sampled)
        examples = examples[: args.max_examples]
    else:
        examples = [row for rows in per_subset.values() for row in rows]

    _logger.info(
        "Evaluating %d examples with %d workers (thinking=%s, max_new_tokens=%d)",
        len(examples), args.workers, args.thinking, args.max_new_tokens,
    )

    results: list[dict] = []
    errors = 0

    def _process(ex: dict) -> dict:
        fen = ex["input"]
        item = {
            "fen": fen,
            "question": ex["question"],
            "format_examples": ex.get("format_examples") or [],
            "task_type": ex.get("task_type", ""),
        }
        try:
            full, answer, completion_tokens = infer_one(args.base_url, args.model, item, args.thinking, args.strip_fen, args.max_new_tokens, args.temperature, args.board_tokens, args.add_context)
        except Exception as e:
            _logger.warning("Request failed for FEN %s: %s", fen, e)
            full, answer, completion_tokens = "", "", 0
        return {
            "task_id": ex.get("task_id", ""),
            "task_group": ex.get("task_group", "unknown"),
            "task_type": ex.get("task_type", ""),
            "fen": fen,
            "question": ex["question"],
            "correct_answer": ex["correct_answer"],
            "answer_type": ex.get("answer_type", "single"),
            "predicted": full,
            "predicted_answer": _extract_final_answer(answer),
            "exact": match_exact(answer, ex["correct_answer"], ex.get("answer_type", "single")),
            "completion_tokens": completion_tokens,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process, ex): ex for ex in examples}
        for fut in tqdm(as_completed(futures), total=len(examples), desc="ChessQA eval"):
            results.append(fut.result())

    # Aggregate
    by_group: dict[str, list[bool]] = defaultdict(list)
    by_group_toks: dict[str, list[int]] = defaultdict(list)
    for r in results:
        by_group[r["task_group"]].append(r["exact"])
        by_group_toks[r["task_group"]].append(r.get("completion_tokens", 0))

    total = sum(r["exact"] for r in results)
    all_toks = [r.get("completion_tokens", 0) for r in results]
    avg_toks = sum(all_toks) / len(all_toks) if all_toks else 0
    truncated = sum(1 for t in all_toks if t >= args.max_new_tokens * 0.98)

    lines = [
        "",
        "=" * 50,
        "CHESSQA EVAL RESULTS",
        "=" * 50,
        f"Model:    {args.model}",
        f"Thinking: {args.thinking}",
        f"Max tok:  {args.max_new_tokens}",
        f"Examples: {len(results)}",
        "-" * 50,
        f"Overall:  {100*total/len(results):.1f}%  ({total}/{len(results)})",
        f"Avg tok:  {avg_toks:.0f}  truncated={truncated}/{len(results)}",
        "-" * 50,
    ]
    for group in sorted(by_group):
        hits = sum(by_group[group])
        n = len(by_group[group])
        gtoks = sum(by_group_toks[group]) / len(by_group_toks[group]) if by_group_toks[group] else 0
        gtrunc = sum(1 for t in by_group_toks[group] if t >= args.max_new_tokens * 0.98)
        lines.append(f"  {group:<28} {100*hits/n:.1f}%  ({hits}/{n})  avg_tok={gtoks:.0f}  trunc={gtrunc}")
    lines.append("=" * 50)

    print("\n".join(lines), flush=True)

    if args.output:
        out = {
            "overall_exact": total / len(results),
            "by_group": {g: sum(v) / len(v) for g, v in by_group.items()},
            "args": vars(args),
            "results": results,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
