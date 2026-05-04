#!/usr/bin/env python3
"""Benchmark Qwen3.5-4B-Base vs Qwen3.5-4B (instruct) on 100 mate-in-1 puzzles.

Both models receive identical chat-format prompts (base has control tokens
trained in per its model card). Instruct uses enable_thinking=False.

Puzzles loaded from data/processed/puzzle_grpo/eval.jsonl (Lichess database).

Usage:
    python scripts/benchmark_mate1.py
    python scripts/benchmark_mate1.py --n 100 --seed 42
    python scripts/benchmark_mate1.py --puzzles-file data/processed/puzzle_grpo/eval.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PUZZLES_FILE = "data/processed/puzzle_grpo/eval.jsonl"


def load_puzzles(path: str, n: int, seed: int) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "mateIn1" not in rec.get("themes", []):
                continue
            try:
                board = chess.Board(rec["fen"])
                move = chess.Move.from_uci(rec["solution_uci"])
                san = board.san(move)
            except Exception:
                continue
            records.append({
                "fen": rec["fen"],
                "uci": rec["solution_uci"],
                "san": san,
                "color": rec.get("color", "White"),
                "rating": rec.get("rating", 0),
            })

    rng = random.Random(seed)
    rng.shuffle(records)
    selected = records[:n]
    print(f"Loaded {len(selected)} mate-in-1 puzzles from {path} "
          f"(rating range: {min(p['rating'] for p in selected)}–{max(p['rating'] for p in selected)})",
          flush=True)
    return selected


def board_ascii(board: chess.Board) -> str:
    lines = ["  a b c d e f g h"]
    for rank in range(7, -1, -1):
        row = f"{rank + 1} "
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            row += (piece.symbol() if piece else ".") + " "
        lines.append(row.rstrip())
    lines.append(f"  {'White' if board.turn == chess.WHITE else 'Black'} to move")
    return "\n".join(lines)


def make_prompt(tokenizer, fen: str, color: str) -> str:
    board = chess.Board(fen)
    content = (
        f"{board_ascii(board)}\n\n"
        f"FEN: {fen}\n\n"
        f"It's {color}'s turn. There is a checkmate in one move. "
        f"What is the move? Reply with only the move in standard algebraic notation (e.g. Qxg7#)."
    )
    messages = [{"role": "user", "content": content}]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def extract_move(text: str, board: chess.Board) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try SAN-like tokens first
    for cand in re.findall(r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O-O|O-O", text):
        try:
            move = board.parse_san(cand.rstrip("+#"))
            if move in board.legal_moves:
                return cand
        except Exception:
            pass
    # Try UCI
    for cand in re.findall(r"\b[a-h][1-8][a-h][1-8]\b", text):
        try:
            move = chess.Move.from_uci(cand)
            if move in board.legal_moves:
                return cand
        except Exception:
            pass
    return None


def run_benchmark(model_name: str, puzzles: list[dict]) -> list[dict]:
    n = len(puzzles)
    print(f"\n{'='*60}", flush=True)
    print(f"Model : {model_name}", flush=True)
    print(f"Puzzles: {n}  (thinking=False)", flush=True)
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for i, puzzle in enumerate(puzzles):
        board = chess.Board(puzzle["fen"])
        prompt = make_prompt(tokenizer, puzzle["fen"], puzzle["color"])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        predicted = extract_move(response, board)
        correct = False
        if predicted:
            try:
                is_uci = bool(re.match(r"^[a-h][1-8][a-h][1-8][qrbn]?$", predicted))
                pred_move = chess.Move.from_uci(predicted) if is_uci else board.parse_san(predicted.rstrip("+#"))
                correct = pred_move == chess.Move.from_uci(puzzle["uci"])
            except Exception:
                pass

        mark = "✓" if correct else "✗"
        print(f"  {i+1:3d}. [{mark}] expected={puzzle['san']:8s}  predicted={repr(predicted)}", flush=True)
        print(f"       response: {repr(response)}", flush=True)
        results.append({"correct": correct, "expected": puzzle["san"], "predicted": predicted, "response": response})

    score = sum(r["correct"] for r in results)
    print(f"\n  Score: {score}/{n}  ({100*score/n:.1f}%)", flush=True)

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzles-file", default=DEFAULT_PUZZLES_FILE)
    parser.add_argument("--n", type=int, default=100, help="Number of puzzles")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    puzzles = load_puzzles(args.puzzles_file, args.n, args.seed)
    if not puzzles:
        print("No puzzles loaded — check --puzzles-file path", file=sys.stderr)
        sys.exit(1)

    configs = [
        "Qwen/Qwen3.5-4B-Base",
        "Qwen/Qwen3.5-4B",   # instruct
    ]

    all_results = {}
    for model_name in configs:
        all_results[model_name] = run_benchmark(model_name, puzzles)

    n = len(puzzles)
    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Model':<40}  Score", flush=True)
    for name, res in all_results.items():
        score = sum(r["correct"] for r in res)
        print(f"  {name:<40}  {score}/{n}  ({100*score/n:.1f}%)", flush=True)

    print(f"\n  {'#':<5}  {'rating':<7}  {'expected':<10}", end="", flush=True)
    for name in all_results:
        short = name.split("/")[-1][:18]
        print(f"  {short:<18}", end="", flush=True)
    print()
    for i, puzzle in enumerate(puzzles):
        print(f"  {i+1:<5}  {puzzle['rating']:<7}  {puzzle['san']:<10}", end="", flush=True)
        for res in all_results.values():
            mark = "✓" if res[i]["correct"] else "✗"
            print(f"  {mark:<18}", end="", flush=True)
        print()


if __name__ == "__main__":
    main()
