#!/usr/bin/env python3
"""View puzzle GRPO training completions from /tmp/puzzle-grpo.log.

Usage:
    python scripts/view_grpo_log.py              # last 10 steps
    python scripts/view_grpo_log.py -n 20        # last 20 steps
    python scripts/view_grpo_log.py --step 42    # specific step
    python scripts/view_grpo_log.py --follow     # tail -f mode
    python scripts/view_grpo_log.py --full       # read full completions from JSONL
    python scripts/view_grpo_log.py --step 0 --full --all  # all completions at step
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict

LOG_FILE = "/tmp/puzzle-grpo.log"
COMPLETIONS_JSONL = "/tmp/puzzle-grpo-completions.jsonl"

_ROLLOUT_RE = re.compile(
    r"\[step=(\d+)\] rollout=[\d.]+s \| (\d+) completions \(\d+ failed\) \| "
    r"correct=(\d+) \([\d.]+%\) wrong=(\d+) no_fmt=(\d+) \| "
    r"avg_reward=([-\d.]+) avg_tok=([\d.]+)"
)
_BEST_RE = re.compile(
    r"\[step=(\d+)\] BEST \| solution=(\S+) predicted=(\S+) reward=([-\d.]+) tok=(\d+)"
)


def parse_log(lines: list[str]) -> dict[int, dict]:
    """Parse log lines into per-step records."""
    steps = {}
    i = 0
    while i < len(lines):
        line = lines[i]

        m = _ROLLOUT_RE.search(line)
        if m:
            step = int(m.group(1))
            if step not in steps:
                steps[step] = {}
            steps[step].update({
                "step": step,
                "completions": int(m.group(2)),
                "correct": int(m.group(3)),
                "wrong": int(m.group(4)),
                "no_fmt": int(m.group(5)),
                "avg_reward": float(m.group(6)),
                "avg_tok": float(m.group(7)),
            })
            i += 1
            continue

        m = _BEST_RE.search(line)
        if m:
            step = int(m.group(1))
            if step not in steps:
                steps[step] = {}
            steps[step].update({
                "step": step,
                "solution": m.group(2),
                "predicted": m.group(3),
                "best_reward": float(m.group(4)),
                "best_tok": int(m.group(5)),
            })
            # Collect continuation lines (the snippet)
            snippet_lines = []
            i += 1
            while i < len(lines):
                l = lines[i]
                # Stop at next log entry (timestamp pattern)
                if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", l):
                    break
                snippet_lines.append(l.rstrip())
                i += 1
            steps[step]["snippet"] = "\n".join(snippet_lines).strip()
            continue

        i += 1

    return steps


def load_completions_jsonl() -> dict[int, list[dict]]:
    """Load all completions from JSONL, grouped by step."""
    by_step: dict[int, list[dict]] = defaultdict(list)
    try:
        with open(COMPLETIONS_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    by_step[rec["step"]].append(rec)
                except (json.JSONDecodeError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return dict(by_step)


def render_step(rec: dict, verbose: bool = True, full_completions: list[dict] | None = None, show_all: bool = False) -> str:
    step = rec.get("step", "?")
    avg_r = rec.get("avg_reward", float("nan"))
    correct = rec.get("correct", "?")
    wrong = rec.get("wrong", "?")
    no_fmt = rec.get("no_fmt", "?")
    avg_tok = rec.get("avg_tok", 0)
    solution = rec.get("solution", "?")
    predicted = rec.get("predicted", "?")
    best_r = rec.get("best_reward", float("nan"))
    best_tok = rec.get("best_tok", "?")
    snippet = rec.get("snippet", "")

    header = (
        f"{'='*60}\n"
        f"step={step}  avg_reward={avg_r:.3f}  avg_tok={avg_tok:.0f}\n"
        f"correct={correct}  wrong={wrong}  no_fmt={no_fmt}\n"
        f"BEST: solution={solution} predicted={predicted} reward={best_r:.2f} tok={best_tok}"
    )

    if full_completions is not None:
        if not full_completions:
            return header + "\n(no JSONL data for this step — run with a later step)\n"
        # Sort by reward descending, show best (or all)
        sorted_completions = sorted(full_completions, key=lambda x: x.get("reward", -999), reverse=True)
        to_show = sorted_completions if show_all else sorted_completions[:1]
        parts = [header]
        for idx, c in enumerate(to_show):
            parts.append(f"\n{'─'*40}")
            if show_all:
                parts.append(f"[{idx+1}/{len(to_show)}] reward={c.get('reward', '?'):.2f} tok={c.get('tokens', '?')} solution={c.get('solution', '?')} predicted={c.get('predicted', '?')}")
            parts.append(c.get("completion", "(empty)"))
        return "\n".join(parts) + "\n"

    if verbose and snippet:
        return header + f"\n{'-'*40}\n{snippet}\n"
    return header + "\n"


def read_log() -> list[str]:
    try:
        with open(LOG_FILE) as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"Log file not found: {LOG_FILE}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="View GRPO completion logs")
    parser.add_argument("-n", "--last", type=int, default=10, help="Show last N steps (default 10)")
    parser.add_argument("--step", type=int, help="Show a specific step")
    parser.add_argument("--follow", action="store_true", help="Follow log (like tail -f)")
    parser.add_argument("--no-snippet", action="store_true", help="Hide completion snippets")
    parser.add_argument("--full", action="store_true", help=f"Read full completions from {COMPLETIONS_JSONL}")
    parser.add_argument("--all", action="store_true", help="Show all completions per step (with --full)")
    args = parser.parse_args()

    completions_by_step = load_completions_jsonl() if args.full else None

    if args.follow:
        seen_steps = set()
        print(f"Following {LOG_FILE} ... (Ctrl-C to stop)\n")
        while True:
            lines = read_log()
            steps = parse_log(lines)
            if args.full:
                completions_by_step = load_completions_jsonl()
            for step in sorted(steps):
                if step not in seen_steps:
                    fc = completions_by_step.get(step) if completions_by_step is not None else None
                    print(render_step(steps[step], verbose=not args.no_snippet, full_completions=fc, show_all=args.all))
                    seen_steps.add(step)
            time.sleep(2)
        return

    lines = read_log()
    steps = parse_log(lines)

    if args.step is not None:
        if args.step in steps:
            fc = completions_by_step.get(args.step) if completions_by_step is not None else None
            print(render_step(steps[args.step], verbose=not args.no_snippet, full_completions=fc, show_all=args.all))
        else:
            print(f"Step {args.step} not found. Available: {sorted(steps.keys())}")
        return

    recent = sorted(steps.keys())[-args.last:]
    for step in recent:
        fc = completions_by_step.get(step) if completions_by_step is not None else None
        print(render_step(steps[step], verbose=not args.no_snippet, full_completions=fc, show_all=args.all))


if __name__ == "__main__":
    main()
