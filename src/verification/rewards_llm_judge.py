"""GRPO reward function: LLM judge (GCC-Eval) for chess coaching completions.

Single reward: the judge server scores each completion on 6 metrics and returns
a combined score. No rule-based rewards — quality signal comes entirely from the
judge model (Qwen3.5-35B-A3B-GPTQ-Int4 on port 8400).

Usage:
    from verification.rewards_llm_judge import combined_reward, JUDGE_SERVER_URL
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

import chess
import httpx

log = logging.getLogger(__name__)

JUDGE_SERVER_URL = os.environ.get("JUDGE_SERVER_URL", "http://localhost:8400")
_JUDGE_TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "300.0"))

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_FEN_RE = re.compile(r"FEN:\s*(\S+(?:\s+\S+){5})")
_MOVE_PLAYED_RE = re.compile(
    r"Move(?:\s+played)?:\s*(?:<\|[^|>]*\|>)?([A-Za-z][A-Za-z0-9+#=\-]*|O-O(?:-O)?)"
)
_SENTINEL_RE = re.compile(r"<\|[^|>]*\|>")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion_str(completion: list[dict[str, Any]] | str) -> str:
    if isinstance(completion, str):
        return completion
    for msg in reversed(completion):
        content = msg.get("content")
        if content:
            return content
    return ""


def _prompt_str(prompt: list[dict[str, Any]] | str) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n".join(m.get("content") or "" for m in prompt if m.get("content"))


def extract_comment(text: str) -> str:
    """Extract coaching comment from a completion.

    Tries to find text after </think>. If the model didn't use a think block,
    falls back to the full text stripped of tool syntax. The judge receives this
    alongside the full trajectory, so it's fine if this is approximate.
    """
    idx = text.find("</think>")
    if idx != -1:
        comment = text[idx + len("</think>") :].strip()
    else:
        # No think block — treat entire output as candidate comment
        comment = text.strip()

    # Strip tool call/response blocks
    comment = _TOOL_CALL_BLOCK_RE.sub("", comment)
    comment = _TOOL_RESPONSE_RE.sub("", comment)
    # Strip XML-style tags but keep their content
    comment = re.sub(r"<[^>]+>", " ", comment)
    return comment.strip()


def extract_fen_from_prompt(prompt: list[dict[str, Any]] | str) -> str:
    m = _FEN_RE.search(_prompt_str(prompt))
    return m.group(1).strip() if m else ""


def extract_move_san_from_prompt(prompt: list[dict[str, Any]] | str) -> str:
    m = _MOVE_PLAYED_RE.search(_prompt_str(prompt))
    if not m:
        return ""
    return _SENTINEL_RE.sub("", m.group(1)).strip()


def extract_engine_eval_from_trajectory(trajectory: str) -> dict[str, Any]:
    """Extract engine eval data from tool responses embedded in the trajectory."""
    import json as _json

    result: dict[str, Any] = {"score_cp": 0, "best_move": "", "cp_loss": 0, "sf15_terms": None}
    for resp_text in [m.group(1).strip() for m in _TOOL_RESPONSE_RE.finditer(trajectory)]:
        try:
            data = _json.loads(resp_text)
        except Exception:
            continue
        if "score_cp" in data and result["score_cp"] == 0:
            result["score_cp"] = int(data.get("score_cp") or 0)
            result["best_move"] = str(data.get("best_move") or "")
            result["cp_loss"] = int(data.get("cp_loss") or 0)
        if "notable" in data and data["notable"] and result["sf15_terms"] is None:
            result["sf15_terms"] = data["notable"]
    return result


# ---------------------------------------------------------------------------
# Judge HTTP call
# ---------------------------------------------------------------------------


async def _call_judge(
    client: httpx.AsyncClient,
    fen: str,
    move_san: str,
    engine_eval: dict[str, Any],
    trajectory: str,
    comment: str,
) -> float | None:
    """POST to judge server; returns combined score mapped to [-1, +1], or None on failure."""
    try:
        resp = await client.post(
            f"{JUDGE_SERVER_URL}/evaluate",
            json={
                "fen": fen,
                "move_san": move_san,
                "engine_eval": engine_eval,
                "trajectory": trajectory,
                "comment": comment,
            },
            timeout=_JUDGE_TIMEOUT,
        )
        resp.raise_for_status()
        combined: float = float(resp.json()["combined"])
        return combined * 2.0 - 1.0  # map [0, 1] → [-1, +1]
    except Exception as exc:
        log.error(
            "Judge call FAILED — fen=%r move=%r error=%s: %s",
            fen,
            move_san,
            type(exc).__name__,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def combined_reward(
    prompts: list[list[dict[str, Any]] | str],
    completions: list[list[dict[str, Any]] | str],
    **kwargs: Any,
) -> list[float | None]:
    """LLM judge reward for GRPO.

    Fires all judge calls concurrently (asyncio.gather). Returns None for
    any sample where the judge fails — TRL skips those from the gradient update.
    Returns 0.0 for samples with no coaching comment (no point judging).
    """
    fens = kwargs.get("fen", [""] * len(prompts))
    move_sans = kwargs.get("move_san", [""] * len(prompts))

    async def _evaluate_one(prompt, completion, fen, move_san) -> float | None:
        trajectory = _completion_str(completion)
        comment = extract_comment(trajectory)
        # Still judge even with no comment — judge penalises missing coaching text
        if not fen:
            fen = extract_fen_from_prompt(prompt)
        if not move_san:
            move_san = extract_move_san_from_prompt(prompt)
        engine_eval = extract_engine_eval_from_trajectory(trajectory)
        async with httpx.AsyncClient() as client:
            return await _call_judge(client, fen, move_san, engine_eval, trajectory, comment)

    async def _batch() -> list[float | None]:
        return list(
            await asyncio.gather(
                *[
                    _evaluate_one(p, c, f, m)
                    for p, c, f, m in zip(prompts, completions, fens, move_sans)
                ]
            )
        )

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, _batch()).result()
            else:
                return loop.run_until_complete(_batch())
        except RuntimeError:
            return asyncio.run(_batch())
    except Exception as exc:
        log.error("reward batch FAILED entirely: %s: %s", type(exc).__name__, exc)
        return [None] * len(prompts)
