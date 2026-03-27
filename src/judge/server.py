"""GCC-Eval judge server for chess coaching completions.

Implements a FastAPI server that evaluates chess coaching completions on 6 metrics
(Correctness, Think Quality, Completeness, Relevance, Clarity, Fluency) using an
LLM judge backed by a vLLM OpenAI-compatible endpoint.

The judge verifies concrete chess claims with Stockfish tools before scoring,
following the GCC-Eval methodology (arxiv 2410.20811).

Runs on port 8400.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import chess
import chess.engine
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8300/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
SF15_PATH = os.environ.get("SF15_PATH", os.path.expanduser("~/.local/bin/stockfish-15"))

_MAX_TOOL_ROUNDS = 8
_HTTP_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert chess coach evaluating a student AI's chess commentary.

You MUST use your tools to verify concrete claims before scoring. Do not guess — call a tool.

## What requires tool verification

- Move legality or quality claim → call is_move_legal or stockfish_eval
- Strategic claim ("improves activity", "weakens king safety", "gains space", "creates threats",
  "strengthens pawns", "improves bishop", "activates rook") → call sf15_term_diff
  sf15_term_diff returns per-term diffs (moving side's perspective, positive = improved):
  Terms: Mobility, King safety, Threats, Material, Pawns, Bishops, Rooks, Queens, Space, Passed, Initiative
- Tactical claim (fork, pin, mate threat) → call stockfish_eval to check the position
- Piece placement claim → call get_legal_moves to verify what is and isn't possible

DO NOT award correctness credit for any claim you have not verified.
Be skeptical. If the comment says "this improves piece activity", call sf15_term_diff and check Mobility.

## Scoring

CORRECTNESS (1–3): Correctness is close to a binary decision.
   - 3: all verified claims are factually correct
   - 2: minor inaccuracies or a few unverified assertions
   - 1: any major claim is demonstrably false, or significant facts stated without verification

THINK_QUALITY (1–5): Quality of the reasoning process.
   - 5: used tools, acted on results, reasoning is position-specific and leads logically to the comment
   - 4: good tool use and reasoning, minor gaps
   - 3: some tool use but results not fully integrated, or partly generic reasoning
   - 2: minimal tool use, mostly generic thinking
   - 1: no tool use, generic platitudes, or comment contradicts the think block

COMPLETENESS (1–5): Coverage of critical position factors.
   - 5: covers the most significant SF15 term changes and explains WHY the move is good/bad vs alternatives
   - 3: addresses the move but misses important factors
   - 1: superficial — no explanation of impact or alternatives

RELEVANCE (1–5): Focus on this specific position.
   - 5: every sentence is specific to this position and move
   - 3: mostly relevant with minor filler
   - 1: generic advice that applies to any position

CLARITY (1–5): Specificity and concreteness.
   - 5: references specific moves, squares, or pieces — nothing vague
   - 3: some concrete references mixed with vague statements
   - 1: entirely vague ("improves your position", "good move")

FLUENCY (1–5): Language quality and coaching tone.
   - 5: coaching tone, second person ("You play Nd5..."), coherent transitions
   - 3: readable but awkward phrasing or inconsistent person
   - 1: hard to read or first-person throughout

Think step by step, verify your claims with tools, then output scores as JSON on the final line:
{"correctness": X, "think_quality": X, "completeness": X, "relevance": X, "clarity": X, "fluency": X, "reasoning": "one sentence"}
"""

_JUDGE_USER_TEMPLATE = """\
Chess position: {fen}
{board_ascii}
Move played: {move_san}
Engine evaluation: actual move {score_cp}cp, best move {best_move} ({cp_loss}cp loss)
{sf15_section}
=== TRAINEE TRAJECTORY ===
{trajectory}

=== COACHING COMMENT (extracted) ===
{comment}

Your evaluation steps:
1. List every concrete claim in the comment (moves, strategic factors, tactical features)
2. For each move claim: call is_move_legal or stockfish_eval to verify
3. For each strategic claim (activity, safety, space, threats, pawns, etc.): call sf15_term_diff to verify
4. Compare verified facts against the comment — note any errors
5. Score all 6 metrics as JSON
"""

# ---------------------------------------------------------------------------
# Tool definitions (sent to the judge LLM)
# ---------------------------------------------------------------------------

_JUDGE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "stockfish_eval",
            "description": (
                "Evaluate a chess position with Stockfish. "
                "Returns the centipawn score (from white's perspective), best move, "
                "best move in SAN notation, and mate_in (null if no forced mate)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "FEN string of the position to evaluate.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Search depth (default 18).",
                        "default": 18,
                    },
                },
                "required": ["fen"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "is_move_legal",
            "description": "Check whether a move (in SAN notation) is legal in the given position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "FEN string of the position.",
                    },
                    "move_san": {
                        "type": "string",
                        "description": "Move in Standard Algebraic Notation (e.g. 'Nf3', 'e4', 'O-O').",
                    },
                },
                "required": ["fen", "move_san"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_legal_moves",
            "description": "Return all legal moves in a position as a list of SAN strings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "FEN string of the position.",
                    },
                },
                "required": ["fen"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sf15_term_diff",
            "description": (
                "Run Stockfish 15 classical evaluation before and after a move, "
                "returning per-term diffs from the moving side's perspective. "
                "Terms: Mobility, King safety, Threats, Material, Pawns, Bishops, "
                "Rooks, Queens, Space, Passed, Initiative. "
                "Positive diff = term improved for the player who made the move. "
                "Use this to verify strategic claims: 'improves piece activity' → check Mobility, "
                "'weakens king safety' → check King safety, 'creates threats' → check Threats, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "FEN of the position BEFORE the move.",
                    },
                    "move_san": {
                        "type": "string",
                        "description": "The move to analyse in SAN notation (e.g. 'Nd5', 'e4', 'O-O').",
                    },
                },
                "required": ["fen", "move_san"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Stockfish engine pool
# ---------------------------------------------------------------------------

_ENGINE_POOL: list[chess.engine.SimpleEngine] = []
_ENGINE_LOCK = asyncio.Lock()
_POOL_SIZE = 4
_SF_DEPTH = 18


async def _get_engine() -> chess.engine.SimpleEngine:
    """Borrow an engine from the pool, creating a new one if the pool is empty."""
    async with _ENGINE_LOCK:
        if _ENGINE_POOL:
            return _ENGINE_POOL.pop()
    # Create outside the lock to avoid blocking other coroutines
    transport, engine = await chess.engine.popen_uci(STOCKFISH_PATH)  # type: ignore[attr-defined]
    return engine  # type: ignore[return-value]


async def _return_engine(engine: chess.engine.SimpleEngine) -> None:
    """Return an engine to the pool, quitting it if the pool is full."""
    async with _ENGINE_LOCK:
        if len(_ENGINE_POOL) < _POOL_SIZE:
            _ENGINE_POOL.append(engine)
            return
    try:
        await engine.quit()  # type: ignore[attr-defined]
    except Exception:
        pass


async def _prewarm_engines(n: int = 2) -> None:
    """Pre-warm the engine pool at startup."""
    for _ in range(n):
        try:
            transport, engine = await chess.engine.popen_uci(STOCKFISH_PATH)  # type: ignore[attr-defined]
            async with _ENGINE_LOCK:
                if len(_ENGINE_POOL) < _POOL_SIZE:
                    _ENGINE_POOL.append(engine)  # type: ignore[arg-type]
                else:
                    await engine.quit()  # type: ignore[attr-defined]
                    break
        except Exception as exc:
            log.warning("Engine pre-warm failed: %s", exc)
            break


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def _tool_stockfish_eval(fen: str, depth: int = 18) -> dict[str, Any]:
    """Execute the stockfish_eval tool server-side.

    Args:
        fen: FEN string of the position.
        depth: Stockfish search depth.

    Returns:
        Dict with score_cp, best_move, best_move_san, mate_in.
    """
    engine = await _get_engine()
    try:
        board = chess.Board(fen)
        info = await engine.analyse(board, chess.engine.Limit(depth=depth))  # type: ignore[attr-defined]
        score_obj = info["score"].white()
        if score_obj.is_mate():
            score_cp = 30000 if score_obj.mate() > 0 else -30000  # type: ignore[operator]
            mate_in: int | None = score_obj.mate()  # type: ignore[assignment]
        else:
            score_cp = score_obj.score() or 0
            mate_in = None

        best_move_uci: str | None = None
        best_move_san: str | None = None
        if "pv" in info and info["pv"]:
            best_mv = info["pv"][0]
            best_move_uci = best_mv.uci()
            try:
                best_move_san = board.san(best_mv)
            except Exception:
                best_move_san = best_move_uci

        return {
            "score_cp": score_cp,
            "best_move": best_move_uci,
            "best_move_san": best_move_san,
            "mate_in": mate_in,
        }
    except Exception as exc:
        log.warning("stockfish_eval failed for FEN %r: %s", fen, exc)
        return {
            "error": str(exc),
            "score_cp": None,
            "best_move": None,
            "best_move_san": None,
            "mate_in": None,
        }
    finally:
        await _return_engine(engine)


async def _tool_is_move_legal(fen: str, move_san: str) -> dict[str, Any]:
    """Execute the is_move_legal tool server-side.

    Args:
        fen: FEN string of the position.
        move_san: Move in SAN notation.

    Returns:
        Dict with legal (bool) and reason (str).
    """
    try:
        board = chess.Board(fen)
        board.parse_san(move_san)
        return {"legal": True, "reason": f"{move_san} is a legal move in this position."}
    except chess.IllegalMoveError:
        return {"legal": False, "reason": f"{move_san} is not a legal move in this position."}
    except chess.AmbiguousMoveError:
        return {
            "legal": False,
            "reason": f"{move_san} is ambiguous — multiple pieces can make this move.",
        }
    except Exception as exc:
        return {"legal": False, "reason": f"Could not parse {move_san!r}: {exc}"}


async def _tool_get_legal_moves(fen: str) -> dict[str, Any]:
    """Execute the get_legal_moves tool server-side.

    Args:
        fen: FEN string of the position.

    Returns:
        Dict with moves list (SAN strings).
    """
    try:
        board = chess.Board(fen)
        moves = [board.san(mv) for mv in board.legal_moves]
        return {"moves": sorted(moves)}
    except Exception as exc:
        return {"moves": [], "error": str(exc)}


async def _tool_sf15_term_diff(fen: str, move_san: str) -> dict[str, Any]:
    """Execute the sf15_term_diff tool server-side.

    Runs Stockfish 15 classical evaluation before and after a move, returning
    per-term diffs from the moving side's perspective.

    Args:
        fen: FEN string of the position before the move.
        move_san: The move in SAN notation.

    Returns:
        Dict with per-term diffs and a summary of notable changes.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _sf15_term_diff_sync, fen, move_san)
        return result
    except Exception as exc:
        log.warning("sf15_term_diff failed for FEN %r move %r: %s", fen, move_san, exc)
        return {"error": str(exc)}


def _sf15_term_diff_sync(fen: str, move_san: str) -> dict[str, Any]:
    """Blocking SF15 term diff — runs in executor thread."""
    import os as _os
    import sys

    # Dynamically import sf15_eval from data pipeline
    sf15_eval_path = _os.path.join(_os.path.dirname(__file__), "..", "..", "data", "pipeline")
    if sf15_eval_path not in sys.path:
        sys.path.insert(0, sf15_eval_path)

    # Temporarily set SF15_PATH so sf15_eval picks it up
    old_sf15 = _os.environ.get("SF15_PATH")
    _os.environ["SF15_PATH"] = SF15_PATH
    try:
        from sf15_eval import get_sf15_eval  # type: ignore[import]
    except ImportError:
        return {"error": "sf15_eval module not available — set SF15_PATH correctly"}
    finally:
        if old_sf15 is None:
            _os.environ.pop("SF15_PATH", None)
        else:
            _os.environ["SF15_PATH"] = old_sf15

    try:
        board = chess.Board(fen)
        eval_before = get_sf15_eval(board.fen())
        mv = board.parse_san(move_san)
        white_moved = board.turn == chess.WHITE
        board.push(mv)
        eval_after = get_sf15_eval(board.fen())
    except Exception as exc:
        return {"error": f"Board/move error: {exc}"}

    if not eval_before or not eval_after:
        return {"error": "SF15 eval returned empty result"}

    diffs: dict[str, float] = {}
    for term in eval_before:
        if term not in eval_after:
            continue
        if white_moved:
            delta = (eval_after[term]["White"] - eval_after[term]["Black"]) - (
                eval_before[term]["White"] - eval_before[term]["Black"]
            )
        else:
            delta = (eval_after[term]["Black"] - eval_after[term]["White"]) - (
                eval_before[term]["Black"] - eval_before[term]["White"]
            )
        diffs[term] = round(delta, 3)

    # Sort by absolute value descending
    sorted_diffs = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)
    notable = [(t, d) for t, d in sorted_diffs if abs(d) >= 0.05]

    summary_parts = []
    for term, delta in notable[:5]:
        sign = "+" if delta >= 0 else ""
        summary_parts.append(f"{term}: {sign}{delta:.3f}")

    return {
        "move": move_san,
        "moving_side": "white" if white_moved else "black",
        "diffs": dict(sorted_diffs),
        "notable": {t: d for t, d in notable},
        "summary": ", ".join(summary_parts) if summary_parts else "no notable term changes",
        "interpretation": (
            "Positive = term improved for the moving side. "
            "E.g. Mobility +0.32 means the moving side gained piece activity."
        ),
    }


async def _execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Dispatch a tool call to the appropriate server-side implementation.

    Args:
        tool_name: Name of the tool to call.
        tool_args: Arguments for the tool.

    Returns:
        JSON-encoded result string.
    """
    if tool_name == "stockfish_eval":
        result = await _tool_stockfish_eval(
            fen=tool_args["fen"],
            depth=int(tool_args.get("depth", _SF_DEPTH)),
        )
    elif tool_name == "is_move_legal":
        result = await _tool_is_move_legal(
            fen=tool_args["fen"],
            move_san=tool_args["move_san"],
        )
    elif tool_name == "get_legal_moves":
        result = await _tool_get_legal_moves(fen=tool_args["fen"])
    elif tool_name == "sf15_term_diff":
        result = await _tool_sf15_term_diff(
            fen=tool_args["fen"],
            move_san=tool_args["move_san"],
        )
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result)


# ---------------------------------------------------------------------------
# LLM multi-turn tool-use loop
# ---------------------------------------------------------------------------


async def _run_judge(
    messages: list[dict[str, Any]],
    client: httpx.AsyncClient,
) -> tuple[str, int, int]:
    """Run the judge LLM with tool-use loop.

    Executes the judge LLM with up to _MAX_TOOL_ROUNDS rounds of tool calls,
    then returns the final assistant message text.

    Args:
        messages: Initial conversation messages.
        client: httpx.AsyncClient for LLM calls.

    Returns:
        Tuple of (final_text, verified_claims_count, failed_claims_count).
    """
    tool_rounds = 0
    verified_claims = 0
    failed_claims = 0

    while tool_rounds <= _MAX_TOOL_ROUNDS:
        payload: dict[str, Any] = {
            "model": JUDGE_MODEL,
            "messages": messages,
            "tools": _JUDGE_TOOLS,
            "tool_choice": "auto" if tool_rounds < _MAX_TOOL_ROUNDS else "none",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "max_tokens": 4096,
        }

        try:
            resp = await client.post(
                f"{JUDGE_BASE_URL}/chat/completions",
                json=payload,
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Judge LLM request failed: {exc}") from exc

        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        # Append assistant message to history (strip reasoning_content before appending)
        messages.append({k: v for k, v in message.items() if k != "reasoning_content"})

        # If no tool calls, we're done
        if finish_reason != "tool_calls" or not message.get("tool_calls"):
            # SGLang with --reasoning-parser qwen3 puts <think> content in reasoning_content
            # and post-think content in content. If content is empty, the JSON scores are
            # likely inside the think block — fall back to reasoning_content.
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            final_text = content if content.strip() else reasoning
            return final_text, verified_claims, failed_claims

        # Execute each tool call and append results
        for tc in message["tool_calls"]:
            tc_id = tc["id"]
            func = tc["function"]
            tool_name = func["name"]
            try:
                tool_args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                tool_args = {}

            tool_result_str = await _execute_tool(tool_name, tool_args)
            tool_result = json.loads(tool_result_str)

            # Count verified / failed claims for is_move_legal calls
            if tool_name == "is_move_legal":
                if tool_result.get("legal"):
                    verified_claims += 1
                else:
                    failed_claims += 1

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result_str,
                }
            )

        tool_rounds += 1

    # Forced final turn with no tools allowed — should not normally reach here
    final_text = messages[-1].get("content") or ""
    return final_text, verified_claims, failed_claims


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

_SCORE_JSON_RE = re.compile(
    r'\{[^{}]*"correctness"\s*:\s*[\d.]+[^{}]*\}',
    re.DOTALL,
)

_METRIC_KEYS = ("correctness", "think_quality", "completeness", "relevance", "clarity", "fluency")
_METRIC_WEIGHTS = {
    "correctness": 0.30,
    "think_quality": 0.25,
    "completeness": 0.15,
    "relevance": 0.10,
    "clarity": 0.10,
    "fluency": 0.10,
}


def _parse_scores(judge_text: str) -> dict[str, float]:
    """Parse the JSON scores block from the judge's response.

    Args:
        judge_text: Full text output from the judge LLM.

    Returns:
        Dict with float scores for all 6 metrics. Defaults to 0.5 if not found.
    """
    # No defaults — None means the judge failed to score this metric
    parsed: dict[str, float | None] = {k: None for k in _METRIC_KEYS}

    # Find the last JSON block that looks like scores
    matches = list(_SCORE_JSON_RE.finditer(judge_text))
    if not matches:
        all_braces = list(re.finditer(r"\{[^{}]+\}", judge_text, re.DOTALL))
        if all_braces:
            matches = all_braces

    for m in reversed(matches):
        try:
            obj = json.loads(m.group(0))
            if "correctness" in obj:
                for key in _METRIC_KEYS:
                    val = obj.get(key)
                    if val is not None:
                        try:
                            parsed[key] = float(val)
                        except (ValueError, TypeError):
                            pass
                return parsed
        except json.JSONDecodeError:
            continue

    log.warning("Could not parse scores from judge response. Last 500 chars: %r", judge_text[-500:])
    return parsed


# Per-metric scale: correctness is 1-3, all others 1-5
_METRIC_SCALE: dict[str, float] = {
    "correctness": 3.0,
    "think_quality": 5.0,
    "completeness": 5.0,
    "relevance": 5.0,
    "clarity": 5.0,
    "fluency": 5.0,
}


def _compute_combined(scores: dict[str, float | None]) -> float:
    """Compute the weighted combined score, normalised to [0, 1].

    Each metric is normalised by its own scale ((score-1)/(max-1)).
    Missing scores (None) are excluded from the weighted average — the
    weights of present metrics are renormalised so the result stays in [0,1].

    Args:
        scores: Dict of metric scores (raw scale per metric), may contain None.

    Returns:
        Weighted combined score in [0, 1], or 0.0 if no scores present.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for key in _METRIC_KEYS:
        val = scores.get(key)
        if val is None:
            continue
        scale = _METRIC_SCALE[key]
        normalised = max(0.0, min(1.0, (val - 1.0) / (scale - 1.0)))
        w = _METRIC_WEIGHTS[key]
        weighted_sum += w * normalised
        total_weight += w
    if total_weight == 0.0:
        return 0.0
    # Renormalise so missing metrics don't deflate the score
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------

_cache: dict[str, "EvaluationResponse"] = {}
_cache_lock = asyncio.Lock()
_CACHE_MAXSIZE = 1000


def _cache_key(fen: str, move_san: str, comment: str) -> str:
    h = hashlib.sha256(f"{fen}|{move_san}|{comment}".encode()).hexdigest()[:16]
    return h


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EngineEval(BaseModel):
    """Engine evaluation data for a position."""

    score_cp: int = Field(default=0, description="Centipawn score from white's perspective.")
    best_move: str = Field(default="", description="Best move in UCI notation.")
    cp_loss: int = Field(default=0, description="Centipawn loss vs best move.")
    sf15_terms: dict[str, float] | None = Field(
        default=None,
        description=(
            "SF15 classical eval term diffs for the played move (from moving side's perspective). "
            "Keys: Mobility, King safety, Threats, Material, Pawns, Bishops, Rooks, Queens, "
            "Space, Passed, Initiative. Positive = improved for moving side."
        ),
    )


class EvaluationRequest(BaseModel):
    """Request body for the /evaluate endpoint."""

    fen: str = Field(..., description="FEN string of the position before the move.")
    move_san: str = Field(..., description="Move played in SAN notation.")
    engine_eval: EngineEval = Field(default_factory=EngineEval)
    trajectory: str = Field(
        ..., description="Full trainee trajectory including tool calls and think block."
    )
    comment: str = Field(..., description="Extracted coaching comment text only.")


class EvaluationResponse(BaseModel):
    """Response body from the /evaluate endpoint."""

    correctness: float | None  # 1-3 scale; None if judge failed to score
    think_quality: float | None  # 1-5 scale
    completeness: float | None  # 1-5 scale
    relevance: float | None  # 1-5 scale
    clarity: float | None  # 1-5 scale
    fluency: float | None  # 1-5 scale
    combined: float  # normalised to [0, 1]; renormalised over present metrics
    verified_claims: int
    failed_claims: int
    judge_reasoning: str


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan: pre-warm Stockfish engine pool on startup."""
    log.info("Pre-warming Stockfish engine pool (path=%s)", STOCKFISH_PATH)
    try:
        await _prewarm_engines(n=2)
        log.info("Engine pool ready with %d engine(s)", len(_ENGINE_POOL))
    except Exception as exc:
        log.warning("Could not pre-warm engines: %s", exc)
    yield
    # Shutdown: close all pooled engines
    async with _ENGINE_LOCK:
        for eng in _ENGINE_POOL:
            try:
                await eng.quit()  # type: ignore[attr-defined]
            except Exception:
                pass
        _ENGINE_POOL.clear()
    log.info("Engine pool shut down")


app = FastAPI(
    title="Chess Coaching Judge Server",
    description="GCC-Eval judge server for chess coaching completions.",
    version="1.0.0",
    lifespan=_lifespan,
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Return server health and configured judge model."""
    return {"status": "ok", "model": JUDGE_MODEL}


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluate a chess coaching completion using the GCC-Eval methodology.

    The judge LLM verifies concrete chess claims with Stockfish tools before
    scoring on 6 metrics: correctness, think_quality, completeness, relevance,
    clarity, and fluency.

    Args:
        request: Evaluation request with FEN, move, engine eval, trajectory, and comment.

    Returns:
        EvaluationResponse with per-metric scores, combined score, and judge reasoning.
    """
    cache_key = _cache_key(request.fen, request.move_san, request.comment)

    async with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

    # Build ASCII board representation (before the move)
    try:
        _board = chess.Board(request.fen)
        _rows = str(_board).split("\n")
        _lines = ["  a b c d e f g h"]
        for _i, _row in enumerate(_rows):
            _lines.append(f"{8 - _i} {_row}")
        _lines.append(f"  ({'White' if _board.turn == chess.WHITE else 'Black'} to move)")
        board_ascii = "\n".join(_lines)
    except Exception:
        board_ascii = ""

    # Build optional SF15 section from pre-extracted terms (saves a tool call if available)
    sf15_section = ""
    if request.engine_eval.sf15_terms:
        notable = sorted(
            request.engine_eval.sf15_terms.items(), key=lambda x: abs(x[1]), reverse=True
        )
        parts = [f"  {t}: {'+' if d >= 0 else ''}{d:.3f}" for t, d in notable[:6]]
        sf15_section = (
            "SF15 term diffs for this move (pre-computed, verify with sf15_term_diff if unsure):\n"
            + "\n".join(parts)
            + "\n"
        )

    user_prompt = _JUDGE_USER_TEMPLATE.format(
        fen=request.fen,
        board_ascii=board_ascii,
        move_san=request.move_san,
        score_cp=request.engine_eval.score_cp,
        best_move=request.engine_eval.best_move or "N/A",
        cp_loss=request.engine_eval.cp_loss,
        sf15_section=sf15_section,
        trajectory=request.trajectory,
        comment=request.comment,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    async with httpx.AsyncClient() as client:
        judge_text, verified_claims, failed_claims = await _run_judge(messages, client)

    scores = _parse_scores(judge_text)
    combined = _compute_combined(scores)

    # Extract reasoning from the parsed JSON if present
    reasoning = judge_text
    try:
        matches = list(_SCORE_JSON_RE.finditer(judge_text))
        if matches:
            obj = json.loads(matches[-1].group(0))
            reasoning = obj.get("reasoning", judge_text)
    except Exception:
        pass

    response = EvaluationResponse(
        correctness=scores["correctness"],
        think_quality=scores["think_quality"],
        completeness=scores["completeness"],
        relevance=scores["relevance"],
        clarity=scores["clarity"],
        fluency=scores["fluency"],
        combined=combined,
        verified_claims=verified_claims,
        failed_claims=failed_claims,
        judge_reasoning=reasoning,
    )

    async with _cache_lock:
        if len(_cache) >= _CACHE_MAXSIZE:
            # Evict oldest entry (first inserted key)
            oldest_key = next(iter(_cache))
            del _cache[oldest_key]
        _cache[cache_key] = response

    return response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("src.judge.server:app", host="0.0.0.0", port=8400, reload=False)
