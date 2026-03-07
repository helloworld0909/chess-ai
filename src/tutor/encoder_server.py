"""Encoder inference server for ChessLMWithEncoder.

Standalone FastAPI server that loads ChessLMWithEncoder and exposes:
  POST /api/encoder/analyze  — takes {fen, move_uci}, runs Stockfish for key
                               lines, builds the joint prompt, streams the
                               annotated lines + coaching comment.

Launch via: recipes-train/qwen3.5-4b-encoder-phase1-sft/serve.sh
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import chess
import chess.engine
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.tutor.encoder_inference import (
    build_board_tensors,
    generate_stream,
    inject_sentinels,
    load_encoder_model,
)
from src.tutor.prompts import JOINT_SYSTEM_PROMPT, board_ascii, format_joint_user_prompt, move_facts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_engine: chess.engine.SimpleEngine | None = None
_sf15_path: str | None = None  # path to SF15 binary (None = skip SF15 annotations)

CHECKPOINT_DIR = os.environ.get("ENCODER_CHECKPOINT", "checkpoints/qwen3.5-4b-encoder-phase1-sft")
CONFIG_PATH = os.environ.get(
    "ENCODER_CONFIG", "recipes-train/qwen3.5-4b-encoder-phase1-sft/config.yaml"
)
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
SF15_PATH = os.environ.get("SF15_PATH", "")  # empty = disabled
STOCKFISH_DEPTH = int(os.environ.get("STOCKFISH_DEPTH", "18"))
HOST = os.environ.get("ENCODER_HOST", "0.0.0.0")
PORT = int(os.environ.get("ENCODER_PORT", "8200"))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model, _tokenizer, _engine, _sf15_path

    _logger.info("Loading encoder model from %s ...", CHECKPOINT_DIR)
    _model, _tokenizer = load_encoder_model(CHECKPOINT_DIR, CONFIG_PATH)
    _logger.info("Model loaded.")

    _logger.info("Starting Stockfish at %s", STOCKFISH_PATH)
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    _logger.info("Stockfish ready.")

    if SF15_PATH and os.path.isfile(SF15_PATH):
        _sf15_path = SF15_PATH
        _logger.info("SF15 annotations enabled: %s", SF15_PATH)
    else:
        _sf15_path = None
        _logger.warning("SF15_PATH not set or not found — running without SF15 term annotations")

    yield

    if _engine:
        _engine.quit()


app = FastAPI(title="Chess Encoder Inference Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    fen: str
    move_uci: str
    depth: int = STOCKFISH_DEPTH
    multipv: int = 5
    temperature: float = 0.7
    max_new_tokens: int = 2048


# ---------------------------------------------------------------------------
# Stockfish helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SF15 annotation helpers (mirrors generate_phase2_data.py logic exactly)
# ---------------------------------------------------------------------------


def _sf15_terms(term_diffs: dict[str, float]) -> list[tuple[str, float]]:
    notable = [(abs(d), term, d) for term, d in term_diffs.items() if abs(d) >= 0.10]
    if not notable:
        return []
    notable.sort(reverse=True)
    return [(term, d) for _, term, d in notable[:3]]


def _fmt_terms(terms: list[tuple[str, float]]) -> str:
    parts = []
    for term, d in terms:
        sign = "+" if d >= 0 else "−"
        parts.append(f"{term.lower()} {sign}{abs(d):.2f}")
    return "; ".join(parts)


def _annotate_sans_with_sf15(
    sans: list[str],
    start_fen: str,
    white_to_move_at_start: bool,
) -> list[str]:
    """Walk a list of SANs and return annotated strings like 'Nd5 [mobility +0.32; threats −0.15]'.

    Falls back to plain SAN if SF15 is unavailable or a move fails.
    """
    if _sf15_path is None:
        return list(sans)

    from data.pipeline.sf15_eval import get_sf15_eval

    result: list[str] = []
    board = chess.Board(start_fen)
    try:
        eval_before = get_sf15_eval(start_fen)
    except Exception:
        return list(sans)

    for san in sans:
        try:
            move = board.parse_san(san)
            white_moved = board.turn == chess.WHITE
            board.push(move)
            fen_after = board.fen()
        except Exception:
            result.append(san)
            continue

        try:
            eval_after = get_sf15_eval(fen_after)
            diffs: dict[str, float] = {}
            for term in eval_before:
                if term not in eval_after:
                    continue
                adv_before = eval_before[term]["White"] - eval_before[term]["Black"]
                adv_after = eval_after[term]["White"] - eval_after[term]["Black"]
                delta = adv_after - adv_before
                if not white_moved:
                    delta = -delta
                diffs[term] = round(delta, 2)
            terms = _sf15_terms(diffs)
            eval_before = eval_after
        except Exception:
            terms = []

        if terms:
            result.append(f"{san} [{_fmt_terms(terms)}]")
        else:
            result.append(san)

    return result


# ---------------------------------------------------------------------------


def _line_params(spread_cp: int) -> tuple[int, int]:
    """Return (line_depth_half_moves, n_engine_lines) based on position spread.

    Depth is capped at 6 half-moves (the model cannot learn from deeper sequences).
    """
    if spread_cp >= 200:
        return 6, 5
    elif spread_cp >= 100:
        return 6, 4
    elif spread_cp >= 50:
        return 5, 3
    else:
        return 4, 2


def _get_analysis(
    fen: str, move_uci: str, depth: int
) -> tuple[list[str], list[list[str]], str, int | None, int | None, str, str, str, list[str]]:
    """Run Stockfish and return analysis for the position.

    Returns:
        played_line_sans: SAN list for the PLAYED LINE (student move + engine continuation)
        engine_line_sans: list of SAN lists for engine alternative lines
        eval_label: human label for position eval (White's perspective)
        root_cp: centipawns from White's perspective before student move (None if mate)
        student_cp: centipawns from White's perspective after student move (None if mate)
        best_move_san: SAN of engine's top choice from pre-move position
        played_line_str: annotated PLAYED LINE string for prompt (with SF15 brackets)
        played_line_str_plain: plain SAN PLAYED LINE string (for UI display)
        engine_line_strs: annotated engine line strings for prompt (with SF15 brackets)
    """
    assert _engine is not None
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    move_san = board.san(move)

    # Analyse pre-move position (multipv=6 to get spread)
    result = _engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=6,
    )

    root_cp: int | None = None
    best_move_san: str = ""
    all_lines: list[tuple[list[str], int | None]] = []  # (sans, cp_white)
    cp_values_stm: list[int] = []  # side-to-move perspective
    sign = 1 if board.turn == chess.WHITE else -1

    for i, info in enumerate(result):
        pv = info.get("pv", [])
        if not pv:
            continue

        score = info.get("score")
        cp_white: int | None = None
        if score is not None:
            s = score.white()
            if s.mate() is not None:
                if i == 0:
                    root_cp = None
            else:
                cp_val = s.score()
                if i == 0:
                    root_cp = cp_val
                    try:
                        best_move_san = board.san(pv[0])
                    except Exception:
                        pass
                cp_white = cp_val
                cp_values_stm.append(sign * cp_val)

        # Convert PV to SAN (no depth cap here; we'll truncate later)
        b = board.copy()
        sans: list[str] = []
        for mv in pv:
            try:
                sans.append(b.san(mv))
                b.push(mv)
            except Exception:
                break
        if sans:
            all_lines.append((sans, cp_white))

    # Compute spread and adaptive params
    spread_cp = (max(cp_values_stm) - min(cp_values_stm)) if len(cp_values_stm) >= 2 else 0
    line_depth, n_engine_lines = _line_params(spread_cp)

    # Eval label (White's perspective)
    if root_cp is None:
        eval_label = "forced mate"
    else:
        cp = root_cp
        if cp >= 200:
            eval_label = "winning for white"
        elif cp >= 60:
            eval_label = "good for white"
        elif cp >= -60:
            eval_label = "equal"
        elif cp >= -200:
            eval_label = "good for black"
        else:
            eval_label = "winning for black"

    # Build PLAYED LINE: student's move + engine continuation from board_after
    board_after = board.copy()
    board_after.push(move)

    played_sans: list[str] = [move_san]
    try:
        cont_result = _engine.analyse(
            board_after,
            chess.engine.Limit(depth=max(depth - 2, 6)),
            multipv=1,
        )
        if cont_result:
            pv_cont = cont_result[0].get("pv", [])
            b = board_after.copy()
            for mv in pv_cont:
                try:
                    played_sans.append(b.san(mv))
                    b.push(mv)
                except Exception:
                    break
    except Exception:
        pass

    played_sans_truncated = played_sans[:line_depth]

    # Get student move cp
    student_cp: int | None = None
    try:
        student_info = _engine.analyse(board_after, chess.engine.Limit(depth=max(depth - 2, 8)))
        s = student_info.get("score")
        if s is not None:
            w = s.white()
            student_cp = None if w.mate() is not None else w.score()
    except Exception:
        pass

    # Select engine alternative lines (skip played move)
    engine_line_sans: list[list[str]] = []
    for sans, cp_w in all_lines:
        if not sans:
            continue
        if sans[0] == move_san:
            continue
        engine_line_sans.append(sans[:line_depth])
        if len(engine_line_sans) >= n_engine_lines:
            break

    played_line_str_plain = " → ".join(played_sans_truncated)

    # Annotate with SF15 terms to match training data format
    white_to_move = board.turn == chess.WHITE
    # Played line starts from pre-move board (student's move is first)
    played_annotated = _annotate_sans_with_sf15(played_sans_truncated, fen, white_to_move)
    played_line_str = " → ".join(played_annotated)

    engine_line_strs = []
    for line_sans in engine_line_sans:
        ann = _annotate_sans_with_sf15(line_sans, fen, white_to_move)
        engine_line_strs.append(" → ".join(ann))

    return (
        played_sans_truncated,
        engine_line_sans,
        eval_label,
        root_cp,
        student_cp,
        best_move_san,
        played_line_str,
        played_line_str_plain,
        engine_line_strs,
    )


def _classify_move(
    student_san: str,
    best_san: str,
    student_cp: int | None,
    best_cp: int | None,
    turn: chess.Color,
) -> tuple[str, int]:
    """Classify the student's move vs the engine best. Returns (classification, cp_loss)."""
    if student_san == best_san:
        return "Best", 0

    if student_cp is None or best_cp is None:
        return "Good", 0

    # cp_loss from the side-to-move's perspective
    sign = 1 if turn == chess.WHITE else -1
    cp_loss = max(0, sign * (best_cp - student_cp))

    if cp_loss == 0:
        return "Best", 0
    elif cp_loss <= 10:
        return "Great", cp_loss
    elif cp_loss <= 30:
        return "Good", cp_loss
    elif cp_loss <= 100:
        return "Inaccuracy", cp_loss
    elif cp_loss <= 300:
        return "Mistake", cp_loss
    else:
        return "Blunder", cp_loss


# ---------------------------------------------------------------------------
# Stream endpoint
# ---------------------------------------------------------------------------


@app.post("/api/encoder/analyze")
async def analyze_encoder(req: AnalyzeRequest) -> StreamingResponse:
    """Analyze a move with the encoder model. Streams SSE events:

    event: meta   — JSON: {eval_label, key_lines (plain SAN strings)}
    event: think  — thinking token chunk
    event: token  — completion token chunk
    event: done   — empty, signals end
    event: error  — JSON with error message
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return StreamingResponse(
        _stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream(req: AnalyzeRequest) -> AsyncGenerator[str, None]:
    board = chess.Board(req.fen)

    # SAN for student move
    try:
        move = chess.Move.from_uci(req.move_uci)
        move_san = board.san(move)
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        return

    # Stockfish analysis (run in thread to not block event loop)
    try:
        (
            played_line_sans,
            engine_line_sans,
            eval_label,
            root_cp,
            student_cp,
            best_move_san,
            played_line_str,
            played_line_str_plain,
            engine_line_strs,
        ) = await asyncio.get_event_loop().run_in_executor(
            None, _get_analysis, req.fen, req.move_uci, req.depth
        )
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': f'Stockfish: {e}'})}\n\n"
        return

    # Classify the student's move
    try:
        # best_cp = root_cp (eval of position before the move from White's perspective)
        # For classification we need best line's cp vs student's resulting cp
        # _get_analysis already computed root_cp and student_cp
        classification, cp_loss = _classify_move(
            move_san, best_move_san, student_cp, root_cp, board.turn
        )
    except Exception:
        classification, cp_loss = "Good", 0

    # Emit meta (plain SAN strings for the UI — no SF15 brackets)
    all_line_strs = [played_line_str_plain] + [" → ".join(line) for line in engine_line_sans]
    is_best = move_san == best_move_san
    meta = {
        "eval_label": eval_label,
        "key_lines": all_line_strs,
        "move_san": move_san,
        "classification": classification,
        "cp_loss": cp_loss,
        "best_move": best_move_san,
        "is_best": is_best,
    }
    yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

    # Build board ascii
    board_before_str = board_ascii(board)
    facts = move_facts(board, move)
    board_after = board.copy()
    board_after.push(move)
    board_after_str = board_ascii(board_after)

    # Build user prompt (SF15-annotated, no sentinels yet)
    user_content = format_joint_user_prompt(
        board_ascii_str=board_before_str,
        fen=req.fen,
        move_san=move_san,
        eval_str=eval_label,
        facts=facts,
        board_after_str=board_after_str,
        fen_after=board_after.fen(),
        played_line=played_line_str,
        key_lines=engine_line_strs,
    )

    messages = [
        {"role": "system", "content": JOINT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Inject <|vision_pad|> sentinels
    messages_with_sentinels = inject_sentinels(messages, move_san)

    # Build board tensors — played line and engine lines all start from pre-move board
    board_tensors = build_board_tensors(req.fen, move_san, engine_line_sans, played_line_sans)

    # Stream generation in a thread (model.generate is synchronous)
    in_think = False
    think_buf = ""

    def _run_stream():
        return generate_stream(
            _model,
            _tokenizer,
            messages_with_sentinels,
            board_tensors,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )

    try:
        loop = asyncio.get_event_loop()
        gen = await loop.run_in_executor(None, _run_stream)

        # Can't iterate a sync generator directly in async — collect in thread
        # Instead, use queue-based approach
        import queue
        import threading

        q: queue.Queue[str | None] = queue.Queue()

        def _producer():
            try:
                for chunk in generate_stream(
                    _model,
                    _tokenizer,
                    messages_with_sentinels,
                    board_tensors,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                ):
                    q.put(chunk)
            finally:
                q.put(None)  # sentinel

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        in_think = False
        buf = ""

        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break

            buf += chunk
            # Route <think>...</think> to think events, rest to token
            while buf:
                if not in_think:
                    start = buf.find("<think>")
                    if start == -1:
                        # No think tag — check for partial tag at end
                        partial = _partial_tag(buf, "<think>")
                        safe = buf[: len(buf) - partial]
                        if safe:
                            yield f"event: token\ndata: {json.dumps(safe)}\n\n"
                        buf = buf[len(buf) - partial :] if partial else ""
                        break
                    else:
                        before = buf[:start]
                        if before:
                            yield f"event: token\ndata: {json.dumps(before)}\n\n"
                        in_think = True
                        buf = buf[start + len("<think>") :]
                else:
                    end = buf.find("</think>")
                    if end == -1:
                        partial = _partial_tag(buf, "</think>")
                        safe = buf[: len(buf) - partial]
                        if safe:
                            yield f"event: think\ndata: {json.dumps(safe)}\n\n"
                        buf = buf[len(buf) - partial :] if partial else ""
                        break
                    else:
                        think_chunk = buf[:end]
                        if think_chunk:
                            yield f"event: think\ndata: {json.dumps(think_chunk)}\n\n"
                        in_think = False
                        buf = buf[end + len("</think>") :]

        # Flush remaining
        if buf:
            event = "think" if in_think else "token"
            yield f"event: {event}\ndata: {json.dumps(buf)}\n\n"

        thread.join()

    except Exception as e:
        _logger.exception("Generation error")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        return

    yield "event: done\ndata: {}\n\n"


def _partial_tag(text: str, tag: str) -> int:
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


# ---------------------------------------------------------------------------
# Debug endpoint — returns exact prompt text and raw model output
# ---------------------------------------------------------------------------


@app.post("/api/encoder/debug")
async def debug_encoder(req: AnalyzeRequest) -> dict:
    """Return the exact tokenized prompt and raw model output for inspection."""
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    board = chess.Board(req.fen)
    try:
        move = chess.Move.from_uci(req.move_uci)
        move_san = board.san(move)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    (
        played_line_sans,
        engine_line_sans,
        eval_label,
        _,
        _,
        _,
        played_line_str,
        played_line_str_plain,
        engine_line_strs,
    ) = await asyncio.get_event_loop().run_in_executor(
        None, _get_analysis, req.fen, req.move_uci, req.depth
    )
    all_line_strs = [played_line_str_plain] + [" → ".join(line) for line in engine_line_sans]

    board_before_str = board_ascii(board)
    facts = move_facts(board, move)
    board_after = board.copy()
    board_after.push(move)
    board_after_str = board_ascii(board_after)

    user_content = format_joint_user_prompt(
        board_ascii_str=board_before_str,
        fen=req.fen,
        move_san=move_san,
        eval_str=eval_label,
        facts=facts,
        board_after_str=board_after_str,
        fen_after=board_after.fen(),
        played_line=played_line_str,
        key_lines=engine_line_strs,
    )

    messages = [
        {"role": "system", "content": JOINT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    messages_with_sentinels = inject_sentinels(messages, move_san)
    board_tensors = build_board_tensors(req.fen, move_san, engine_line_sans, played_line_sans)

    # Render the exact prompt text the tokenizer will see
    prompt_text = _tokenizer.apply_chat_template(
        messages_with_sentinels,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Count sentinels in tokenized form
    from src.encoder import MOVE_TOKEN_ID

    encoded = _tokenizer.apply_chat_template(
        messages_with_sentinels,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    n_sentinels = int((input_ids == MOVE_TOKEN_ID).sum().item())
    n_tokens = int(input_ids.shape[-1])

    # Run non-streaming generation to get raw output
    import queue as _queue
    import threading

    from transformers import TextIteratorStreamer

    device = next(_model.parameters()).device
    input_ids_dev = input_ids.to(device)
    attention_mask = (input_ids_dev != _tokenizer.pad_token_id).long()
    bt = board_tensors.to(device)

    import torch as _torch

    from src.encoder import MOVE_TOKEN_ID as _MTID
    from src.tutor.encoder_inference import generate_stream

    raw_chunks = []
    for chunk in generate_stream(
        _model,
        _tokenizer,
        messages_with_sentinels,
        board_tensors,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    ):
        raw_chunks.append(chunk)
    raw_output = "".join(raw_chunks)

    return {
        "move_san": move_san,
        "eval_label": eval_label,
        "played_line": played_line_str,
        "engine_lines": engine_line_strs,
        "n_input_tokens": n_tokens,
        "n_sentinel_tokens": n_sentinels,
        "n_board_tensors": int(board_tensors.shape[0]),
        "prompt_text": prompt_text,
        "raw_output": raw_output,
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "stockfish_ready": _engine is not None,
        "checkpoint": CHECKPOINT_DIR,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
