"""Encoder inference server for ChessLMWithEncoder.

Standalone FastAPI server that loads ChessLMWithEncoder and exposes:
  POST /api/encoder/analyze  — takes {fen, move_uci}, runs Stockfish for key
                               lines, builds the joint prompt, streams the
                               annotated lines + coaching comment.

Launch via: recipes-train/qwen3-4b-encoder-phase1-sft/serve.sh
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

CHECKPOINT_DIR = os.environ.get("ENCODER_CHECKPOINT", "checkpoints/qwen3-4b-encoder-phase1-sft")
CONFIG_PATH = os.environ.get(
    "ENCODER_CONFIG", "recipes-train/qwen3-4b-encoder-phase1-sft/config.yaml"
)
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
STOCKFISH_DEPTH = int(os.environ.get("STOCKFISH_DEPTH", "18"))
HOST = os.environ.get("ENCODER_HOST", "0.0.0.0")
PORT = int(os.environ.get("ENCODER_PORT", "8200"))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model, _tokenizer, _engine

    _logger.info("Loading encoder model from %s ...", CHECKPOINT_DIR)
    _model, _tokenizer = load_encoder_model(CHECKPOINT_DIR, CONFIG_PATH)
    _logger.info("Model loaded.")

    _logger.info("Starting Stockfish at %s", STOCKFISH_PATH)
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    _logger.info("Stockfish ready.")

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


def _get_key_lines(fen: str, depth: int, multipv: int) -> tuple[list[list[str]], str, int | None]:
    """Run Stockfish and return (key_lines, eval_label, root_cp).

    key_lines: list of SAN sequences, one per PV line.
    eval_label: human label for the position eval (from side to move's perspective).
    root_cp: centipawns from White's perspective (None if mate).
    """
    assert _engine is not None
    board = chess.Board(fen)
    result = _engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=multipv,
    )

    key_lines: list[list[str]] = []
    root_cp: int | None = None

    for i, info in enumerate(result):
        pv = info.get("pv", [])
        if not pv:
            continue

        score = info.get("score")
        if i == 0 and score is not None:
            s = score.white()
            if s.mate() is not None:
                root_cp = None
            else:
                root_cp = s.score()

        # Convert PV to SAN
        b = board.copy()
        sans: list[str] = []
        for mv in pv[:8]:  # cap at 8 half-moves
            try:
                sans.append(b.san(mv))
                b.push(mv)
            except Exception:
                break
        if sans:
            key_lines.append(sans)

    # Eval label from side-to-move perspective
    if root_cp is None:
        eval_label = "a forced mate sequence exists"
    else:
        cp = root_cp if board.turn == chess.WHITE else -root_cp
        if cp >= 200:
            eval_label = "winning for me"
        elif cp >= 60:
            eval_label = "good for me"
        elif cp >= -60:
            eval_label = "roughly equal"
        elif cp >= -200:
            eval_label = "good for my opponent"
        else:
            eval_label = "losing"

    return key_lines, eval_label, root_cp


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

    # Stockfish key lines (run in thread to not block event loop)
    try:
        key_lines, eval_label, root_cp = await asyncio.get_event_loop().run_in_executor(
            None, _get_key_lines, req.fen, req.depth, req.multipv
        )
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': f'Stockfish: {e}'})}\n\n"
        return

    # Emit meta (key lines as plain SAN strings for the UI to display)
    key_line_strs = [" → ".join(line) for line in key_lines]
    meta = {"eval_label": eval_label, "key_lines": key_line_strs, "move_san": move_san}
    yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

    # Build board ascii
    board_before_str = board_ascii(board)
    facts = move_facts(board, move)
    board_after = board.copy()
    board_after.push(move)
    board_after_str = board_ascii(board_after)

    # Build user prompt (plain SAN, no sentinels yet)
    user_content = format_joint_user_prompt(
        board_ascii_str=board_before_str,
        fen=req.fen,
        move_san=move_san,
        eval_str=eval_label,
        facts=facts,
        board_after_str=board_after_str,
        fen_after=board_after.fen(),
        key_lines=key_line_strs,
    )

    messages = [
        {"role": "system", "content": JOINT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Inject <|vision_pad|> sentinels
    messages_with_sentinels = inject_sentinels(messages, move_san)

    # Build board tensors
    board_tensors = build_board_tensors(req.fen, move_san, key_lines)

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
