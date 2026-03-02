"""Prepare HuggingFace datasets for chess coaching fine-tuning.

Three-phase pipeline:
1. Extract — download & parse HF datasets into intermediate JSONL
2. Augment — add Stockfish analysis (classification, candidates, threats)
3. Format  — compute move facts, format into training messages, split

Usage:
    uv run python data/pipeline/prepare_datasets.py \
        --stockfish /home/zheng/.local/bin/stockfish \
        --output-dir data/processed \
        --max-per-source 0  # 0 = no limit
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import logging
import random
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, cast

import chess
import chess.pgn

# Add src/ to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chess_mcp.stockfish import Stockfish
from tutor.analysis import (
    TreeNode,
    build_move_tree,
    compute_cct,
    compute_position_context,
    format_score,
    tree_node_from_dict,
    tree_node_to_dict,
)
from tutor.prompts import (
    SYSTEM_PROMPT,
    TEXTBOOK_FEW_SHOT,
    TEXTBOOK_SYSTEM_PROMPT,
    board_ascii,
    format_textbook_prompt,
    format_user_prompt,
)
from tutor.tools import CHESS_TOOLS
from tutor.tools import web_search as _web_search

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intermediate representation
# ---------------------------------------------------------------------------


@dataclass
class RawSample:
    """Intermediate format between extraction and augmentation."""

    fen: str
    move_uci: str
    move_san: str
    coaching_text: str  # the annotation or coaching response
    thinking_text: str  # for <think> block
    source: str


@dataclass
class AugmentedSample:
    """Sample with Stockfish analysis added."""

    fen: str
    move_uci: str
    move_san: str
    coaching_text: str
    thinking_text: str
    source: str
    classification: str
    eval_str: str
    best_move_san: str
    cp_loss: int
    candidates: list[str]  # e.g. ["e4 (+0.35)", "d4 (+0.30)"]
    opponent_threats: list[str]
    cct: dict[str, list[str]] = field(
        default_factory=dict
    )  # {"checks": [...], "captures": [...], "threats": [...]}
    tool_messages: list[dict] = field(default_factory=list)  # intermediate tool call turns
    move_tree: list[TreeNode] = field(default_factory=list)  # 3-level response tree
    position_context: dict = field(default_factory=dict)  # positional/structural/phase context


# ---------------------------------------------------------------------------
# Transformers — one per dataset source
# ---------------------------------------------------------------------------


class BaseTransformer(ABC):
    """Base class for dataset transformers."""

    @abstractmethod
    def extract(self, max_samples: int = 0) -> Iterator[RawSample]:
        """Yield RawSamples from the dataset source.

        Args:
            max_samples: Maximum samples to extract (0 = no limit).
        """
        ...


class ChessCotTransformer(BaseTransformer):
    """Transform kr4t0n/chess-cot into coaching samples.

    The dataset has: fen, board, reasoning, response (UCI), reward.
    We reframe 'find the best move' reasoning as coaching explanation.
    """

    DATASET_NAME = "kr4t0n/chess-cot"
    MIN_REWARD = 0.5
    MAX_THINKING_LEN = 6000

    # Substrings that flag meta-reasoning — notation checks, legality, bookkeeping
    _META_PATTERNS = (
        "uci notation",
        "in uci",
        "going from ",
        "passes through",
        "the move would be written",
        "let me check",
        "let me verify",
        "let me re-check",
        "wait, let me",
        "to verify",
        "double-check",
        "is indeed a legal",
        "is a legal move",
        "no pieces occupying",
        "starting and ending square",
        "looking at the board representation",
        "the 1st rank",
        "the board representation",
        "can legally move",
        "i don't see any immediate",
        "there are no immediate",
        "i'm playing as",
        "i am playing as",
        "threats:",
        "it's a standard opening",
        "it is a standard opening",
    )
    # Keywords that indicate genuine chess insight
    # Strategic/analytical keywords — excludes bare piece names (rook/bishop/queen)
    # which also appear in position-enumeration paragraphs
    _CHESS_KEYWORDS = (
        "gambit",
        "defense",
        "attack",
        "develop",
        "controls the center",
        "pawn structure",
        "king safety",
        "tactical",
        "strategic",
        "initiative",
        "activity",
        "pressure",
        "weakness",
        "advantage",
        "tempo",
        "space",
        "counterplay",
        "fork",
        "pin",
        "skewer",
        "checkmate",
        "mate in",
        "endgame",
        "opening",
        "middlegame",
        "because",
        "allows",
        "prevents",
        "creates",
        "improves",
        "activates",
        "centralizes",
        "stronger",
        "weaker",
        "winning",
        "losing",
        "equal position",
    )

    # Regex: paragraph is mostly bullet-list lines (piece enumeration)
    @classmethod
    def _is_list_paragraph(cls, para: str) -> bool:
        """Return True if the paragraph is mostly bullet-list lines."""
        lines = [ln for ln in para.splitlines() if ln.strip()]
        if not lines:
            return False
        bullet_lines = sum(1 for ln in lines if ln.strip().startswith("- "))
        return bullet_lines / len(lines) >= 0.5

    @classmethod
    def _extract_coaching(cls, reasoning: str) -> str:
        """Extract the most chess-insightful prose paragraph from the reasoning.

        The move is already known from the `response` field. Skip:
        - Bullet-list position enumerations ("- Rook on a1 ...")
        - Meta-reasoning (notation, legality checks)
        Then return the first paragraph that contains genuine chess insight.
        """
        paragraphs = [p.strip() for p in reasoning.split("\n\n") if p.strip()]

        fallback: str = ""
        for para in paragraphs:
            lower = para.lower()
            if len(para) < 60:
                continue
            if cls._is_list_paragraph(para):
                continue
            if any(pattern in lower for pattern in cls._META_PATTERNS):
                continue
            if any(kw in lower for kw in cls._CHESS_KEYWORDS):
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
                snippet = " ".join(sentences[-2:]) if len(sentences) >= 2 else para
                return snippet[:500].strip()
            if not fallback:
                fallback = para

        return fallback[:500].strip() if fallback else reasoning[:300].strip()

    def extract(self, max_samples: int = 0) -> Iterator[RawSample]:
        from datasets import load_dataset

        logger.info("Loading %s...", self.DATASET_NAME)
        ds = load_dataset(self.DATASET_NAME, split="train")

        count = 0
        for _row in ds:
            if max_samples and count >= max_samples:
                break

            row: dict[str, Any] = cast(dict[str, Any], _row)
            # Filter by reward quality
            if row.get("reward", 0) < self.MIN_REWARD:
                continue

            fen = row.get("fen", "")
            response = row.get("response", "")
            reasoning = row.get("reasoning", "")

            if not fen or not response:
                continue

            # Validate the move
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(response)
                if move not in board.legal_moves:
                    continue
                san = board.san(move)
            except (ValueError, chess.InvalidMoveError):
                continue

            # Truncate long reasoning for <think> block
            thinking = reasoning[: self.MAX_THINKING_LEN]
            if len(reasoning) > self.MAX_THINKING_LEN:
                thinking += "..."

            # Extract strategic coaching insight from the reasoning conclusion.
            # The reasoning typically ends with "So my move is X" — we want
            # the substantive paragraph just before that conclusion.
            coaching = self._extract_coaching(reasoning)

            yield RawSample(
                fen=fen,
                move_uci=response,
                move_san=san,
                coaching_text=coaching,
                thinking_text=thinking,
                source="chess_cot",
            )
            count += 1

        logger.info("chess-cot: extracted %d samples", count)


class IcannosTransformer(BaseTransformer):
    """Transform Icannos/chess_studies PGN annotations into coaching samples.

    Reuses quality-filtering logic from extract_annotations.py.
    """

    DATASET_NAME = "Icannos/chess_studies"
    MIN_ANNOTATION_LENGTH = 60

    # Instructive keywords (from extract_annotations.py)
    INSTRUCTIVE_KEYWORDS = [
        "because",
        "since",
        "therefore",
        "thus",
        "hence",
        "in order to",
        "so that",
        "allowing",
        "weak",
        "strong",
        "control",
        "pressure",
        "pawn structure",
        "outpost",
        "file",
        "diagonal",
        "attack",
        "defend",
        "threat",
        "pin",
        "fork",
        "discovery",
        "skewer",
        "sacrifice",
        "exchange",
        "development",
        "center",
        "initiative",
        "tempo",
        "king safety",
        "castle",
        "activity",
        "plan",
        "idea",
        "strategy",
        "positional",
        "advantage",
        "compensation",
        "equality",
        "better",
        "worse",
        "winning",
        "losing",
    ]

    # Lichess board annotation markup (arrows, colored squares) — not instructive text
    _LICHESS_MARKUP_RE = re.compile(r"\[%(?:cal|csl)\s+[^\]]*\]")
    # Patterns that indicate interactive prompts rather than coaching
    _INTERACTIVE_PHRASES = [
        "what would you play",
        "what do you think",
        "can you find",
        "your turn",
        "remember the movements",
        "remember the moves",
        "psst,",
        "with the arrows",
        "as i indicate",
    ]

    def _clean_comment(self, comment: str) -> str:
        """Strip Lichess markup and normalize whitespace."""
        cleaned = self._LICHESS_MARKUP_RE.sub("", comment)
        return " ".join(cleaned.split())

    def _is_quality(self, comment: str) -> bool:
        """Check if annotation is instructive enough."""
        cleaned = self._clean_comment(comment)
        if len(cleaned) < self.MIN_ANNOTATION_LENGTH:
            return False
        lower = cleaned.lower()
        # Reject interactive prompts / questions
        if any(phrase in lower for phrase in self._INTERACTIVE_PHRASES):
            return False
        # Reject annotations that end with a question (quizzes, not coaching)
        if cleaned.rstrip().endswith("?"):
            return False
        # Must contain at least one instructive keyword
        return any(kw in lower for kw in self.INSTRUCTIVE_KEYWORDS)

    def _extract_from_pgn(self, pgn_text: str) -> Iterator[RawSample]:
        """Extract annotated positions from a single PGN string."""
        pgn_io = io.StringIO(pgn_text)

        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break

            board = game.board()
            node = game

            while node.variations:
                next_node = node.variation(0)
                if next_node.move:
                    raw_comment = (next_node.comment or "").strip()
                    comment = self._clean_comment(raw_comment)

                    if self._is_quality(raw_comment):
                        fen = board.fen()
                        move = next_node.move
                        try:
                            san = board.san(move)
                        except Exception:
                            board.push(move)
                            node = next_node
                            continue

                        yield RawSample(
                            fen=fen,
                            move_uci=move.uci(),
                            move_san=san,
                            coaching_text=comment,
                            thinking_text="",
                            source="icannos_studies",
                        )

                    board.push(next_node.move)
                node = next_node

    # Direct CSV URLs (the HF loading script is no longer supported)
    CSV_URLS = [
        "https://huggingface.co/datasets/Icannos/chess_studies/resolve/main/lichess_studies.csv",
        "https://huggingface.co/datasets/Icannos/chess_studies/resolve/main/others.csv",
    ]

    def extract(self, max_samples: int = 0) -> Iterator[RawSample]:
        import csv

        import httpx

        logger.info("Loading %s (direct CSV download)...", self.DATASET_NAME)

        count = 0
        for url in self.CSV_URLS:
            if max_samples and count >= max_samples:
                break

            logger.info("Downloading %s...", url.split("/")[-1])
            resp = httpx.get(url, follow_redirects=True, timeout=60.0)
            resp.raise_for_status()

            reader = csv.DictReader(io.StringIO(resp.text))
            for row in reader:
                if max_samples and count >= max_samples:
                    break

                pgn_text = row.get("text", "")
                if not pgn_text:
                    continue

                for sample in self._extract_from_pgn(pgn_text):
                    if max_samples and count >= max_samples:
                        break
                    yield sample
                    count += 1

        logger.info("icannos: extracted %d samples", count)


# TODO(jaisonkumar): Jaisonkumar/chess-annotation-dataset removed due to quality issues.
# Dataset had inconsistent annotations and high noise. If reconsidering:
# - Implement better filtering (language detection, annotation quality gates)
# - Consider only high-confidence English annotations
# - Validate against Stockfish before inclusion
# See commit history for full implementation if needed.


class TextbookTransformer(BaseTransformer):
    """Transform existing augmented.jsonl from the textbook pipeline."""

    def __init__(self, augmented_path: Path):
        self.augmented_path = augmented_path

    def extract(self, max_samples: int = 0) -> Iterator[RawSample]:
        if not self.augmented_path.exists():
            logger.warning("Textbook data not found at %s, skipping", self.augmented_path)
            return

        logger.info("Loading textbook data from %s...", self.augmented_path)
        count = 0

        with open(self.augmented_path, encoding="utf-8") as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                fen = row.get("fen", "")
                move_uci = row.get("move_uci", "")
                move_san = row.get("move_san", "")
                annotation = row.get("annotation", "")

                if not fen or not move_uci or not annotation:
                    continue

                # Skip chess variant FENs (e.g. Crazyhouse has "[Pqp]" pocket notation)
                if "[" in fen:
                    continue

                # Use concepts for thinking text
                concepts = row.get("concepts", [])
                thinking = ""
                if concepts:
                    thinking = f"This position involves {', '.join(concepts)}."

                yield RawSample(
                    fen=fen,
                    move_uci=move_uci,
                    move_san=move_san,
                    coaching_text=annotation,
                    thinking_text=thinking,
                    source="textbook",
                )
                count += 1

        logger.info("textbook: extracted %d samples", count)


# ---------------------------------------------------------------------------
# Stockfish augmentation
# ---------------------------------------------------------------------------


def _classify_cp_loss(cp_loss: int) -> str:
    """Classify move quality by centipawn loss."""
    if cp_loss <= 10:
        return "Best"
    if cp_loss <= 30:
        return "Great"
    if cp_loss <= 80:
        return "Good"
    if cp_loss <= 150:
        return "Inaccuracy"
    if cp_loss <= 300:
        return "Mistake"
    return "Blunder"


# format_score, compute_cct, TreeNode, build_move_tree imported from tutor.analysis


async def _analyze_one(
    engine: "Stockfish",
    sample: RawSample,
    depth: int,
) -> AugmentedSample | None:
    """Run Stockfish analysis for a single sample. Returns None on failure."""
    comparison = await engine.compare_moves(sample.fen, sample.move_uci, depth)
    if "error" in comparison:
        return None

    cp_loss = comparison.get("cp_loss", 0)
    best_move_uci = comparison.get("best_move", "")

    analysis = await engine.analyze(sample.fen, depth=depth, multipv=3)
    score = analysis.score
    eval_str = format_score(
        score.mate_in if score.mate_in is not None else (score.centipawns or 0),
        score.mate_in is not None,
    )

    board = chess.Board(sample.fen)
    candidates = []
    for line in analysis.lines[:3]:
        try:
            m = chess.Move.from_uci(line.best_move)
            s = board.san(m)
            ls = line.score
            sc = format_score(
                ls.mate_in if ls.mate_in is not None else (ls.centipawns or 0),
                ls.mate_in is not None,
            )
            candidates.append(f"{s} ({sc})")
        except Exception:
            pass

    try:
        best_move_san = board.san(chess.Move.from_uci(best_move_uci))
    except Exception:
        best_move_san = best_move_uci

    try:
        threats_data = await engine.get_threats(sample.fen, depth=depth)
        threat_moves = threats_data.get("threats", [])
        opponent_threats = []
        threat_board = board.copy()
        threat_board.turn = not threat_board.turn
        for tm in threat_moves[:3]:
            try:
                opponent_threats.append(threat_board.san(chess.Move.from_uci(tm)))
            except Exception:
                pass
    except Exception:
        opponent_threats = []

    cct = compute_cct(board)
    pos_ctx = compute_position_context(board)

    # Build 3-level continuation tree (13 Stockfish calls, depth=12 for speed)
    board_after = board.copy()
    board_after.push(chess.Move.from_uci(sample.move_uci))
    try:
        move_tree = await build_move_tree(engine, board_after, depth=min(depth, 12))
    except Exception as e:
        logger.warning("Move tree build failed for %s: %s", sample.move_uci, e)
        move_tree = []

    return AugmentedSample(
        fen=sample.fen,
        move_uci=sample.move_uci,
        move_san=sample.move_san,
        coaching_text=sample.coaching_text,
        thinking_text=sample.thinking_text,
        source=sample.source,
        classification=_classify_cp_loss(cp_loss),
        eval_str=eval_str,
        best_move_san=best_move_san,
        cp_loss=cp_loss,
        candidates=candidates,
        opponent_threats=opponent_threats,
        cct=cct,
        move_tree=move_tree,
        position_context=pos_ctx,
    )


async def augment_samples(
    samples: list[RawSample],
    stockfish_path: str,
    depth: int = 16,
    cache_path: Path | None = None,
    workers: int = 4,
) -> list[AugmentedSample]:
    """Add Stockfish analysis to raw samples using a parallel worker pool.

    Args:
        samples: Raw extracted samples.
        stockfish_path: Path to Stockfish binary.
        depth: Analysis depth (16 is good balance of speed/accuracy).
        cache_path: Optional JSONL cache for resumable augmentation.
        workers: Number of parallel Stockfish processes.

    Returns:
        List of augmented samples (order matches input).
    """
    # Load cache
    cache: dict[str, dict] = {}
    if cache_path:
        cache = _load_jsonl_cache(cache_path, key_field="_cache_key")
        if cache:
            logger.info("Loaded %d cached augmentations", len(cache))

    # Pre-compute cache keys; separate hits from misses.
    # v3: includes position_context — old entries won't match, forcing regeneration.
    keys = [hashlib.md5(f"v3:{s.fen}:{s.move_uci}".encode()).hexdigest() for s in samples]
    todo_indices = [i for i, k in enumerate(keys) if k not in cache]
    logger.info(
        "Cache: %d hits, %d to augment",
        len(samples) - len(todo_indices),
        len(todo_indices),
    )

    # Result slot per input sample (None = failed or cached-miss)
    aug_results: list[AugmentedSample | None] = [None] * len(samples)

    # Fill in cached results immediately
    for i, sample in enumerate(samples):
        k = keys[i]
        if k in cache:
            entry = cache[k]
            aug_results[i] = AugmentedSample(
                fen=sample.fen,
                move_uci=sample.move_uci,
                move_san=sample.move_san,
                coaching_text=sample.coaching_text,
                thinking_text=sample.thinking_text,
                source=sample.source,
                classification=entry["classification"],
                eval_str=entry["eval_str"],
                best_move_san=entry["best_move_san"],
                cp_loss=entry["cp_loss"],
                candidates=entry["candidates"],
                opponent_threats=entry["opponent_threats"],
                cct=entry.get("cct", {}),
                move_tree=[tree_node_from_dict(n) for n in entry.get("move_tree", [])],
                position_context=entry.get("position_context", {}),
            )

    if todo_indices:
        # Build work queue
        queue: asyncio.Queue[int] = asyncio.Queue()
        for idx in todo_indices:
            await queue.put(idx)

        cache_lock = asyncio.Lock()
        cache_file = open(cache_path, "a", encoding="utf-8") if cache_path else None
        done_count = [0]  # mutable counter shared across workers

        async def worker() -> None:
            engine = Stockfish(path=stockfish_path, depth=depth, threads=1, hash_mb=256)
            await engine.start()
            try:
                while True:
                    try:
                        idx = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    try:
                        aug = await _analyze_one(engine, samples[idx], depth)
                        if aug is not None:
                            aug_results[idx] = aug
                            if cache_file:
                                entry = {
                                    "_cache_key": keys[idx],
                                    "classification": aug.classification,
                                    "eval_str": aug.eval_str,
                                    "best_move_san": aug.best_move_san,
                                    "cp_loss": aug.cp_loss,
                                    "candidates": aug.candidates,
                                    "opponent_threats": aug.opponent_threats,
                                    "cct": aug.cct,
                                    "move_tree": [tree_node_to_dict(n) for n in aug.move_tree],
                                    "position_context": aug.position_context,
                                }
                                async with cache_lock:
                                    cache_file.write(json.dumps(entry) + "\n")
                                    cache_file.flush()
                    except Exception as e:
                        logger.warning("Error augmenting sample %d: %s", idx, e)
                        # Restart dead engine to prevent cascade failures
                        err = str(e).lower()
                        if any(
                            k in err
                            for k in ("connection lost", "eof", "process died", "broken pipe")
                        ):
                            try:
                                await engine.stop()
                            except Exception:
                                pass
                            engine = Stockfish(
                                path=stockfish_path, depth=depth, threads=1, hash_mb=256
                            )
                            try:
                                await engine.start()
                                logger.info("Engine restarted after crash (sample %d)", idx)
                            except Exception as restart_err:
                                logger.warning("Failed to restart engine: %s", restart_err)
                                break  # give up this worker slot
                    finally:
                        queue.task_done()
                        done_count[0] += 1
                        if done_count[0] % 100 == 0:
                            logger.info(
                                "Augmented %d / %d samples",
                                done_count[0],
                                len(todo_indices),
                            )
            finally:
                try:
                    await engine.stop()
                except Exception:
                    pass  # process already dead — ignore

        try:
            await asyncio.gather(*[worker() for _ in range(min(workers, len(todo_indices)))])
        finally:
            if cache_file:
                cache_file.close()

    results = [r for r in aug_results if r is not None]
    logger.info("Augmentation complete: %d / %d succeeded", len(results), len(samples))
    return results


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_jsonl_cache(path: Path, key_field: str = "_key") -> dict[str, dict]:
    """Load a JSONL cache file into a dict keyed by *key_field*."""
    cache: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    k = entry.get(key_field, "")
                    if k:
                        cache[k] = entry
                except json.JSONDecodeError:
                    continue
    return cache


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_COMMENT_RE = re.compile(r"<comment>(.*?)</comment>", re.DOTALL)


def _strip_thinking(text: str) -> tuple[str, str]:
    """Extract the thinking block from LLM output.

    Handles the Qwen3 quirk where ``</think>`` appears without ``<think>``.
    Returns (thinking_text, text_with_think_removed).
    """
    # Normalize missing opening tag (Qwen3 quirk)
    if "</think>" in text and "<think>" not in text:
        text = "<think>" + text
    match = _THINK_RE.search(text)
    if match:
        thinking = match.group(0)[len("<think>") : -len("</think>")].strip()
        remainder = _THINK_RE.sub("", text).strip()
        return thinking, remainder
    if "<think>" in text:  # truncated — no closing tag
        return "", ""
    return "", text  # no thinking block at all


def _extract_comment(text: str) -> str:
    """Extract the coaching comment from a <comment>...</comment> block.

    Always strips the <think> block first so that any mention of
    ``<comment>`` inside the thinking section (e.g. "wrapped in <comment>
    tags.") is not mistakenly matched as the output tag.

    Falls back to the post-think remainder if no comment tags are present,
    for backwards compatibility with cached responses.
    """
    _, remainder = _strip_thinking(text)
    search_in = remainder if remainder.strip() else text
    match = _COMMENT_RE.search(search_in)
    if match:
        return match.group(1).strip()
    return remainder


_STUB_RE = re.compile(r"^[.\s…\-_*]+$")  # only dots, spaces, dashes, etc.
_MAX_COACHING_LEN = 2000


def _is_valid_coaching(text: str) -> bool:
    """Return False for empty, SKIP, stubs, or overly long text."""
    stripped = text.strip()
    if not stripped or stripped.upper() == "SKIP":
        return False
    if _STUB_RE.match(stripped):
        return False
    if len(stripped) < 20:  # too short to be a real coaching comment
        return False
    if len(stripped) > _MAX_COACHING_LEN:  # overly long / padding
        return False
    return True


async def _dispatch_tool_call(tc: Any, aug_fen: str, sf_pool: Any) -> tuple[str, str]:
    """Execute a single tool call, return (result_json, trace_line)."""
    args = json.loads(tc.function.arguments)
    if tc.function.name == "web_search":
        query = args.get("query", "")
        result_str = await _web_search(query)
        return result_str, f"web_search({query!r}) → {result_str[:200]}"
    # analyze_position
    fen_q = args.get("fen", aug_fen)
    multipv = min(int(args.get("multipv", 3)), 5)
    tool_result = await sf_pool.analyze(fen_q, multipv=multipv)
    result_str = json.dumps(tool_result)
    return result_str, f"analyze_position(fen={fen_q}, multipv={multipv}) → {result_str}"


# ---------------------------------------------------------------------------
# LLM coaching generation with Stockfish tool use (Phase 2.5)
# ---------------------------------------------------------------------------


class StockfishPool:
    """Pool of Stockfish instances shared by the LLM tool-call handler."""

    def __init__(self, stockfish_path: str, size: int = 8, depth: int = 14):
        self._path = stockfish_path
        self._size = size
        self._depth = depth
        self._queue: asyncio.Queue = asyncio.Queue()

    async def start(self) -> None:
        for _ in range(self._size):
            sf = Stockfish(path=self._path, depth=self._depth, threads=1, hash_mb=128)
            await sf.start()
            self._queue.put_nowait(sf)

    async def stop(self) -> None:
        while not self._queue.empty():
            sf = self._queue.get_nowait()
            await sf.stop()

    async def analyze(self, fen: str, multipv: int = 3) -> dict:
        """Run analysis and return a JSON-serialisable result dict."""
        sf: Stockfish = await asyncio.wait_for(self._queue.get(), timeout=30.0)
        try:
            board = chess.Board(fen)
            analysis = await asyncio.wait_for(
                sf.analyze(fen, depth=self._depth, multipv=multipv), timeout=30.0
            )
            sc = analysis.score
            eval_str = format_score(
                sc.mate_in if sc.mate_in is not None else (sc.centipawns or 0),
                sc.mate_in is not None,
            )
            result: dict = {
                "eval": eval_str,
                "side_to_move": "White" if board.turn == chess.WHITE else "Black",
                "top_moves": [],
            }
            for line in analysis.lines[:multipv]:
                try:
                    m = chess.Move.from_uci(line.best_move)
                    ls = line.score
                    lsc = format_score(
                        ls.mate_in if ls.mate_in is not None else (ls.centipawns or 0),
                        ls.mate_in is not None,
                    )
                    # Build short PV in SAN
                    pv_board = board.copy()
                    pv_sans: list[str] = []
                    for uci in line.pv[:5]:
                        try:
                            pv_m = chess.Move.from_uci(uci)
                            pv_sans.append(pv_board.san(pv_m))
                            pv_board.push(pv_m)
                        except Exception:
                            break
                    result["top_moves"].append(
                        {"move": board.san(m), "eval": lsc, "pv": " ".join(pv_sans)}
                    )
                except Exception:
                    pass
            return result
        finally:
            self._queue.put_nowait(sf)


def _game_phase(fen: str) -> str:
    """Classify a position as 'opening', 'middlegame', or 'endgame'."""
    board = chess.Board(fen)
    fullmove = board.fullmove_number
    non_king = sum(
        len(board.pieces(pt, c))
        for c in chess.COLORS
        for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    )
    has_queens = bool(
        board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK)
    )
    if fullmove <= 12 and non_king >= 24:
        return "opening"
    if non_king <= 10 or (not has_queens and non_king <= 16):
        return "endgame"
    return "middlegame"


async def _coach_one(
    aug: AugmentedSample,
    client: Any,  # AsyncOpenAI — imported lazily inside generate_coaching_with_llm
    llm_model: str,
    sf_pool: StockfishPool,
) -> tuple[str, str | None, list[dict]]:
    """Run one sample through the agentic coaching loop.

    Returns (new_thinking_text, new_coaching_text, tool_turns).
    tool_turns is the list of intermediate assistant-with-tool-calls + tool-result
    messages to include in the training data so the model learns tool use.
    Falls back to (aug.thinking_text, "", []) on failure.
    """
    has_prior_thinking = bool(aug.thinking_text)

    board = chess.Board(aug.fen)

    # Compute the FEN after the move so the LLM can analyze the resulting position
    board_after = board.copy()
    board_after.push(chess.Move.from_uci(aug.move_uci))
    fen_after = board_after.fen()
    phase = _game_phase(aug.fen)

    # Textbook samples: single direct rephrase call — no tools, no fact injection.
    # Facts are excluded to prevent the LLM from overriding the expert's stated reasoning
    # with mechanical observations that may be true but irrelevant to the expert's point.
    is_textbook = aug.source == "textbook" and bool(aug.coaching_text)
    if is_textbook:
        user_msg = format_textbook_prompt(
            board_ascii_str=board_ascii(board),
            san=aug.move_san,
            classification=aug.classification,
            eval_str=aug.eval_str,
            expert_annotation=aug.coaching_text,
            fen=aug.fen,
        )
        few_shot: list[dict] = []
        for user_ex, asst_ex in TEXTBOOK_FEW_SHOT:
            few_shot.append({"role": "user", "content": user_ex})
            few_shot.append({"role": "assistant", "content": asst_ex})
        messages = [
            {"role": "system", "content": TEXTBOOK_SYSTEM_PROMPT},
            *few_shot,
            {"role": "user", "content": user_msg},
        ]
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=llm_model,
                    messages=messages,
                    max_tokens=16384,
                    temperature=1.0,
                    extra_body={"thinking_token_budget": 1024},
                ),
                timeout=900.0,
            )
        except Exception as e:
            logger.warning("LLM textbook error for %s %s: %s", aug.move_san, aug.fen[:20], e)
            return aug.thinking_text, None, []
        msg = resp.choices[0].message
        raw = (msg.content or "").strip()
        # vLLM --reasoning-parser returns thinking in model_extra['reasoning']
        lm_thinking = ((msg.model_extra or {}).get("reasoning") or "").strip()
        if not lm_thinking:
            lm_thinking, _ = _strip_thinking(raw)
        text = _extract_comment(raw)
        if text.strip().upper() == "SKIP" or not _is_valid_coaching(text):
            return aug.thinking_text, None, []  # None signals: discard this sample
        return lm_thinking, text, []
    else:
        user_msg = format_user_prompt(
            board_ascii_str=board_ascii(board),
            san=aug.move_san,
            classification=aug.classification,
            eval_str=aug.eval_str,
            best_move=aug.best_move_san,
            cp_loss=aug.cp_loss,
            candidates=aug.candidates,
            opponent_threats=aug.opponent_threats,
            fen=aug.fen,
            cct=aug.cct or {},
            move_tree=aug.move_tree or [],
            position_context=aug.position_context or {},
        )
        # Phase-specific tool directive
        if phase == "opening":
            tool_directive = (
                "\n\nThis is an OPENING position. Focus on opening theory — name the "
                "opening or variation and state whether this is a recognized book move "
                "or a deviation. Use web_search to look up the exact opening name. "
                "When calling web_search, include the FULL move sequence from move 1 in your "
                "query, e.g. 'Ruy Lopez e4 e5 Nf3 Nc6 Bb5' — do NOT use FEN strings or "
                "isolated moves like 'Bb5 theory'. "
                "You do NOT need to call analyze_position unless the move is an "
                "Inaccuracy/Mistake/Blunder and you want to show the correct continuation."
            )
        elif phase == "endgame":
            tool_directive = (
                f"\n\nThis is an ENDGAME position. You MUST call analyze_position to "
                f"explore key variations before writing your comment. Analyze the "
                f"position AFTER this move (FEN = {fen_after}) and compare with the "
                f"best alternative (FEN = {aug.fen}) to explain the winning or losing idea. "
                f"Feel free to make multiple tool calls to trace concrete lines."
            )
        else:  # middlegame
            tool_directive = (
                f"\n\nThis is a MIDDLEGAME position. Call analyze_position to verify "
                f"the key tactical or strategic lines before commenting. "
                f"Start by analyzing the position AFTER this move (FEN = {fen_after}). "
                f"You may also analyze before the move (FEN = {aug.fen}) to compare alternatives."
            )
        system_prompt = SYSTEM_PROMPT

    user_msg_with_directive = user_msg + tool_directive

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg_with_directive},
    ]
    tool_trace: list[str] = []

    for round_idx in range(7):  # up to 5 tool calls + 1 final answer + 1 safety
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=llm_model,
                    messages=messages,
                    tools=CHESS_TOOLS,
                    tool_choice="auto",
                    max_tokens=16384,
                    temperature=1.0,
                    extra_body={"thinking_token_budget": 1024},
                ),
                timeout=300.0,
            )
        except Exception as e:
            logger.warning("LLM call error for %s %s: %s", aug.move_san, aug.fen[:20], e)
            return aug.thinking_text, None, []

        choice = resp.choices[0]
        msg = choice.message

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            # Append assistant turn with tool_calls
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                try:
                    result_str, trace = await _dispatch_tool_call(tc, aug.fen, sf_pool)
                    tool_trace.append(trace)
                except Exception as e:
                    result_str = json.dumps({"error": str(e)})
                    tool_trace.append(f"{tc.function.name} error: {e}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
            continue  # back to LLM with tool results

        # Final response — extract coaching text and tool turns
        tool_turns = messages[2:]
        raw = (msg.content or "").strip()
        # vLLM --reasoning-parser returns thinking in model_extra['reasoning']
        lm_thinking = ((msg.model_extra or {}).get("reasoning") or "").strip()
        if not lm_thinking:
            lm_thinking, _ = _strip_thinking(raw)
        text = _extract_comment(raw)

        if has_prior_thinking:
            # chess_cot: discard LLM thinking, keep original reasoning
            thinking = aug.thinking_text
            if tool_trace:
                thinking += "\n\n[Engine exploration]\n" + "\n".join(tool_trace)
        else:
            # Icannos/textbook: use LLM thinking + tool trace
            parts: list[str] = []
            if lm_thinking:
                parts.append(lm_thinking)
            if tool_trace:
                parts.append("[Engine exploration]\n" + "\n".join(tool_trace))
            thinking = "\n\n".join(parts)

        if not _is_valid_coaching(text):
            logger.warning(
                "Stub/empty coaching %r for %s %s after %d rounds (finish=%s)",
                text[:40],
                aug.move_san,
                aug.fen[:20],
                round_idx + 1,
                choice.finish_reason,
            )
            text = ""
        return thinking, text, tool_turns

    logger.warning(
        "Loop exhausted (7 rounds) for %s %s — no final answer", aug.move_san, aug.fen[:20]
    )
    return aug.thinking_text, None, []


async def generate_coaching_with_llm(
    samples: list[AugmentedSample],
    llm_base_url: str,
    llm_model: str,
    stockfish_path: str,
    cache_path: Path | None = None,
    llm_workers: int = 128,
) -> list[AugmentedSample]:
    """Generate grounded coaching using an LLM agent with Stockfish tool use.

    The LLM can call analyze_position() to explore variations before writing
    its coaching comment, eliminating hallucinated piece positions and tactics.
    Results are cached so the step is fully resumable.
    """
    import logging as _logging
    import time as _time

    from openai import AsyncOpenAI

    # Suppress httpx request-level logs from the OpenAI client
    _logging.getLogger("httpx").setLevel(_logging.WARNING)

    # Load cache — entries store {"_key", "coaching", "thinking"(optional)}
    coaching_cache: dict[str, dict] = {}
    if cache_path:
        coaching_cache = _load_jsonl_cache(cache_path)
        if coaching_cache:
            logger.info("LLM coaching cache: %d entries loaded", len(coaching_cache))

    client = AsyncOpenAI(base_url=llm_base_url, api_key="dummy")
    cache_lock = asyncio.Lock()
    cache_file = open(cache_path, "a", encoding="utf-8") if cache_path else None
    done_count = [0]  # all samples processed (cache hits + generations)
    generated_count = [0]  # only LLM generations (the bottleneck)
    start_time = [_time.monotonic()]

    # Stockfish pool for agentic tool calls — 8 instances queue naturally
    sf_pool = StockfishPool(stockfish_path, size=8, depth=14)
    await sf_pool.start()

    results = list(samples)

    async def rewrite_one(idx: int, aug: AugmentedSample) -> None:
        import time

        has_thinking = bool(aug.thinking_text)
        # Textbook uses a separate version key so fixes to the textbook prompt
        # invalidate only textbook cache entries, leaving chess_cot/icannos intact.
        is_tb = aug.source == "textbook"
        key_str = (
            f"llm9:{aug.source}:{aug.fen}:{aug.move_uci}:{has_thinking}"
            if is_tb
            else f"llm7:{aug.fen}:{aug.move_uci}:{has_thinking}"
        )
        cache_key = hashlib.md5(key_str.encode()).hexdigest()

        if cache_key in coaching_cache:
            entry = coaching_cache[cache_key]
            if entry.get("_skip"):
                results[idx] = None  # type: ignore[assignment]
                done_count[0] += 1
                return
            results[idx] = dataclasses.replace(
                aug,
                coaching_text=entry["coaching"],
                thinking_text=entry.get("thinking", aug.thinking_text),
                tool_messages=entry.get("tool_messages", []),
            )
            done_count[0] += 1
            return

        t0 = time.monotonic()
        new_thinking, new_coaching, new_tool_messages = await _coach_one(
            aug, client, llm_model, sf_pool
        )
        elapsed = time.monotonic() - t0
        n_tool_calls = len([m for m in new_tool_messages if m.get("role") == "assistant"])
        generated_count[0] += 1  # Track LLM generations (the bottleneck)

        if new_coaching is None:
            # LLM signalled SKIP — cache decision and drop from results
            results[idx] = None  # type: ignore[assignment]
            async with cache_lock:
                if cache_file:
                    skip_entry: dict = {"_key": cache_key, "_skip": True}
                    if new_thinking:
                        skip_entry["thinking"] = new_thinking
                    cache_file.write(json.dumps(skip_entry) + "\n")
                    cache_file.flush()
            done_count[0] += 1
            logger.info(
                "[%d/%d] SKIP (LLM filtered): %s %s",
                done_count[0],
                len(samples),
                aug.move_san,
                aug.fen[:20],
            )
        elif new_coaching:
            results[idx] = dataclasses.replace(
                aug,
                coaching_text=new_coaching,
                thinking_text=new_thinking,
                tool_messages=new_tool_messages,
            )
            async with cache_lock:
                if cache_file:
                    entry_out: dict = {
                        "_key": cache_key,
                        "coaching": new_coaching,
                        "tool_messages": new_tool_messages,
                    }
                    if new_thinking:
                        entry_out["thinking"] = new_thinking
                    cache_file.write(json.dumps(entry_out) + "\n")
                    cache_file.flush()
            done_count[0] += 1
            logger.info(
                "[%d/%d] %s → %d chars, %d tool calls (%.1fs)",
                done_count[0],
                len(samples),
                aug.move_san,
                len(new_coaching),
                n_tool_calls,
                elapsed,
            )
        else:
            done_count[0] += 1
            logger.warning(
                "[%d/%d] empty coaching: %s %s (%.1fs)",
                done_count[0],
                len(samples),
                aug.move_san,
                aug.fen[:20],
                elapsed,
            )

    total = len(samples)
    logger.info(
        "LLM coaching %d samples via %s (%d workers)",
        total,
        llm_base_url,
        llm_workers,
    )
    start_time[0] = _time.monotonic()

    # Bounded queue provides backpressure: producer blocks when workers are busy.
    # Queue size = 2× workers so workers always have a next item ready.
    queue: asyncio.Queue[tuple[int, AugmentedSample] | None] = asyncio.Queue(
        maxsize=llm_workers * 2
    )

    async def producer() -> None:
        for i, s in enumerate(samples):
            await queue.put((i, s))
        # One sentinel per worker to signal shutdown
        for _ in range(llm_workers):
            await queue.put(None)

    async def worker() -> None:
        while True:
            item = await queue.get()
            if item is None:
                return
            idx, aug = item
            await rewrite_one(idx, aug)

    async def progress_reporter() -> None:
        """Log speed and ETA every 60 seconds (based on LLM generation rate, not cache hits)."""
        while True:
            await asyncio.sleep(60)
            done = done_count[0]
            generated = generated_count[0]
            if done >= total:
                return
            if generated == 0:
                continue
            elapsed = _time.monotonic() - start_time[0]
            gen_rate = generated / elapsed  # LLM generations/s (the bottleneck)
            remaining = total - done
            eta_s = int(remaining / gen_rate) if gen_rate > 0 else 0
            eta_h, eta_rem = divmod(eta_s, 3600)
            eta_m, eta_s2 = divmod(eta_rem, 60)
            eta_str = f"{eta_h}h {eta_m}m" if eta_h else f"{eta_m}m {eta_s2}s"
            logger.info(
                "\033[94m[Progress] %d/%d total (%d gen @ %.2f gen/s) — ETA: %s\033[0m",
                done,
                total,
                generated,
                gen_rate,
                eta_str,
            )

    await asyncio.gather(
        producer(),
        progress_reporter(),
        *[worker() for _ in range(llm_workers)],
    )

    if cache_file:
        cache_file.close()
    await sf_pool.stop()

    n_skipped = sum(1 for r in results if r is None)
    results = [r for r in results if r is not None]
    logger.info("LLM coaching complete: %d kept, %d skipped by LLM", len(results), n_skipped)
    return results


# ---------------------------------------------------------------------------
# Final formatting
# ---------------------------------------------------------------------------


def format_training_sample(aug: AugmentedSample) -> dict:
    """Convert an augmented sample into the training messages format.

    The user message format is identical to what _llm_comment() sends
    at inference time, via the shared format_user_prompt() function.
    """
    board = chess.Board(aug.fen)

    user_prompt = format_user_prompt(
        board_ascii_str=board_ascii(board),
        san=aug.move_san,
        classification=aug.classification,
        eval_str=aug.eval_str,
        best_move=aug.best_move_san,
        cp_loss=aug.cp_loss,
        candidates=aug.candidates,
        opponent_threats=aug.opponent_threats,
        fen=aug.fen,
        cct=aug.cct or {},
        move_tree=aug.move_tree or [],
        position_context=aug.position_context or {},
    )

    # Build coaching text: use provided text; last-resort fallback only
    coaching = aug.coaching_text
    if not coaching:
        # Should rarely happen — use a minimal classification-grounded line
        coaching = f"{aug.move_san} is the {aug.classification.lower()} move in this position."

    assistant = coaching

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            *aug.tool_messages,
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source": aug.source,
            "fen": aug.fen,
            "move_uci": aug.move_uci,
            "classification": aug.classification,
        },
    }


# ---------------------------------------------------------------------------
# Dedup, split, and write
# ---------------------------------------------------------------------------


def dedup_samples(samples: list[AugmentedSample]) -> list[AugmentedSample]:
    """Remove duplicate (fen, move_uci) pairs, keeping the longest coaching text."""
    seen: dict[str, AugmentedSample] = {}
    for s in samples:
        key = f"{s.fen}:{s.move_uci}"
        if key not in seen or len(s.coaching_text) > len(seen[key].coaching_text):
            seen[key] = s
    deduped = list(seen.values())
    logger.info("Dedup: %d → %d samples", len(samples), len(deduped))
    return deduped


def split_and_write(
    samples: list[AugmentedSample],
    output_dir: Path,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[int, int]:
    """Shuffle, split, format, and write train.jsonl + eval.jsonl.

    Returns:
        Tuple of (train_count, eval_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rng.shuffle(samples)

    split_idx = max(1, int(len(samples) * (1 - eval_fraction)))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    for path, subset in [(train_path, train_samples), (eval_path, eval_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for aug in subset:
                sample = format_training_sample(aug)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(
        "Written %d train + %d eval to %s",
        len(train_samples),
        len(eval_samples),
        output_dir,
    )
    return len(train_samples), len(eval_samples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def run_pipeline(
    stockfish_path: str,
    output_dir: Path,
    max_per_source: int = 0,
    depth: int = 16,
    textbook_path: Path | None = None,
    skip_augment: bool = False,
    workers: int = 4,
    llm_coach: bool = False,
    llm_base_url: str = "http://localhost:8100/v1",
    llm_model: str = "Qwen/Qwen3.5-35B-A3B-FP8",
    llm_workers: int = 128,
    only_sources: list[str] | None = None,
) -> None:
    """Run the full extract → augment → format pipeline.

    Args:
        only_sources: If set, only run transformers whose source name is in
            this list. Valid values: chess_cot, icannos, textbook.
            Default (None) runs all available transformers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _want(source: str) -> bool:
        return only_sources is None or source in only_sources

    # Phase 1: Extract
    logger.info("=== Phase 1: Extract ===")
    all_raw: list[RawSample] = []

    transformers: list[BaseTransformer] = []
    if _want("chess_cot"):
        transformers.append(ChessCotTransformer())
    if _want("icannos"):
        transformers.append(IcannosTransformer())
    # jaisonkumar dataset removed — see TODO comment above for context
    if textbook_path and _want("textbook"):
        transformers.append(TextbookTransformer(textbook_path))

    for tx in transformers:
        try:
            samples = list(tx.extract(max_samples=max_per_source))
            all_raw.extend(samples)
        except Exception as e:
            logger.error("Failed to extract from %s: %s", type(tx).__name__, e)

    logger.info("Total extracted: %d raw samples", len(all_raw))

    if not all_raw:
        logger.error("No samples extracted. Check dataset availability.")
        return

    # Phase 2: Augment with Stockfish
    if skip_augment:
        logger.info("=== Phase 2: Skipped (--skip-augment) ===")
        # Create minimal augmented samples without engine data
        augmented = [
            AugmentedSample(
                fen=s.fen,
                move_uci=s.move_uci,
                move_san=s.move_san,
                coaching_text=s.coaching_text,
                thinking_text=s.thinking_text,
                source=s.source,
                classification="Unknown",
                eval_str="0.00",
                best_move_san=s.move_san,
                cp_loss=0,
                candidates=[],
                opponent_threats=[],
            )
            for s in all_raw
        ]
    else:
        logger.info("=== Phase 2: Augment (%d samples) ===", len(all_raw))
        cache_path = output_dir / ".augment_cache.jsonl"
        augmented = await augment_samples(
            all_raw,
            stockfish_path,
            depth=depth,
            cache_path=cache_path,
            workers=workers,
        )

    # Phase 2.5: LLM coaching generation (optional)
    if llm_coach:
        llm_cache = output_dir / ".llm_coaching_cache.jsonl"
        augmented = await generate_coaching_with_llm(
            augmented,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            stockfish_path=stockfish_path,
            cache_path=llm_cache,
            llm_workers=llm_workers,
        )

    # Phase 3: Format and split
    logger.info("=== Phase 3: Format & Split ===")
    deduped = dedup_samples(augmented)
    train_n, eval_n = split_and_write(deduped, output_dir)

    logger.info("=== Done: %d train + %d eval samples ===", train_n, eval_n)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare HuggingFace datasets for chess coaching fine-tuning"
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to Stockfish binary (default: $STOCKFISH_PATH)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for train.jsonl and eval.jsonl",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        help="Max samples per dataset source (0 = unlimited)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=16,
        help="Stockfish analysis depth",
    )
    parser.add_argument(
        "--textbook-path",
        type=Path,
        default=None,
        help="Path to augmented.jsonl from the textbook pipeline",
    )
    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip Stockfish augmentation (for testing format only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel Stockfish workers (default: 4)",
    )
    parser.add_argument(
        "--llm-coach",
        action="store_true",
        help="Use local vLLM to generate coaching text for chess_cot samples",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:8100/v1",
        help="vLLM OpenAI-compatible base URL (default: http://localhost:8100/v1)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen3.5-35B-A3B-FP8",
        help="Model name served by vLLM",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=64,
        help="Number of concurrent LLM request workers (default: 64)",
    )
    parser.add_argument(
        "--only-sources",
        type=str,
        default=None,
        help=(
            "Comma-separated list of sources to process. "
            "Valid: chess_cot,icannos,textbook. "
            "Default: all sources."
        ),
    )

    args = parser.parse_args()

    stockfish_path = args.stockfish or __import__("os").environ.get("STOCKFISH_PATH", "")
    if not stockfish_path and not args.skip_augment:
        parser.error("--stockfish or STOCKFISH_PATH required unless --skip-augment is set")

    only_sources = (
        [s.strip() for s in args.only_sources.split(",") if s.strip()]
        if args.only_sources
        else None
    )

    asyncio.run(
        run_pipeline(
            stockfish_path=stockfish_path,
            output_dir=args.output_dir,
            max_per_source=args.max_per_source,
            depth=args.depth,
            textbook_path=args.textbook_path,
            skip_augment=args.skip_augment,
            workers=args.workers,
            llm_coach=args.llm_coach,
            llm_base_url=args.llm_url,
            llm_model=args.llm_model,
            llm_workers=args.llm_workers,
            only_sources=only_sources,
        )
    )


if __name__ == "__main__":
    main()
