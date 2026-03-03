"""Async Stockfish wrapper for UCI communication."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator


class ScoreType(Enum):
    CENTIPAWNS = "cp"
    MATE = "mate"


@dataclass
class Score:
    """Stockfish evaluation score."""

    type: ScoreType
    value: int

    @property
    def centipawns(self) -> int | None:
        """Return score in centipawns, or None if mate."""
        return self.value if self.type == ScoreType.CENTIPAWNS else None

    @property
    def mate_in(self) -> int | None:
        """Return moves to mate, or None if not mate."""
        return self.value if self.type == ScoreType.MATE else None

    @property
    def win_probability(self) -> float:
        """Convert centipawn score to win probability (0-1)."""
        if self.type == ScoreType.MATE:
            return 1.0 if self.value > 0 else 0.0
        # Lichess formula: win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)
        import math

        return 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * self.value)) - 1)

    def __str__(self) -> str:
        if self.type == ScoreType.MATE:
            return f"M{self.value}" if self.value > 0 else f"M{self.value}"
        return f"{self.value / 100:+.2f}"


@dataclass
class AnalysisLine:
    """A single line of analysis (PV)."""

    depth: int
    seldepth: int
    score: Score
    nodes: int
    nps: int
    time_ms: int
    pv: list[str]  # Principal variation (list of moves in UCI notation)
    multipv: int = 1
    wdl: tuple[int, int, int] | None = None  # Win/Draw/Loss in permille from Stockfish

    @property
    def best_move(self) -> str:
        """Return the best move from this line."""
        return self.pv[0] if self.pv else ""

    @property
    def win_probability(self) -> float:
        """Win probability using native WDL if available, else Score sigmoid fallback."""
        if self.wdl is not None:
            return self.wdl[0] / 1000.0
        return self.score.win_probability


@dataclass
class Analysis:
    """Complete analysis result."""

    fen: str
    depth: int
    lines: list[AnalysisLine]
    best_move: str

    @property
    def score(self) -> Score:
        """Return the score of the best line."""
        return self.lines[0].score if self.lines else Score(ScoreType.CENTIPAWNS, 0)

    @property
    def win_probability(self) -> float:
        """Win probability using native WDL if available, else sigmoid fallback."""
        return self.lines[0].win_probability if self.lines else 0.5

    @property
    def pv(self) -> list[str]:
        """Return the principal variation of the best line."""
        return self.lines[0].pv if self.lines else []


class StockfishError(Exception):
    """Stockfish-related errors."""

    pass


class Stockfish:
    """Async wrapper for Stockfish UCI engine."""

    def __init__(
        self,
        path: str | None = None,
        depth: int = 20,
        threads: int = 4,
        hash_mb: int = 256,
    ):
        self.path = path or os.environ.get("STOCKFISH_PATH", "stockfish")
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._ready = False

    async def start(self) -> None:
        """Start the Stockfish process."""
        if self._process is not None:
            return

        try:
            self._process = await asyncio.create_subprocess_exec(
                self.path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise StockfishError(f"Stockfish not found at: {self.path}")

        # Initialize UCI
        await self._send("uci")
        await self._wait_for("uciok")

        # Configure options
        await self._send(f"setoption name Threads value {self.threads}")
        await self._send(f"setoption name Hash value {self.hash_mb}")
        await self._send("setoption name UCI_AnalyseMode value true")
        await self._send("setoption name UCI_ShowWDL value true")

        await self._send("isready")
        await self._wait_for("readyok")
        self._ready = True

    async def stop(self) -> None:
        """Stop the Stockfish process."""
        if self._process is None:
            return

        await self._send("quit")
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._process.kill()

        self._process = None
        self._ready = False

    async def _send(self, command: str) -> None:
        """Send a command to Stockfish."""
        if self._process is None or self._process.stdin is None:
            raise StockfishError("Stockfish not started")

        self._process.stdin.write(f"{command}\n".encode())
        await self._process.stdin.drain()

    async def _readline(self) -> str:
        """Read a line from Stockfish output."""
        if self._process is None or self._process.stdout is None:
            raise StockfishError("Stockfish not started")

        line = await self._process.stdout.readline()
        if not line:
            raise StockfishError("Stockfish process died (EOF)")
        return line.decode().strip()

    async def _wait_for(self, expected: str) -> str:
        """Wait for a specific response."""
        while True:
            line = await self._readline()
            if line.startswith(expected):
                return line

    async def _read_until(self, expected: str) -> AsyncIterator[str]:
        """Read lines until expected string."""
        while True:
            line = await self._readline()
            yield line
            if line.startswith(expected):
                break

    def _parse_info_line(self, line: str) -> AnalysisLine | None:
        """Parse an 'info' line from Stockfish output."""
        if not line.startswith("info ") or " pv " not in line:
            return None

        parts = line.split()

        def get_value(key: str) -> int:
            try:
                idx = parts.index(key)
                return int(parts[idx + 1])
            except (ValueError, IndexError):
                return 0

        # Parse score
        score_type = ScoreType.CENTIPAWNS
        score_value = 0
        try:
            score_idx = parts.index("score")
            if parts[score_idx + 1] == "cp":
                score_type = ScoreType.CENTIPAWNS
                score_value = int(parts[score_idx + 2])
            elif parts[score_idx + 1] == "mate":
                score_type = ScoreType.MATE
                score_value = int(parts[score_idx + 2])
        except (ValueError, IndexError):
            pass

        # Parse WDL (Win/Draw/Loss in permille, e.g. "wdl 140 859 1")
        wdl: tuple[int, int, int] | None = None
        try:
            wdl_idx = parts.index("wdl")
            wdl = (int(parts[wdl_idx + 1]), int(parts[wdl_idx + 2]), int(parts[wdl_idx + 3]))
        except (ValueError, IndexError):
            pass

        # Parse PV
        pv: list[str] = []
        try:
            pv_idx = parts.index("pv")
            pv = parts[pv_idx + 1 :]
        except ValueError:
            pass

        return AnalysisLine(
            depth=get_value("depth"),
            seldepth=get_value("seldepth"),
            score=Score(score_type, score_value),
            nodes=get_value("nodes"),
            nps=get_value("nps"),
            time_ms=get_value("time"),
            pv=pv,
            multipv=get_value("multipv") or 1,
            wdl=wdl,
        )

    async def analyze(
        self,
        fen: str,
        depth: int | None = None,
        multipv: int = 1,
    ) -> Analysis:
        """Analyze a position and return the result."""
        async with self._lock:
            if not self._ready:
                await self.start()

            depth = depth or self.depth

            # Set up position and multi-PV
            await self._send(f"setoption name MultiPV value {multipv}")
            await self._send("ucinewgame")
            await self._send("isready")
            await self._wait_for("readyok")
            await self._send(f"position fen {fen}")
            await self._send(f"go depth {depth}")

            # Collect analysis lines
            lines: dict[int, AnalysisLine] = {}
            best_move = ""

            async for line in self._read_until("bestmove"):
                if line.startswith("info "):
                    parsed = self._parse_info_line(line)
                    if parsed:
                        # Keep the deepest line for each multipv index
                        if (
                            parsed.multipv not in lines
                            or parsed.depth > lines[parsed.multipv].depth
                        ):
                            lines[parsed.multipv] = parsed
                elif line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        best_move = parts[1]

            # Sort lines by multipv index
            sorted_lines = [lines[k] for k in sorted(lines.keys())]

            return Analysis(
                fen=fen,
                depth=depth,
                lines=sorted_lines,
                best_move=best_move,
            )

    async def get_best_move(self, fen: str, depth: int | None = None) -> str:
        """Get the best move for a position."""
        analysis = await self.analyze(fen, depth)
        return analysis.best_move

    async def get_eval(self, fen: str, depth: int | None = None) -> AnalysisLine:
        """Get the evaluation of a position (includes native WDL when available)."""
        analysis = await self.analyze(fen, depth)
        if analysis.lines:
            return analysis.lines[0]
        # Fallback: empty line with zero score
        return AnalysisLine(
            depth=0,
            seldepth=0,
            score=Score(ScoreType.CENTIPAWNS, 0),
            nodes=0,
            nps=0,
            time_ms=0,
            pv=[],
        )

    async def compare_moves(
        self,
        fen: str,
        user_move: str,
        depth: int | None = None,
    ) -> dict:
        """Compare a user's move against the engine's best move."""
        import chess

        board = chess.Board(fen)

        # Validate user move
        try:
            move = chess.Move.from_uci(user_move)
            if move not in board.legal_moves:
                return {"error": f"Illegal move: {user_move}"}
        except ValueError:
            return {"error": f"Invalid move format: {user_move}"}

        # Get analysis of current position
        analysis = await self.analyze(fen, depth, multipv=3)
        best_move = analysis.best_move
        best_score = analysis.score
        wdl_before = analysis.lines[0].wdl if analysis.lines else None

        # Apply user move and get resulting position eval
        board.push(move)
        user_analysis = await self.analyze(board.fen(), depth)

        # Score is from opponent's perspective, so negate it
        user_score = Score(
            user_analysis.score.type,
            -user_analysis.score.value,
        )
        # WDL is also from opponent's perspective — swap W and L
        raw_wdl = user_analysis.lines[0].wdl if user_analysis.lines else None
        wdl_after = (raw_wdl[2], raw_wdl[1], raw_wdl[0]) if raw_wdl is not None else None

        # Calculate centipawn loss
        cp_loss = 0
        if best_score.centipawns is not None and user_score.centipawns is not None:
            cp_loss = best_score.centipawns - user_score.centipawns

        # Win probability loss (WDL-based when available, else sigmoid)
        wp_before = wdl_before[0] / 1000.0 if wdl_before is not None else best_score.win_probability
        wp_after = wdl_after[0] / 1000.0 if wdl_after is not None else user_score.win_probability
        wp_loss = round(wp_before - wp_after, 4)

        # Classify move
        if user_move == best_move:
            classification = "Best"
        elif cp_loss <= 10:
            classification = "Great"
        elif cp_loss <= 30:
            classification = "Good"
        elif cp_loss <= 80:
            classification = "Inaccuracy"
        elif cp_loss <= 150:
            classification = "Mistake"
        else:
            classification = "Blunder"

        return {
            "user_move": user_move,
            "best_move": best_move,
            "user_score": str(user_score),
            "best_score": str(best_score),
            "cp_loss": cp_loss,
            "wp_loss": wp_loss,
            "wdl_before": wdl_before,
            "wdl_after": wdl_after,
            "classification": classification,
            "pv": analysis.pv[:5],  # Top 5 moves of best line
            "is_best": user_move == best_move,
        }

    async def get_threats(self, fen: str, depth: int | None = None) -> dict:
        """Identify tactical threats in the position."""
        import chess

        board = chess.Board(fen)

        # Analyze from opponent's perspective (null move)
        if board.is_valid() and not board.is_game_over():
            # Get opponent's best response by switching sides
            board.push(chess.Move.null())
            if board.is_valid():
                opponent_analysis = await self.analyze(board.fen(), depth or 15, multipv=3)
                board.pop()

                threats = []
                for line in opponent_analysis.lines:
                    if line.pv:
                        threat_move = line.pv[0]
                        # Check if this is a significant threat
                        if line.score.centipawns is not None and line.score.centipawns > 50:
                            threats.append(
                                {
                                    "move": threat_move,
                                    "score": str(line.score),
                                    "type": self._classify_threat(board, threat_move),
                                }
                            )

                return {"fen": fen, "threats": threats}

        return {"fen": fen, "threats": []}

    def _classify_threat(self, board: chess.Board, move_uci: str) -> str:
        """Classify the type of threat."""
        import chess

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return "unknown"

        # Check various threat types
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                return f"capture_{chess.piece_name(captured.piece_type)}"

        board_copy = board.copy()
        board_copy.push(move)

        if board_copy.is_checkmate():
            return "checkmate"
        if board_copy.is_check():
            return "check"

        # Check for fork potential (simplified)
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KNIGHT:
            attacks = board_copy.attacks(move.to_square)
            valuable_targets = sum(
                1
                for sq in attacks
                if board_copy.piece_at(sq)
                and board_copy.piece_at(sq).color != piece.color
                and board_copy.piece_at(sq).piece_type in [chess.QUEEN, chess.ROOK, chess.KING]
            )
            if valuable_targets >= 2:
                return "fork"

        return "positional"

    async def __aenter__(self) -> "Stockfish":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.stop()
