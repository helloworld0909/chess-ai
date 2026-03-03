"""MCP Tool definitions for chess analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from .stockfish import Stockfish


@dataclass
class ToolResult:
    """Result from a tool invocation."""

    success: bool
    data: dict[str, Any]
    error: str | None = None


class ChessTools:
    """Collection of chess analysis tools for MCP."""

    def __init__(self, stockfish: Stockfish):
        self.stockfish = stockfish

    async def get_best_move(self, fen: str, depth: int = 20) -> ToolResult:
        """Get the best move for a position.

        Args:
            fen: Position in FEN notation
            depth: Search depth (default 20)

        Returns:
            Best move, score, and principal variation
        """
        try:
            # Validate FEN
            try:
                chess.Board(fen)
            except ValueError as e:
                return ToolResult(success=False, data={}, error=f"Invalid FEN: {e}")

            analysis = await self.stockfish.analyze(fen, depth, multipv=1)

            return ToolResult(
                success=True,
                data={
                    "best_move": analysis.best_move,
                    "score": str(analysis.score),
                    "score_type": analysis.score.type.value,
                    "score_value": analysis.score.value,
                    "win_probability": round(analysis.win_probability, 3),
                    "depth": analysis.depth,
                    "pv": analysis.pv[:10],
                },
            )
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))

    async def get_eval(self, fen: str, depth: int = 20) -> ToolResult:
        """Get the evaluation of a position.

        Args:
            fen: Position in FEN notation
            depth: Search depth (default 20)

        Returns:
            Centipawn score, mate distance, and win probability
        """
        try:
            try:
                chess.Board(fen)
            except ValueError as e:
                return ToolResult(success=False, data={}, error=f"Invalid FEN: {e}")

            line = await self.stockfish.get_eval(fen, depth)
            score = line.score

            result_data = {
                "score": str(score),
                "win_probability": round(line.win_probability, 3),
            }

            if score.centipawns is not None:
                result_data["centipawns"] = score.centipawns
                result_data["pawns"] = round(score.centipawns / 100, 2)
            if score.mate_in is not None:
                result_data["mate_in"] = score.mate_in

            return ToolResult(success=True, data=result_data)
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))

    async def get_threats(self, fen: str, depth: int = 15) -> ToolResult:
        """Identify tactical threats in a position.

        Args:
            fen: Position in FEN notation
            depth: Search depth (default 15)

        Returns:
            List of tactical threats
        """
        try:
            try:
                board = chess.Board(fen)
            except ValueError as e:
                return ToolResult(success=False, data={}, error=f"Invalid FEN: {e}")

            # Check for immediate threats
            threats_data = await self.stockfish.get_threats(fen, depth)

            # Add check detection
            in_check = board.is_check()
            checkers = []
            if in_check:
                for sq in board.checkers():
                    piece = board.piece_at(sq)
                    if piece:
                        checkers.append(
                            {
                                "square": chess.square_name(sq),
                                "piece": piece.symbol(),
                            }
                        )

            # Detect hanging pieces (pieces that can be captured for free)
            hanging = []
            for move in board.legal_moves:
                if board.is_capture(move):
                    captured = board.piece_at(move.to_square)
                    attacker = board.piece_at(move.from_square)
                    if captured and attacker:
                        # Check if capture is "free" (no recapture or winning exchange)
                        captured_value = self._piece_value(captured.piece_type)
                        attacker_value = self._piece_value(attacker.piece_type)
                        if captured_value >= attacker_value:
                            hanging.append(
                                {
                                    "square": chess.square_name(move.to_square),
                                    "piece": captured.symbol(),
                                    "attacker": chess.square_name(move.from_square),
                                }
                            )

            return ToolResult(
                success=True,
                data={
                    "in_check": in_check,
                    "checkers": checkers,
                    "threats": threats_data.get("threats", []),
                    "hanging_pieces": hanging[:3],  # Limit to top 3
                },
            )
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))

    async def compare_moves(self, fen: str, move: str, depth: int = 20) -> ToolResult:
        """Compare a user's move against the engine's best move.

        Args:
            fen: Position in FEN notation
            move: User's move in UCI notation (e.g., "e2e4")
            depth: Search depth (default 20)

        Returns:
            Comparison including classification, centipawn loss, and refutation
        """
        try:
            try:
                board = chess.Board(fen)
            except ValueError as e:
                return ToolResult(success=False, data={}, error=f"Invalid FEN: {e}")

            # Validate move
            try:
                chess_move = chess.Move.from_uci(move)
                if chess_move not in board.legal_moves:
                    return ToolResult(
                        success=False,
                        data={},
                        error=f"Illegal move: {move}",
                    )
            except ValueError:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Invalid move format: {move}. Use UCI notation (e.g., e2e4)",
                )

            comparison = await self.stockfish.compare_moves(fen, move, depth)

            if "error" in comparison:
                return ToolResult(success=False, data={}, error=comparison["error"])

            return ToolResult(
                success=True,
                data={
                    "user_move": comparison["user_move"],
                    "best_move": comparison["best_move"],
                    "is_best": comparison["is_best"],
                    "classification": comparison["classification"],
                    "cp_loss": comparison["cp_loss"],
                    "user_score": comparison["user_score"],
                    "best_score": comparison["best_score"],
                    "refutation_line": comparison["pv"],
                },
            )
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))

    async def get_legal_moves(self, fen: str) -> ToolResult:
        """Get all legal moves for a position.

        Args:
            fen: Position in FEN notation

        Returns:
            List of legal moves in UCI notation
        """
        try:
            try:
                board = chess.Board(fen)
            except ValueError as e:
                return ToolResult(success=False, data={}, error=f"Invalid FEN: {e}")

            moves = [move.uci() for move in board.legal_moves]

            # Categorize moves
            captures = []
            checks = []
            regular = []

            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                if board.is_capture(move):
                    captures.append(move_uci)
                elif board.gives_check(move):
                    checks.append(move_uci)
                else:
                    regular.append(move_uci)

            return ToolResult(
                success=True,
                data={
                    "total": len(moves),
                    "captures": captures,
                    "checks": checks,
                    "other": regular,
                },
            )
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))

    async def validate_move(self, fen: str, move: str) -> ToolResult:
        """Validate if a move is legal.

        Args:
            fen: Position in FEN notation
            move: Move in UCI notation

        Returns:
            Whether move is legal and resulting position
        """
        try:
            try:
                board = chess.Board(fen)
            except ValueError as e:
                return ToolResult(success=False, data={}, error=f"Invalid FEN: {e}")

            try:
                chess_move = chess.Move.from_uci(move)
            except ValueError:
                return ToolResult(
                    success=True,
                    data={
                        "legal": False,
                        "reason": "Invalid move format",
                    },
                )

            if chess_move not in board.legal_moves:
                return ToolResult(
                    success=True,
                    data={
                        "legal": False,
                        "reason": "Move is not legal in this position",
                    },
                )

            # Apply move
            san = board.san(chess_move)
            board.push(chess_move)

            return ToolResult(
                success=True,
                data={
                    "legal": True,
                    "san": san,
                    "resulting_fen": board.fen(),
                    "is_check": board.is_check(),
                    "is_checkmate": board.is_checkmate(),
                    "is_stalemate": board.is_stalemate(),
                },
            )
        except Exception as e:
            return ToolResult(success=False, data={}, error=str(e))

    def _piece_value(self, piece_type: chess.PieceType) -> int:
        """Get approximate piece value."""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }
        return values.get(piece_type, 0)
