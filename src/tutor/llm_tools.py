"""LLM tool definitions for chess analysis.

Defines OpenAI-compatible function schemas for chess tools
and handlers that call the underlying Stockfish engine.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chess_mcp.stockfish import Stockfish

# OpenAI function calling tool definitions
CHESS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_best_move",
            "description": "Get the best move for a chess position. Returns the engine's recommended move with evaluation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "The chess position in FEN notation (e.g., 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1')",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Analysis depth (higher = stronger but slower). Default is 15.",
                        "default": 15,
                    },
                },
                "required": ["fen"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_eval",
            "description": "Get the evaluation score for a chess position. Returns centipawn score or mate distance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {"type": "string", "description": "The chess position in FEN notation"},
                    "depth": {
                        "type": "integer",
                        "description": "Analysis depth. Default is 15.",
                        "default": 15,
                    },
                },
                "required": ["fen"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_move",
            "description": "Analyze a specific move and classify it as Best/Great/Good/Inaccuracy/Mistake/Blunder. Compares the move against the engine's best move.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "The chess position in FEN notation BEFORE the move is made",
                    },
                    "move": {
                        "type": "string",
                        "description": "The move to analyze in UCI format (e.g., 'e2e4', 'g1f3') or SAN format (e.g., 'e4', 'Nf3')",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Analysis depth. Default is 15.",
                        "default": 15,
                    },
                },
                "required": ["fen", "move"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_legal_moves",
            "description": "Get all legal moves in a position, categorized by type (captures, checks, other).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {"type": "string", "description": "The chess position in FEN notation"}
                },
                "required": ["fen"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_move",
            "description": "Check if a move is legal in a given position. Returns the resulting position if legal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {"type": "string", "description": "The chess position in FEN notation"},
                    "move": {
                        "type": "string",
                        "description": "The move to validate in UCI or SAN format",
                    },
                },
                "required": ["fen", "move"],
            },
        },
    },
]


class ChessToolHandler:
    """Handles tool calls by executing them against Stockfish."""

    def __init__(self, stockfish: Stockfish):
        self.stockfish = stockfish

    async def handle_tool_call(self, name: str, arguments: dict) -> str:
        """Execute a tool call and return the result as JSON string."""
        try:
            if name == "get_best_move":
                return await self._get_best_move(arguments)
            elif name == "get_eval":
                return await self._get_eval(arguments)
            elif name == "analyze_move":
                return await self._analyze_move(arguments)
            elif name == "get_legal_moves":
                return await self._get_legal_moves(arguments)
            elif name == "validate_move":
                return await self._validate_move(arguments)
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _get_best_move(self, args: dict) -> str:
        fen = args["fen"]
        depth = args.get("depth", 15)

        analysis = await self.stockfish.analyze(fen, depth=depth)
        score = analysis.score

        result = {
            "best_move": analysis.best_move,
            "score": str(score),
            "win_probability": f"{analysis.win_probability:.1%}",
            "principal_variation": analysis.pv[:5],
        }

        if score.mate_in is not None:
            result["mate_in"] = score.mate_in

        return json.dumps(result)

    async def _get_eval(self, args: dict) -> str:
        fen = args["fen"]
        depth = args.get("depth", 15)

        line = await self.stockfish.get_eval(fen, depth=depth)
        score = line.score

        result: dict[str, str | int] = {
            "score": str(score),
            "win_probability": f"{line.win_probability:.1%}",
        }

        if score.centipawns is not None:
            result["centipawns"] = score.centipawns
        if score.mate_in is not None:
            result["mate_in"] = score.mate_in

        return json.dumps(result)

    async def _analyze_move(self, args: dict) -> str:
        fen = args["fen"]
        move = args["move"]
        depth = args.get("depth", 15)

        comparison = await self.stockfish.compare_moves(fen, move, depth=depth)

        if "error" in comparison:
            return json.dumps({"error": comparison["error"]})

        return json.dumps(
            {
                "move": move,
                "classification": comparison["classification"],
                "is_best": comparison["is_best"],
                "centipawn_loss": comparison["cp_loss"],
                "best_move": comparison["best_move"],
                "best_move_score": str(comparison.get("best_score", "")),
            }
        )

    async def _get_legal_moves(self, args: dict) -> str:
        import chess

        fen = args["fen"]
        board = chess.Board(fen)

        moves = {
            "total": len(list(board.legal_moves)),
            "captures": [],
            "checks": [],
            "other": [],
        }

        for move in board.legal_moves:
            san = board.san(move)
            if board.is_capture(move):
                moves["captures"].append(san)
            elif board.gives_check(move):
                moves["checks"].append(san)
            else:
                moves["other"].append(san)

        return json.dumps(moves)

    async def _validate_move(self, args: dict) -> str:
        from verification.legality import parse_move_flexible

        fen = args["fen"]
        move = args["move"]

        result = parse_move_flexible(fen, move)

        if not result.valid:
            return json.dumps(
                {
                    "legal": False,
                    "error": result.error,
                }
            )

        return json.dumps(
            {
                "legal": True,
                "move_uci": result.move_uci,
                "move_san": result.move_san,
                "resulting_fen": result.resulting_fen,
                "is_check": result.is_check,
                "is_checkmate": result.is_checkmate,
                "is_capture": result.is_capture,
            }
        )
