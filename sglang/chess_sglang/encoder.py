"""Re-export chess encoder components from chess-ai/src.

This avoids code duplication — the SGLang package shares the same CNN
implementation used during training.

Adds chess-ai/src to sys.path so imports resolve correctly regardless of
where SGLang is launched from.
"""

from __future__ import annotations

import sys
from pathlib import Path

# chess-ai/src is two levels up from this file (sglang/chess_sglang/encoder.py)
_CHESS_AI_SRC = Path(__file__).parent.parent.parent / "src"
if str(_CHESS_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_CHESS_AI_SRC))

from encoder import BOARD_TOKEN, BOARD_TOKEN_ID, BOARD_TOKENS_PER_POSITION  # noqa: F401
from encoder.board_tensor import board_to_tensor  # noqa: F401
from encoder.cnn import ChessEncoder  # noqa: F401

__all__ = [
    "ChessEncoder",
    "board_to_tensor",
    "BOARD_TOKEN",
    "BOARD_TOKEN_ID",
    "BOARD_TOKENS_PER_POSITION",
]
