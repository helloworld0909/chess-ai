from .board_tensor import board_to_tensor
from .cnn import ChessEncoder

# Sentinel token injected into the text for each board position.
# Uses an existing Qwen3 vocabulary token (<|vision_pad|>, ID=248055) that never
# appears naturally in chess text — no tokenizer modification needed.
# The collator replaces each sentinel with a CNN board embedding at embedding time;
# the token ID itself is never predicted (label = -100).
#
# Each board position is represented by BOARD_TOKENS_PER_POSITION (65) consecutive
# sentinel tokens — 64 per-square (row-major a1..h8) + 1 global summary token.
BOARD_TOKEN = "<|vision_pad|>"
BOARD_TOKEN_ID = 248055  # Qwen3 vocab ID for <|vision_pad|> — unused in text-only model
BOARD_TOKENS_PER_POSITION = 65  # 64 per-square tokens + 1 global summary token (token 64)

# Legacy aliases — kept for imports that haven't been updated yet
MOVE_TOKEN = BOARD_TOKEN
MOVE_TOKEN_ID = BOARD_TOKEN_ID

__all__ = [
    "board_to_tensor",
    "ChessEncoder",
    "BOARD_TOKEN",
    "BOARD_TOKEN_ID",
    "BOARD_TOKENS_PER_POSITION",
    "MOVE_TOKEN",
    "MOVE_TOKEN_ID",
]
