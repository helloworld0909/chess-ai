from .board_tensor import board_to_tensor, boards_to_tensor
from .cnn import ChessEncoder

# Sentinel token injected into the text wherever a chess move appears.
# Uses an existing Qwen3 vocabulary token (fim_pad, ID=151662) that never
# appears naturally in chess text — no tokenizer modification needed.
# The collator replaces this with a CNN board embedding at embedding time;
# the token ID itself is never predicted (label = -100).
MOVE_TOKEN = "<|vision_pad|>"
MOVE_TOKEN_ID = 248055  # Qwen3 vocab ID for <|vision_pad|> — unused in text-only model

__all__ = [
    "board_to_tensor",
    "boards_to_tensor",
    "ChessEncoder",
    "MOVE_TOKEN",
    "MOVE_TOKEN_ID",
]
