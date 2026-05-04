from .board_tensor import board_to_tensor
from .cnn import ChessEncoder

# Sentinel token injected into the text for each board position.
# Uses <|vision_pad|> (ID=248055) — an existing Qwen3 special token that never
# appears naturally in chess text; no tokenizer modification needed.
# The collator replaces each sentinel with a CNN board embedding at embedding time;
# the token ID itself is never predicted (label = -100).
#
# Each board position is represented by BOARD_TOKENS_PER_POSITION (65) consecutive
# sentinel tokens — 64 per-square (row-major a1..h8) + 1 global summary token.
#
# In prompts the sentinel block is wrapped in <|vision_start|>...<|vision_end|>
# (IDs 248053/248054), mirroring Qwen2-VL's image token convention.
BOARD_TOKEN = "<|vision_pad|>"
BOARD_TOKEN_ID = 248055
BOARD_TOKENS_PER_POSITION = 65  # 64 per-square tokens + 1 global summary token (token 64)

VISION_START_TOKEN = "<|vision_start|>"
VISION_START_TOKEN_ID = 248053
VISION_END_TOKEN = "<|vision_end|>"
VISION_END_TOKEN_ID = 248054

# Legacy aliases — kept for imports that haven't been updated yet
MOVE_TOKEN = BOARD_TOKEN
MOVE_TOKEN_ID = BOARD_TOKEN_ID

__all__ = [
    "board_to_tensor",
    "ChessEncoder",
    "BOARD_TOKEN",
    "BOARD_TOKEN_ID",
    "BOARD_TOKENS_PER_POSITION",
    "VISION_START_TOKEN",
    "VISION_START_TOKEN_ID",
    "VISION_END_TOKEN",
    "VISION_END_TOKEN_ID",
    "MOVE_TOKEN",
    "MOVE_TOKEN_ID",
]
