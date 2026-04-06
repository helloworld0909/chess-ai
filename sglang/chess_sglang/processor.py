"""ChessBoardProcessor — SGLang multimodal processor for chess board positions.

Runs in the tokenizer/processor worker (CPU subprocess). Converts FEN strings
(or pre-built board tensors) received via extra_body into raw (19, 8, 8) board
tensors stored as MultimodalDataItem.feature. The CNN forward that produces
the actual (65, 2560) embeddings runs inside ChessQwen3ForCausalLM on the GPU.

Request format (via openai client extra_body):
    # Single board (initial position):
    extra_body={"image_data": [{"format": "board_tensor", "fen": "rnbq..."}]}

    # Multiple boards (initial + tool-call boards):
    extra_body={"image_data": [
        {"format": "board_tensor", "fen": "rnbq..."},   # board 0
        {"format": "board_tensor", "fen": "rnbq..."},   # board 1 (after tool call)
    ]}

The prompt text must already contain the correct number of <|vision_pad|> ×65
sentinel blocks — one block per entry in image_data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch

# Resolve chess-ai/src
_CHESS_AI_SRC = Path(__file__).parent.parent.parent / "src"
if str(_CHESS_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_CHESS_AI_SRC))

import chess  # noqa: E402
from encoder.board_tensor import board_to_tensor  # noqa: E402

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
)
from sglang.srt.multimodal.processors.base_processor import (  # noqa: E402
    BaseMultiModalProcessorOutput,
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

BOARD_TOKEN_ID = 248055       # <|vision_pad|>
BOARD_TOKEN_STR = "<|vision_pad|>"
BOARD_TOKENS_PER_POSITION = 65


class ChessBoardProcessor(BaseMultimodalProcessor):
    """Processor that converts FEN strings to raw board tensors.

    The tensors are stored in MultimodalDataItem.feature and passed to the
    model worker where the CNN runs on GPU.
    """

    models = []  # populated after ChessQwen3ForCausalLM is defined; set in __init__.py

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=BOARD_TOKEN_STR,
            image_token_id=BOARD_TOKEN_ID,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, dict]],
        input_text: str,
        *args,
        **kwargs,
    ) -> dict:
        """Convert list of board dicts into MultimodalDataItems.

        Each dict in image_data must have:
            {"format": "board_tensor", "fen": "<FEN string>"}

        Returns a dict with keys "input_ids" and "mm_items" as expected by
        SGLang's tokenizer manager.
        """
        if not image_data:
            # No boards — text-only fallback
            input_ids = self._processor.tokenizer(
                input_text, return_tensors="pt", add_special_tokens=True
            ).input_ids.flatten()
            return {"input_ids": input_ids.tolist(), "mm_items": []}

        mm_items: List[MultimodalDataItem] = []

        for board_dict in image_data:
            if not isinstance(board_dict, dict):
                raise ValueError(
                    f"ChessBoardProcessor expects dicts with 'fen' key, got {type(board_dict)}"
                )

            fmt = board_dict.get("format", "board_tensor")
            if fmt not in ("board_tensor", "precomputed_embedding"):
                raise ValueError(f"Unknown board format: {fmt!r}")

            fen = board_dict.get("fen")
            if fen is None:
                raise ValueError("Board dict must contain 'fen' key")

            try:
                board = chess.Board(fen)
            except Exception as e:
                raise ValueError(f"Invalid FEN {fen!r}: {e}")

            # Build raw board tensor on CPU — CNN runs on GPU inside the model
            tensor = board_to_tensor(board)  # (19, 8, 8) float32

            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                format=MultimodalInputFormat.NORMAL,  # feature path, not precomputed
                feature=tensor,                        # (19, 8, 8), passed to _cnn_embed
            )
            item.set_pad_value()
            mm_items.append(item)

        # Tokenize the prompt — it already has the right number of sentinel blocks
        input_ids = self._processor.tokenizer(
            input_text, return_tensors="pt", add_special_tokens=True
        ).input_ids.flatten()

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
        }
