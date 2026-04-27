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

The prompt text must contain one <|vision_pad|> placeholder per board entry in
image_data. The scheduler later expands each placeholder into the 65 internal
slots needed for the CNN output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Union

import torch

# Resolve chess-ai/src
_CHESS_AI_SRC = Path(__file__).parent.parent.parent / "src"
if str(_CHESS_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_CHESS_AI_SRC))

import chess  # noqa: E402

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
)
from sglang.srt.multimodal.processors.base_processor import (  # noqa: E402
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
        tokenizer = getattr(_processor, "tokenizer", _processor)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=BOARD_TOKEN_STR,
            image_token_id=BOARD_TOKEN_ID,
        ).build(tokenizer)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, dict]],
        audio_data,
        input_text: str,
        request_obj,
        *args,
        **kwargs,
    ) -> dict:
        """Convert list of board dicts into MultimodalDataItems.

        Each dict in image_data must have:
            {"format": "board_tensor", "fen": "<FEN string>"}

        Returns a dict with keys "input_ids" and "mm_items" as expected by
        SGLang's tokenizer manager.
        """
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        if not image_data:
            # No boards — text-only fallback
            input_ids = tokenizer(
                input_text, return_tensors="pt", add_special_tokens=True
            ).input_ids.flatten()
            return {"input_ids": input_ids.tolist(), "mm_items": []}

        mm_items: List[MultimodalDataItem] = []

        for board_dict in image_data:
            if not isinstance(board_dict, dict):
                continue

            fmt = board_dict.get("format", "board_tensor")
            if fmt not in ("board_tensor", "precomputed_embedding"):
                continue

            fen = board_dict.get("fen")
            if fen is None:
                continue

            try:
                chess.Board(fen)  # validate FEN
            except Exception as e:
                raise ValueError(f"Invalid FEN {fen!r}: {e}")

            # Store FEN as bytes — the model builds the board tensor on GPU.
            # Avoids SGLang's ShmPointerMMData IPC path (only triggered for CPU tensors).
            # Bytes are hashable by SGLang's data_hash (sha256 requires bytes-like).
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                format=MultimodalInputFormat.NORMAL,
                feature=fen.encode(),  # bytes — no tensor, no shm wrapping
            )
            item.set_pad_value()
            mm_items.append(item)

        if not mm_items:
            input_ids = tokenizer(
                input_text, return_tensors="pt", add_special_tokens=True
            ).input_ids.flatten()
            return {"input_ids": input_ids.tolist(), "mm_items": []}

        # Tokenize the prompt. Each board should correspond to one placeholder
        # token; the scheduler later expands that single token to 65 pad slots.
        input_ids = tokenizer(
            input_text, return_tensors="pt", add_special_tokens=True
        ).input_ids.flatten()

        offsets = self.get_mm_items_offset(input_ids, BOARD_TOKEN_ID)
        if len(offsets) != len(mm_items):
            raise ValueError(
                "Board placeholder count does not match image_data count: "
                f"{len(offsets)} placeholders vs {len(mm_items)} boards"
            )

        for item, (start, end) in zip(mm_items, offsets):
            if end - start + 1 != 1:
                raise ValueError(
                    "Each board must occupy exactly one "
                    f"{BOARD_TOKEN_STR} placeholder token; "
                    f"got span length {end - start + 1}"
                )
            item.offsets = [(start, end + 1)]

        expanded_input_len = len(input_ids) + len(mm_items) * (BOARD_TOKENS_PER_POSITION - 1)
        mrope_positions = torch.arange(expanded_input_len, dtype=torch.int64).unsqueeze(0).repeat(3, 1)
        mrope_position_delta = torch.zeros((1, 1), dtype=torch.int64)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }
