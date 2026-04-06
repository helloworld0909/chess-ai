"""ChessLMConfig — Qwen3Config subclass with chess encoder fields.

Saved as config.json alongside the merged weights so SGLang auto-detects
ChessQwen3ForCausalLM via the "architectures" field.
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class ChessLMConfig(PretrainedConfig):
    """Configuration for ChessQwen3ForCausalLM.

    Inherits all Qwen3.5-4B fields from the base config dict; adds chess-specific
    fields read by ChessQwen3ForCausalLM and ChessBoardProcessor.
    """

    model_type = "chess_qwen3"

    def __init__(
        self,
        board_token_id: int = 248055,
        board_tokens_per_position: int = 65,
        encoder_in_channels: int = 19,
        encoder_hidden_size: int = 256,
        encoder_num_blocks: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.board_token_id = board_token_id
        self.board_tokens_per_position = board_tokens_per_position
        self.encoder_in_channels = encoder_in_channels
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_blocks = encoder_num_blocks


# Register so AutoConfig.from_pretrained works with model_type="chess_qwen3"
AutoConfig.register("chess_qwen3", ChessLMConfig)
