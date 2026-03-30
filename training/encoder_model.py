"""Wrapper module coupling Qwen3.5-4B with the ChessEncoder trunk."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from src.encoder import MOVE_TOKEN_ID
from src.encoder.cnn import ChessEncoder

# Number of sentinel tokens injected per board position.
# Must match the number of <|vision_pad|> tokens in the user message and the
# CNN output sequence length (64 = one token per square).
BOARD_TOKENS_PER_POSITION = 64


class ChessLMWithEncoder(nn.Module):
    """Combines a base LLM (e.g., Qwen3.5-4B) with a ResNet board encoder.

    Each board position is represented by BOARD_TOKENS_PER_POSITION (64) consecutive
    <|vision_pad|> sentinel tokens in the input. The encoder processes the 38-channel
    8x8 board tensor and produces 64 per-square embeddings (one per chess square,
    row-major a1..h8), which replace the 64 sentinels in the embedding sequence.

    Each per-square token encodes the full board context from that square's perspective
    (receptive field covers the entire board via deep ResNet convolutions), plus a
    2D learnable positional encoding distinguishing file and rank.

    Usage:
        # collator provides board_tensors_flat (N_boards, 38, 8, 8) and
        # move_counts (B,) alongside input_ids. board_tensors_flat/move_counts
        # are optional — omit them (or pass None) for TRL log-prob passes on
        # completion-only sequences; zero embeddings will be used instead.
        out = model(
            input_ids=ids,             # (B, L)  — L includes 64 sentinels per board
            board_tensors_flat=flat,   # (N_boards, 38, 8, 8)  — optional
            move_counts=counts,        # (B,)  — optional, kept for API compat
            attention_mask=mask,       # (B, L)
            labels=labels,             # (B, L)
        )
    """

    def __init__(
        self,
        llm: nn.Module,
        hidden_size: int = 2560,
        cnn_in_channels: int = 19,
        cnn_hidden_size: int = 256,
        cnn_num_blocks: int = 10,
        move_token_id: int = MOVE_TOKEN_ID,
    ):
        super().__init__()
        self.llm = llm
        self.move_token_id = move_token_id
        self.cnn = ChessEncoder(
            in_channels=cnn_in_channels,
            hidden_size=cnn_hidden_size,
            num_blocks=cnn_num_blocks,
            out_dim=hidden_size,
        )

        if hasattr(self.llm, "get_input_embeddings"):
            self.embed_tokens = self.llm.get_input_embeddings()
        elif hasattr(self.llm.model, "embed_tokens"):
            self.embed_tokens = self.llm.model.embed_tokens
        else:
            raise ValueError("Could not find input embeddings layer in the LLM.")

        # Expose HF model attributes that SFTTrainer / Trainer expect
        self.config = self.llm.config
        self.name_or_path = getattr(self.llm, "name_or_path", "")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Proxy to inner LLM for Trainer compatibility."""
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is not None:
                self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                self.llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Proxy to inner LLM for Trainer compatibility."""
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.LongTensor,
        board_tensors_flat: Optional[torch.Tensor] = None,
        move_counts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """Forward pass — splices 64 CNN embeddings into each sentinel group.

        Each board position occupies exactly BOARD_TOKENS_PER_POSITION (64) consecutive
        sentinel tokens. The CNN encodes the board tensor into (64, H) per-square
        embeddings, which replace those sentinels in the embedding sequence.

        Args:
            input_ids: (B, L) — L includes 64 sentinels per board position.
            board_tensors_flat: (N_boards, 38, 8, 8) — one tensor per board position
                (not per token). If None, zero embeddings are used for all sentinels.
            move_counts: (B,) number of board positions per example. Unused in forward;
                kept for API compatibility.
            attention_mask: (B, L)
            labels: (B, L) — sentinel positions will be set to -100.
        """
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype
        B, L = input_ids.shape

        # 1. Text embeddings: (B, L, H)
        inputs_embeds = self.embed_tokens(input_ids)

        # 2. Find all sentinel positions
        move_mask = input_ids == self.move_token_id
        n_sentinels = move_mask.sum().item()

        if n_sentinels > 0:
            # 3. N_boards = total sentinel tokens / tokens per board
            H = inputs_embeds.shape[-1]
            n_boards = n_sentinels // BOARD_TOKENS_PER_POSITION

            if board_tensors_flat is not None and board_tensors_flat.shape[0] > 0:
                cnn_dtype = next(self.cnn.parameters()).dtype
                # cnn_out: (N_boards, 64, H)
                cnn_out = self.cnn(board_tensors_flat.to(device=device, dtype=cnn_dtype))
                cnn_out = cnn_out.to(dtype=dtype)

                # Pad/trim to exactly n_boards boards
                if cnn_out.shape[0] < n_boards:
                    pad = torch.zeros(
                        n_boards - cnn_out.shape[0],
                        BOARD_TOKENS_PER_POSITION,
                        H,
                        dtype=dtype,
                        device=device,
                    )
                    cnn_out = torch.cat([cnn_out, pad], dim=0)
                else:
                    cnn_out = cnn_out[:n_boards]

                # Flatten to (n_sentinels, H) — matches sentinel positions in order
                cnn_embs = cnn_out.reshape(n_boards * BOARD_TOKENS_PER_POSITION, H)
                # Trim to exact sentinel count (handles edge case where n_sentinels
                # is not a perfect multiple, e.g. due to truncation)
                cnn_embs = cnn_embs[:n_sentinels]
            else:
                cnn_embs = torch.zeros(n_sentinels, H, dtype=dtype, device=device)

            # 4. Scatter into inputs_embeds at all sentinel positions (left-to-right order)
            move_positions = move_mask.nonzero(as_tuple=False)  # (n_sentinels, 2)
            b_idx = move_positions[:, 0]
            l_idx = move_positions[:, 1]
            inputs_embeds = inputs_embeds.index_put((b_idx, l_idx), cnn_embs, accumulate=False)

        # 5. Mask sentinel positions in labels — never predict them
        if labels is not None and n_sentinels > 0:
            labels = labels.clone()
            labels[move_mask] = -100

        # 6. Forward through LLM — sequence length unchanged
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def print_trainable_parameters(self) -> None:
        """Log parameter counts for LLM (LoRA) and CNN encoder."""
        if hasattr(self.llm, "print_trainable_parameters"):
            self.llm.print_trainable_parameters()

        trainable_params = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)
        all_param = sum(p.numel() for p in self.cnn.parameters())
        print(
            f"Encoder params: trainable={trainable_params:,d} || "
            f"all={all_param:,d} || trainable%={100 * trainable_params / all_param:.4f}"
        )
