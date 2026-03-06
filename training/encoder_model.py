"""Wrapper module coupling Qwen3.5-4B with the ChessEncoder trunk."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from src.encoder import MOVE_TOKEN_ID
from src.encoder.cnn import ChessEncoder


class ChessLMWithEncoder(nn.Module):
    """Combines a base LLM (e.g., Qwen3.5-4B) with a ResNet board encoder.

    The encoder processes 38-channel 8x8 spatial tensors (before+after board)
    and injects them as CNN embeddings directly into the token embedding sequence
    at positions occupied by the <|vision_pad|> sentinel token. Each <|move|> in the
    input is replaced in-place with the CNN embedding for that specific
    (board_before, move) pair — the sequence length is unchanged.

    Usage:
        # collator provides board_tensors_flat (N_total, 38, 8, 8) and
        # move_counts (B,) so the forward pass knows which tensors belong
        # to which batch item.
        out = model(
            board_tensors_flat=flat,   # (N_total, 38, 8, 8)
            move_counts=counts,        # (B,)
            input_ids=ids,             # (B, L)
            attention_mask=mask,       # (B, L)
            labels=labels,             # (B, L)
        )
    """

    def __init__(
        self,
        llm: nn.Module,
        hidden_size: int = 2560,
        cnn_hidden_size: int = 512,
        cnn_num_blocks: int = 15,
        move_token_id: int = MOVE_TOKEN_ID,
    ):
        super().__init__()
        self.llm = llm
        self.move_token_id = move_token_id
        # 72M param ResNet (15 blocks, 512 filters, 38-ch input) -> hidden_size dim output
        self.cnn = ChessEncoder(
            in_channels=38,
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
        board_tensors_flat: torch.Tensor,
        move_counts: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """Forward pass — splices CNN embeddings into move token positions.

        Args:
            board_tensors_flat: (N_total, 38, 8, 8) all board tensors for the
                batch, concatenated in sequence order across all examples.
            move_counts: (B,) number of move tokens per example.
            input_ids: (B, L)
            attention_mask: (B, L)
            labels: (B, L) — move token positions will be set to -100.
        """
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype

        # 1. Encode all board tensors in one batched CNN call: (N_total, H)
        # CNN weights are float32; cast board tensors to CNN weight dtype, not embed dtype
        cnn_dtype = next(self.cnn.parameters()).dtype
        cnn_embs_flat = self.cnn(board_tensors_flat.to(device=device, dtype=cnn_dtype))
        cnn_embs_flat = cnn_embs_flat.to(dtype=dtype)  # cast output to LLM dtype

        # 2. Text embeddings: (B, L, H)
        text_embs = self.embed_tokens(input_ids)

        # 3. Find all <|move|> positions: (N_total, 2) of (batch_idx, seq_idx)
        move_positions = (input_ids == self.move_token_id).nonzero(as_tuple=False)

        if move_positions.shape[0] != cnn_embs_flat.shape[0]:
            raise RuntimeError(
                f"move token count mismatch: {move_positions.shape[0]} <|move|> tokens "
                f"in input_ids but {cnn_embs_flat.shape[0]} board tensors provided."
            )

        # 4. Scatter CNN embeddings into a (B, L, H) canvas then merge.
        #    index_put with accumulate=False is differentiable under autocast.
        B, L = input_ids.shape
        H = cnn_embs_flat.shape[-1]
        cnn_canvas = torch.zeros(B, L, H, dtype=dtype, device=device)
        if move_positions.shape[0] > 0:
            b_idx = move_positions[:, 0]
            l_idx = move_positions[:, 1]
            cnn_canvas = cnn_canvas.index_put((b_idx, l_idx), cnn_embs_flat, accumulate=False)

        # 5. Replace text embeddings at move positions with CNN embeddings
        move_mask_3d = (input_ids == self.move_token_id).unsqueeze(-1).expand(B, L, H)
        inputs_embeds = torch.where(move_mask_3d, cnn_canvas, text_embs)

        # 6. Mask <|move|> positions in labels — never predict the move token itself
        if labels is not None:
            labels = labels.clone()
            labels[input_ids == self.move_token_id] = -100

        # 7. Forward through LLM — sequence length unchanged
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
