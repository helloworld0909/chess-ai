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
        # move_counts (B,) alongside input_ids. board_tensors_flat/move_counts
        # are optional — omit them (or pass None) for TRL log-prob passes on
        # completion-only sequences; zero embeddings will be used instead.
        out = model(
            input_ids=ids,             # (B, L)
            board_tensors_flat=flat,   # (N_total, 38, 8, 8)  — optional
            move_counts=counts,        # (B,)  — optional, kept for API compat
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
        input_ids: torch.LongTensor,
        board_tensors_flat: Optional[torch.Tensor] = None,
        move_counts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """Forward pass — splices CNN embeddings into move token positions.

        Args:
            input_ids: (B, L)
            board_tensors_flat: (N_total, 38, 8, 8) all board tensors for the
                batch, concatenated in sequence order across all examples.
                If None (e.g. TRL log-prob pass on completion-only sequences),
                zero embeddings are used for all <|move|> positions.
            move_counts: (B,) number of move tokens per example. Unused in forward
                (counts are derived from input_ids); kept for API compatibility.
            attention_mask: (B, L)
            labels: (B, L) — move token positions will be set to -100.
        """
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype
        B, L = input_ids.shape

        # 1. Text embeddings: (B, L, H)
        inputs_embeds = self.embed_tokens(input_ids)

        # 2. Find all <|move|> positions: (N_total, 2) of (batch_idx, seq_idx)
        move_mask = input_ids == self.move_token_id
        n_moves = move_mask.sum().item()

        if n_moves > 0:
            # 3. Encode all board tensors in one batched CNN call: (N_total, H)
            # CNN weights are float32; cast board tensors to CNN weight dtype, not embed dtype
            if board_tensors_flat is not None and board_tensors_flat.shape[0] > 0:
                cnn_dtype = next(self.cnn.parameters()).dtype
                cnn_embs = self.cnn(board_tensors_flat.to(device=device, dtype=cnn_dtype))
                cnn_embs = cnn_embs.to(dtype=dtype)  # cast output to LLM dtype
                # Truncate/pad to match actual move token count
                if cnn_embs.shape[0] != n_moves:
                    if cnn_embs.shape[0] > n_moves:
                        cnn_embs = cnn_embs[:n_moves]
                    else:
                        pad = torch.zeros(
                            n_moves - cnn_embs.shape[0],
                            cnn_embs.shape[-1],
                            dtype=dtype,
                            device=device,
                        )
                        cnn_embs = torch.cat([cnn_embs, pad], dim=0)
            else:
                H = inputs_embeds.shape[-1]
                cnn_embs = torch.zeros(n_moves, H, dtype=dtype, device=device)

            # 4. Scatter CNN embeddings directly into inputs_embeds at move positions
            #    index_put with accumulate=False is differentiable under autocast.
            move_positions = move_mask.nonzero(as_tuple=False)
            b_idx = move_positions[:, 0]
            l_idx = move_positions[:, 1]
            inputs_embeds = inputs_embeds.index_put((b_idx, l_idx), cnn_embs, accumulate=False)

        # 5. Mask <|move|> positions in labels — never predict the move token itself
        if labels is not None and n_moves > 0:
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
