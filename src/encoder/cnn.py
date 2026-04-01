"""ResNet CNN trunk to encode an 8x8 chess board into 65 LLM tokens.

Architecture loosely based on Leela Chess Zero (LCZero) trunks:
- Input: 19-channel 8x8 spatial tensor (piece planes + castling + ep + move#).
- Blocks: Configurable number of ResidualBlocks ( Conv -> BN -> ReLU -> Conv -> BN -> + )
- Head: 2D learnable positional encoding -> project each of 64 spatial cells to out_dim,
  plus a 65th global token produced by cross-attention over the 64 spatial features.
  Output: (B, 65, out_dim) — 64 per-square tokens + 1 global summary token.

Each spatial token at position (file, rank) encodes all relationships of that square:
what piece occupies it, what attacks/defends it, pawn structure around it, etc.
The ResNet's receptive field covers the full board, so each cell has global context.

The 65th global token aggregates board-level evaluation signals (material balance,
king safety, mobility imbalance) into a single summary vector for the LLM.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += res
        x = self.relu(x)
        return x


class ChessEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 19,
        hidden_size: int = 128,
        num_blocks: int = 6,
        out_dim: int = 2560,  # Qwen3.5-4B hidden size
    ):
        """ResNet trunk producing 65 tokens: 64 per-square + 1 global summary.

        Args:
            in_channels: Depth of the board tensor (19-channel single-board format).
            hidden_size: Number of filters in the ResNet trunk.
            num_blocks: Number of Residual blocks to chain.
            out_dim: Output dimension per token (must match LLM hidden_size).

        Output shape: (B, 65, out_dim)
            Tokens 0–63: per-square tokens in row-major order (a1=0, b1=1, ..., h8=63).
            Token 64: global summary token aggregated via cross-attention.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_size) for _ in range(num_blocks)])

        # 2D learnable positional encoding: (1, hidden_size, 8, 8)
        # Separate file and rank embeddings combined additively — efficient and
        # generalises better than a flat 64-position embedding.
        self.pos_file = nn.Parameter(torch.zeros(1, hidden_size, 1, 8))  # broadcast over ranks
        self.pos_rank = nn.Parameter(torch.zeros(1, hidden_size, 8, 1))  # broadcast over files
        nn.init.trunc_normal_(self.pos_file, std=0.02)
        nn.init.trunc_normal_(self.pos_rank, std=0.02)

        # Project each spatial cell from hidden_size → out_dim.
        # 2-layer MLP with GELU (LLaVA-1.5 standard) gives the alignment phase
        # enough capacity to map CNN features into the LLM's embedding space.
        # LayerNorm at the output anchors the projected vectors to the same
        # norm/distribution as the LLM's token embeddings, preventing the
        # 14× norm mismatch that killed gradient flow in phase0 runs.
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        # Global token: 1 learnable query attending over the 64 spatial features.
        # Same cross-attention pattern as EncoderSF15Head in encoder-pretrain.
        # Operates in the small hidden_size space before projection for efficiency.
        self.global_query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.global_query, std=0.02)
        self.global_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.global_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_size, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-projection spatial features in CNN hidden space.

        Args:
            x: (B, in_channels, 8, 8)

        Returns:
            (B, 64, hidden_size) — before the Linear projection to out_dim.
            Used by the pretraining head to do lightweight cross-attention in
            the small hidden space rather than the large LLM hidden space.
        """
        x = self.conv_input(x)  # (B, H, 8, 8)
        x = self.blocks(x)  # (B, H, 8, 8)
        x = x + self.pos_file + self.pos_rank  # (B, H, 8, 8)
        B, H, _, _ = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, 8, 8, H)
        return x.reshape(B, 64, H)  # (B, 64, H)

    def _compute_global(self, spatial: torch.Tensor) -> torch.Tensor:
        """Compute global summary token via cross-attention over 64 spatial features.

        Args:
            spatial: (B, 64, hidden_size)

        Returns:
            (B, 1, out_dim)
        """
        B = spatial.size(0)
        q = self.global_query.expand(B, -1, -1)  # (B, 1, H)
        k = self.global_k(spatial)  # (B, 64, H)
        v = self.global_v(spatial)  # (B, 64, H)
        scale = self.hidden_size**-0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)  # (B, 1, 64)
        out = attn @ v  # (B, 1, H)
        return self.global_proj(out)  # (B, 1, out_dim)

    def forward_clip(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning grid and global tokens separately (for CLIP training).

        Args:
            x: (B, in_channels, 8, 8)

        Returns:
            grid_tokens: (B, 64, out_dim) — per-square tokens.
            global_token: (B, 1, out_dim) — board-level summary token.
        """
        spatial = self.spatial_features(x)  # (B, 64, H)
        grid_tokens = self.proj(spatial)  # (B, 64, out_dim)
        global_token = self._compute_global(spatial)  # (B, 1, out_dim)
        return grid_tokens, global_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning all 65 tokens concatenated.

        Args:
            x: (B, in_channels, 8, 8) float tensor.

        Returns:
            (B, 65, out_dim) — tokens 0–63 are per-square (row-major a1..h8),
            token 64 is the global summary token.
        """
        grid_tokens, global_token = self.forward_clip(x)
        return torch.cat([grid_tokens, global_token], dim=1)  # (B, 65, out_dim)
