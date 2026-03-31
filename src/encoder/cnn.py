"""ResNet CNN trunk to encode an 8x8 chess board into 64 per-square LLM tokens.

Architecture loosely based on Leela Chess Zero (LCZero) trunks:
- Input: 19-channel 8x8 spatial tensor (piece planes + castling + ep + move#).
- Blocks: Configurable number of ResidualBlocks ( Conv -> BN -> ReLU -> Conv -> BN -> + )
- Head: 2D learnable positional encoding -> project each of 64 spatial cells to out_dim
  Output: (B, 64, out_dim) — one token per square, row-major a1..h1, a2..h2, ..., a8..h8.

Each spatial token at position (file, rank) encodes all relationships of that square:
what piece occupies it, what attacks/defends it, pawn structure around it, etc.
The ResNet's receptive field covers the full board, so each cell has global context.
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
        hidden_size: int = 256,
        num_blocks: int = 10,
        out_dim: int = 2560,  # Qwen3.5-4B hidden size
    ):
        """ResNet trunk producing 64 per-square tokens with 2D positional encoding.

        Args:
            in_channels: Depth of the board tensor (19-channel single-board format).
            hidden_size: Number of filters in the ResNet trunk.
            num_blocks: Number of Residual blocks to chain.
            out_dim: Output dimension per token (must match LLM hidden_size).

        Output shape: (B, 64, out_dim) — 64 tokens, one per square in row-major order
            (a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63).
        """
        super().__init__()
        self.in_channels = in_channels
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
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_channels, 8, 8) float tensor.

        Returns:
            (B, 64, out_dim) — 64 per-square tokens in row-major order.
        """
        return self.proj(self.spatial_features(x))  # (B, 64, out_dim)
