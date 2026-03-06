"""ResNet CNN trunk to encode an 8x8 chess board into an LLM embedding vector.

Architecture loosely based on Leela Chess Zero (LCZero) trunks:
- Input: 38-channel 8x8 spatial tensor (before + after board, each 19 channels)
  For static position queries (no move), channels 19-37 == channels 0-18 (identity).
- Blocks: Configurable number of ResidualBlocks ( Conv -> BN -> ReLU -> Conv -> BN -> + )
- Head: AdaptiveAvgPool2d(1x1) -> Linear -> LLM embedding output dim
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
        in_channels: int = 38,
        hidden_size: int = 256,
        num_blocks: int = 10,
        out_dim: int = 2560,  # Qwen3.5-4B hidden size
    ):
        """Build a 10-block PyTorch ResNet to produce a single soft token.

        Args:
            in_channels: Depth of the board tensor.
                38 = before (19) + after (19). For static positions, before == after.
                19 = single board (legacy / ablation use).
            hidden_size: Number of filters in the ResNet trunk
            num_blocks: Number of Residual blocks to chain
            out_dim: Output dimension of the projection head (must match LLM hidden_size)
        """
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_size) for _ in range(num_blocks)])
        # Pool (N, 256, 8, 8) -> (N, 256, 1, 1) -> (N, 256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Project into LLM space
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Assumes input is floating point Tensor of shape (B, 19, 8, 8)."""
        x = self.conv_input(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        # Returns shape: (B, out_dim)
        return self.proj(x)
