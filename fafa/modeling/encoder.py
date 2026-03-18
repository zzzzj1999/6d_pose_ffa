from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvNormAct, ResidualBlock


class BasicEncoder(nn.Module):
    """A lightweight 1/8-resolution encoder suitable for 256x256 crops."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, 32, kernel_size=7, stride=2),  # 1/2
            ResidualBlock(32),
            ConvNormAct(32, 64, kernel_size=3, stride=2),           # 1/4
            ResidualBlock(64),
            ConvNormAct(64, 96, kernel_size=3, stride=2),           # 1/8
            ResidualBlock(96),
            ConvNormAct(96, feature_dim, kernel_size=3, stride=1),
            ResidualBlock(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)
