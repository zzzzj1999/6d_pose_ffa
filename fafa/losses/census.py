from __future__ import annotations

import torch
import torch.nn.functional as F


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {x.shape}")
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def census_transform(x: torch.Tensor, patch_size: int = 7) -> torch.Tensor:
    gray = rgb_to_gray(x)
    b, _, h, w = gray.shape
    unfold = F.unfold(gray, kernel_size=patch_size, padding=patch_size // 2)
    unfold = unfold.view(b, patch_size * patch_size, h, w)
    center = gray
    diff = unfold - center
    diff = diff / torch.sqrt(0.81 + diff * diff)
    return diff


def census_loss(x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor | None = None, patch_size: int = 7) -> torch.Tensor:
    c1 = census_transform(x1, patch_size=patch_size)
    c2 = census_transform(x2, patch_size=patch_size)
    dist = (c1 - c2) ** 2
    dist = dist / (0.1 + dist)
    dist = dist.mean(dim=1, keepdim=True)
    if mask is None:
        return dist.mean()
    weighted = dist * mask
    denom = mask.sum().clamp(min=1.0)
    return weighted.sum() / denom
