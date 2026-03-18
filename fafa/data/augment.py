from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _rand_uniform(batch: int, low: float, high: float, device: torch.device) -> torch.Tensor:
    return torch.empty((batch, 1, 1, 1), device=device).uniform_(low, high)


def color_jitter_tensor(
    x: torch.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> torch.Tensor:
    """A lightweight torchvision-free color jitter for BCHW tensors in [0,1]."""
    b = x.shape[0]
    device = x.device
    out = x

    if brightness > 0:
        factor = _rand_uniform(b, 1 - brightness, 1 + brightness, device)
        out = out * factor

    if contrast > 0:
        mean = out.mean(dim=(2, 3), keepdim=True)
        factor = _rand_uniform(b, 1 - contrast, 1 + contrast, device)
        out = (out - mean) * factor + mean

    if saturation > 0:
        gray = out.mean(dim=1, keepdim=True)
        factor = _rand_uniform(b, 1 - saturation, 1 + saturation, device)
        out = (out - gray) * factor + gray

    return out.clamp(0.0, 1.0)


def gaussian_noise_tensor(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    if sigma <= 0:
        return x
    return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)


def blur_tensor(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def noisy_student_augment(x: torch.Tensor) -> torch.Tensor:
    out = color_jitter_tensor(x, brightness=0.2, contrast=0.2, saturation=0.15)
    out = gaussian_noise_tensor(out, sigma=0.03)
    if torch.rand(1).item() < 0.5:
        out = blur_tensor(out, kernel_size=5)
    return out
