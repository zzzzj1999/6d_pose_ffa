from __future__ import annotations

import torch


def fft_amplitude_phase(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return amplitude and phase for a BCHW tensor."""
    fx = torch.fft.fft2(x, dim=(-2, -1))
    amp = torch.abs(fx)
    phase = torch.angle(fx)
    return amp, phase


def fft_mix_augment(
    x_s: torch.Tensor,
    x_r: torch.Tensor,
    delta0: float = 0.5,
    beta: float = 1.0,
    clamp: bool = True,
) -> torch.Tensor:
    """Implements Eqs. (2)-(3) from the paper.

    x_s: synthetic image batch, BCHW, float in [0,1]
    x_r: real/style image batch, BCHW, float in [0,1]

    If a sample falls into the dropout branch, its amplitude spectrum is set to 1,
    matching the paper's Eq. (2).
    """
    if x_s.shape != x_r.shape:
        raise ValueError(f"x_s and x_r must have the same shape, got {x_s.shape} vs {x_r.shape}")

    amp_s, phase_s = fft_amplitude_phase(x_s)
    amp_r, _ = fft_amplitude_phase(x_r)

    b = x_s.shape[0]
    alpha = torch.rand((b, 1, 1, 1), device=x_s.device, dtype=x_s.dtype) * beta
    delta = torch.rand((b, 1, 1, 1), device=x_s.device, dtype=x_s.dtype)

    mixed_amp = (1.0 - alpha) * amp_s + alpha * amp_r
    dropout_amp = torch.ones_like(mixed_amp)
    mixed_amp = torch.where(delta < delta0, mixed_amp, dropout_amp)

    complex_spec = mixed_amp * torch.exp(1j * phase_s)
    x_aug = torch.fft.ifft2(complex_spec, dim=(-2, -1)).real
    if clamp:
        x_aug = x_aug.clamp(0.0, 1.0)
    return x_aug
