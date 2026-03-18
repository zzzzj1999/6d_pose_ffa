from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvNormAct, ResidualBlock


def _masked_average_pool_2d(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=(-2, -1))
    mask = mask.clamp(0.0, 1.0)
    denom = mask.sum(dim=(-2, -1)).clamp(min=1.0)
    return (x * mask).sum(dim=(-2, -1)) / denom


class PoseRegressor(nn.Module):
    def __init__(self, hidden_dim: int = 128, translation_scale: float = 0.02) -> None:
        super().__init__()
        self.translation_scale = float(translation_scale)
        self.stem = nn.Sequential(
            ConvNormAct(hidden_dim + 2, hidden_dim, 3, 1),
            ResidualBlock(hidden_dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.rot = nn.Linear(hidden_dim, 6)
        self.trans = nn.Linear(hidden_dim, 3)
        self._init_heads()

    def _init_heads(self) -> None:
        # Start from identity pose update: dR ≈ I, dt ≈ 0.
        nn.init.zeros_(self.rot.weight)
        nn.init.constant_(self.rot.bias, 0.0)
        with torch.no_grad():
            self.rot.bias[:] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        nn.init.zeros_(self.trans.weight)
        nn.init.zeros_(self.trans.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        mean_flow: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate decoupled pose delta.

        Supports either a single feature map pair [B,C,H,W]/[B,2,H,W] or a stack
        of per-context maps [B,N,C,H,W]/[B,N,2,H,W]. In the stacked case, we do
        not average spatial maps across contexts before pooling because different
        source views live in different image coordinate frames.
        """
        if hidden.ndim == 5:
            b, n, c, h, w = hidden.shape
            x = torch.cat([hidden, mean_flow], dim=2).reshape(b * n, c + 2, h, w)
            x = self.stem(x)
            mask_flat = None
            if mask is not None:
                if mask.ndim == 4:
                    mask = mask.unsqueeze(1)
                mask_flat = mask.reshape(b * n, 1, h, w)
            pooled = _masked_average_pool_2d(x, mask_flat).reshape(b, n, -1)
            if mask is not None:
                valid_ctx = (mask.reshape(b, n, -1).sum(dim=-1, keepdim=True) > 0).float()
                pooled = (pooled * valid_ctx).sum(dim=1) / valid_ctx.sum(dim=1).clamp(min=1.0)
            else:
                pooled = pooled.mean(dim=1)
        elif hidden.ndim == 4:
            x = torch.cat([hidden, mean_flow], dim=1)
            x = self.stem(x)
            pooled = _masked_average_pool_2d(x, mask)
        else:
            raise ValueError(f"Unexpected hidden shape: {hidden.shape}")

        latent = self.fc(pooled)
        rot6d = self.rot(latent)
        trans = self.trans(latent) * self.translation_scale
        return rot6d, trans
