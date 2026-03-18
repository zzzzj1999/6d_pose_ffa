from __future__ import annotations

import torch
import torch.nn as nn

from fafa.geometry.warp import backward_warp_target_to_source
from .blocks import ConvGRUCell, ConvNormAct, ResidualBlock


class RAFTLiteFlowRegressor(nn.Module):
    """A compact RAFT-inspired recurrent flow head.

    The residual flow head is intentionally bounded. In this reproduction the
    biggest training failure mode was numerical explosion of predicted flow,
    which then corrupted the pose update. Starting from zero residual flow and
    clipping the residual to a configurable range makes the refinement behave as
    a correction around the analytic prior instead of an unconstrained flow
    generator.
    """

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128, max_disp_feat: float = 32.0) -> None:
        super().__init__()
        in_dim = feature_dim * 3 + 2
        self.max_disp_feat = float(max_disp_feat)
        self.input_proj = nn.Sequential(
            ConvNormAct(in_dim, hidden_dim, 3, 1),
            ResidualBlock(hidden_dim),
        )
        self.gru = ConvGRUCell(hidden_dim, hidden_dim)
        self.flow_head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, 3, 1),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
        )
        # Start as "use the analytic prior flow".
        nn.init.zeros_(self.flow_head[-1].weight)
        if self.flow_head[-1].bias is not None:
            nn.init.zeros_(self.flow_head[-1].bias)

    def forward(
        self,
        real_feat: torch.Tensor,
        synth_feat: torch.Tensor,
        prior_flow_feat: torch.Tensor,
        hidden: torch.Tensor | None = None,
        valid_mask_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        warped_real = backward_warp_target_to_source(real_feat, prior_flow_feat)
        if valid_mask_feat is not None:
            valid_mask_feat = valid_mask_feat.clamp(0.0, 1.0)
            warped_real = warped_real * valid_mask_feat
            synth_feat = synth_feat * valid_mask_feat
            prior_flow_feat = prior_flow_feat * valid_mask_feat

        x = torch.cat([synth_feat, warped_real, synth_feat - warped_real, prior_flow_feat], dim=1)
        x = self.input_proj(x)
        if hidden is None:
            hidden = torch.zeros_like(x)
        hidden = self.gru(hidden, x)
        if valid_mask_feat is not None:
            hidden = hidden * valid_mask_feat

        residual_flow = torch.tanh(self.flow_head(hidden)) * self.max_disp_feat
        if valid_mask_feat is not None:
            residual_flow = residual_flow * valid_mask_feat

        pred_flow = prior_flow_feat + residual_flow
        pred_flow = pred_flow.clamp(-self.max_disp_feat, self.max_disp_feat)
        if valid_mask_feat is not None:
            pred_flow = pred_flow * valid_mask_feat
        return hidden, pred_flow
