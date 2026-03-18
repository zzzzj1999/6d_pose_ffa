from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from fafa.geometry.pose import apply_delta_pose
from fafa.geometry.projection import resize_flow, shape_constraint_flow_from_depth
from .encoder import BasicEncoder
from .flow_regressor import RAFTLiteFlowRegressor
from .pose_regressor import PoseRegressor


class FAFANet(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        outer_iters: int = 4,
        translation_scale: float = 0.02,
        geometric_consistency_px: float = 3.0,
        max_disp_feat: float = 32.0,
        mask_prior_flow: bool = True,
        mask_pred_flow: bool = True,
        masked_pose_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.feature_encoder = BasicEncoder(in_channels=3, feature_dim=feature_dim)
        self.flow_regressor = RAFTLiteFlowRegressor(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            max_disp_feat=max_disp_feat,
        )
        self.pose_regressor = PoseRegressor(hidden_dim=hidden_dim, translation_scale=translation_scale)
        self.outer_iters = int(outer_iters)
        self.geometric_consistency_px = float(geometric_consistency_px)
        self.mask_prior_flow = bool(mask_prior_flow)
        self.mask_pred_flow = bool(mask_pred_flow)
        self.masked_pose_pooling = bool(masked_pose_pooling)

    def forward(
        self,
        real_image: torch.Tensor,
        synth_images: torch.Tensor,
        synth_depths: torch.Tensor,
        synth_masks: torch.Tensor,
        synth_poses: torch.Tensor,
        init_pose: torch.Tensor,
        k: torch.Tensor,
        context_ks: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Run FAFA refinement.

        real_image:   [B,3,H,W]
        synth_images: [B,N,3,H,W]
        synth_depths: [B,N,1,H,W]
        synth_masks:  [B,N,1,H,W]
        synth_poses:  [B,N,4,4]
        init_pose:    [B,4,4]
        k:            [B,3,3]      target intrinsics
        context_ks:   [B,N,3,3]    source synthetic intrinsics; defaults to target intrinsics
        """
        b, n, _, h, w = synth_images.shape
        if context_ks is None:
            context_ks = k.unsqueeze(1).expand(-1, n, -1, -1)

        real_feat = self.feature_encoder(real_image)
        feat_h, feat_w = real_feat.shape[-2:]

        synth_flat = synth_images.reshape(b * n, 3, h, w)
        synth_feat = self.feature_encoder(synth_flat).reshape(b, n, -1, feat_h, feat_w)
        feat_masks = F.interpolate(
            synth_masks.reshape(b * n, 1, h, w),
            size=(feat_h, feat_w),
            mode="nearest",
        ).reshape(b, n, 1, feat_h, feat_w)

        current_pose = init_pose.clone()
        hidden_states: List[torch.Tensor | None] = [None for _ in range(n)]
        flows_full: List[torch.Tensor] = [torch.zeros((b, 2, h, w), device=real_image.device, dtype=real_image.dtype) for _ in range(n)]
        flows_feat: List[torch.Tensor] = [torch.zeros((b, 2, feat_h, feat_w), device=real_image.device, dtype=real_image.dtype) for _ in range(n)]
        prior_flows_full: List[torch.Tensor] = [torch.zeros((b, 2, h, w), device=real_image.device, dtype=real_image.dtype) for _ in range(n)]
        prior_valid_full: List[torch.Tensor] = [torch.zeros((b, 1, h, w), device=real_image.device, dtype=real_image.dtype) for _ in range(n)]

        for _ in range(self.outer_iters):
            hiddens_this_iter: List[torch.Tensor] = []
            flows_this_iter: List[torch.Tensor] = []
            pool_masks_this_iter: List[torch.Tensor] = []
            for i in range(n):
                prior_flow_full, prior_valid_i = shape_constraint_flow_from_depth(
                    synth_depths[:, i],
                    synth_poses[:, i],
                    current_pose,
                    k_src=context_ks[:, i],
                    k_tgt=k,
                    mask_src=synth_masks[:, i],
                )
                prior_valid_i = prior_valid_i * synth_masks[:, i]
                prior_flow_full = prior_flow_full * prior_valid_i if self.mask_prior_flow else prior_flow_full
                prior_flow_feat = resize_flow(prior_flow_full, (feat_h, feat_w))
                feat_valid_i = F.interpolate(prior_valid_i, size=(feat_h, feat_w), mode="nearest")
                feat_support_i = feat_valid_i * feat_masks[:, i]

                hidden_states[i], pred_flow_feat = self.flow_regressor(
                    real_feat,
                    synth_feat[:, i],
                    prior_flow_feat,
                    hidden_states[i],
                    valid_mask_feat=feat_support_i,
                )
                if self.mask_pred_flow:
                    pred_flow_feat = pred_flow_feat * feat_masks[:, i]
                    hidden_states[i] = hidden_states[i] * feat_masks[:, i]

                pred_flow_full = resize_flow(pred_flow_feat, (h, w))
                if self.mask_pred_flow:
                    pred_flow_full = pred_flow_full * synth_masks[:, i]

                flows_full[i] = pred_flow_full
                flows_feat[i] = pred_flow_feat
                prior_flows_full[i] = prior_flow_full
                prior_valid_full[i] = prior_valid_i
                hiddens_this_iter.append(hidden_states[i])
                flows_this_iter.append(pred_flow_feat)
                pool_masks_this_iter.append(feat_masks[:, i] if self.masked_pose_pooling else feat_support_i)

            hidden_stack = torch.stack(hiddens_this_iter, dim=1)
            flow_stack = torch.stack(flows_this_iter, dim=1)
            pool_mask_stack = torch.stack(pool_masks_this_iter, dim=1)
            delta_rot6d, delta_t = self.pose_regressor(
                hidden_stack,
                flow_stack,
                mask=pool_mask_stack if self.masked_pose_pooling else None,
            )
            current_pose = apply_delta_pose(current_pose, delta_rot6d, delta_t, translation_scale=1.0)

        flow_valid_list: List[torch.Tensor] = []
        analytic_flows: List[torch.Tensor] = []
        for i in range(n):
            analytic_flow_i, analytic_valid_i = shape_constraint_flow_from_depth(
                synth_depths[:, i],
                synth_poses[:, i],
                current_pose,
                k_src=context_ks[:, i],
                k_tgt=k,
                mask_src=synth_masks[:, i],
            )
            flow_err = torch.linalg.norm(flows_full[i] - analytic_flow_i, dim=1, keepdim=True)
            valid = analytic_valid_i * (flow_err < self.geometric_consistency_px).float() * synth_masks[:, i]
            flow_valid_list.append(valid)
            analytic_flows.append(analytic_flow_i)

        flows_full_stack = torch.stack(flows_full, dim=1)
        prior_flows_full_stack = torch.stack(prior_flows_full, dim=1)
        flow_valid_stack = torch.stack(flow_valid_list, dim=1)
        prior_valid_stack = torch.stack(prior_valid_full, dim=1)

        return {
            "pose": current_pose,
            "flows": flows_full_stack,
            "flows_feat": torch.stack(flows_feat, dim=1),
            "flow_valid": flow_valid_stack,
            "analytic_flows": torch.stack(analytic_flows, dim=1),
            "prior_flows": prior_flows_full_stack,
            "prior_valid": prior_valid_stack,
            "real_feat": real_feat,
            "synth_feats": synth_feat,
            "debug_flow_abs_max": flows_full_stack.abs().amax(),
            "debug_prior_flow_abs_max": prior_flows_full_stack.abs().amax(),
            "debug_flow_valid_ratio": flow_valid_stack.mean(),
            "debug_mask_ratio": synth_masks.mean(),
        }
