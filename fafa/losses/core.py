from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from fafa.geometry.pose import transform_points
from fafa.geometry.warp import backward_warp_target_to_source, forward_splat_mask
from .census import census_loss


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def weighted_mean(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return x.mean()
    denom = weight.sum().clamp(min=1.0)
    return (x * weight).sum() / denom


def flow_supervision_loss(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    error = charbonnier(pred_flow - target_flow).mean(dim=1, keepdim=True)
    return weighted_mean(error, valid_mask)


def photometric_consistency_loss(
    real_image: torch.Tensor,
    pred_flow: torch.Tensor,
    pseudo_flow: torch.Tensor,
    visible_mask: torch.Tensor,
) -> torch.Tensor:
    real_warp_pred = backward_warp_target_to_source(real_image, pred_flow)
    real_warp_pseudo = backward_warp_target_to_source(real_image, pseudo_flow)
    return census_loss(real_warp_pred, real_warp_pseudo, mask=visible_mask)


def warp_mask_consistency_loss(
    synth_mask: torch.Tensor,
    pred_flow: torch.Tensor,
    pseudo_flow: torch.Tensor,
) -> torch.Tensor:
    pred = forward_splat_mask(synth_mask, pred_flow)
    pseudo = forward_splat_mask(synth_mask, pseudo_flow)
    return F.l1_loss(pred, pseudo)


def feature_level_loss(
    real_feat: torch.Tensor,
    synth_feat: torch.Tensor,
    pred_flow_feat: torch.Tensor,
    valid_mask_feat: torch.Tensor,
) -> torch.Tensor:
    warped_real_feat = backward_warp_target_to_source(real_feat, pred_flow_feat)
    diff = charbonnier(warped_real_feat - synth_feat).mean(dim=1, keepdim=True)
    return weighted_mean(diff, valid_mask_feat)


def point_matching_loss(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    model_points: torch.Tensor,
    symmetric: torch.Tensor | bool,
) -> torch.Tensor:
    pred_pts = transform_points(model_points, pred_pose)
    tgt_pts = transform_points(model_points, target_pose)
    if isinstance(symmetric, bool):
        symmetric = torch.full((pred_pose.shape[0],), symmetric, device=pred_pose.device, dtype=torch.bool)
    symmetric = symmetric.to(device=pred_pose.device, dtype=torch.bool)

    add = torch.linalg.norm(pred_pts - tgt_pts, dim=-1).mean(dim=-1)
    if symmetric.any():
        adds = torch.cdist(pred_pts, tgt_pts, p=2).min(dim=-1).values.mean(dim=-1)
        add = torch.where(symmetric, adds, add)
    return add.mean()


def self_supervised_loss(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    real_image: torch.Tensor,
    synth_masks: torch.Tensor,
    model_points: torch.Tensor,
    symmetric: torch.Tensor,
    gamma1: float,
    gamma2: float,
    gamma3: float,
    gamma4: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    pred_flows = student_out["flows"]         # [B,N,2,H,W]
    pseudo_flows = teacher_out["flows"].detach()
    flow_valid = teacher_out["flow_valid"].detach() * synth_masks

    b, n, _, h, w = pred_flows.shape
    flow_loss = 0.0
    photo_loss = 0.0
    warp_mask_loss = 0.0
    feat_loss = 0.0

    real_feat = student_out["real_feat"]
    synth_feats = student_out["synth_feats"]
    pred_flows_feat = student_out["flows_feat"]
    feat_valid = F.interpolate(flow_valid.reshape(b * n, 1, h, w), size=real_feat.shape[-2:], mode="nearest")
    feat_valid = feat_valid.reshape(b, n, 1, real_feat.shape[-2], real_feat.shape[-1])

    for i in range(n):
        flow_loss = flow_loss + flow_supervision_loss(pred_flows[:, i], pseudo_flows[:, i], flow_valid[:, i])
        photo_loss = photo_loss + photometric_consistency_loss(real_image, pred_flows[:, i], pseudo_flows[:, i], flow_valid[:, i])
        warp_mask_loss = warp_mask_loss + warp_mask_consistency_loss(synth_masks[:, i], pred_flows[:, i], pseudo_flows[:, i])
        feat_loss = feat_loss + feature_level_loss(real_feat, synth_feats[:, i], pred_flows_feat[:, i], feat_valid[:, i])

    flow_loss = flow_loss / n
    photo_loss = photo_loss / n
    warp_mask_loss = warp_mask_loss / n
    feat_loss = feat_loss / n

    pose_loss = point_matching_loss(student_out["pose"], teacher_out["pose"].detach(), model_points, symmetric)
    img_level = gamma1 * flow_loss + gamma2 * photo_loss + warp_mask_loss
    total = gamma3 * pose_loss + gamma4 * img_level + feat_loss
    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_flow": float(flow_loss.detach().cpu()),
        "loss_photo": float(photo_loss.detach().cpu()),
        "loss_warp_mask": float(warp_mask_loss.detach().cpu()),
        "loss_feat": float(feat_loss.detach().cpu()),
        "loss_pose": float(pose_loss.detach().cpu()),
    }
    return total, stats
