from __future__ import annotations

from typing import Dict

import torch

from .pose import pose_geodesic_distance_deg, transform_points


def add_metric(pred_pose: torch.Tensor, gt_pose: torch.Tensor, model_points: torch.Tensor) -> torch.Tensor:
    pred_pts = transform_points(model_points, pred_pose)
    gt_pts = transform_points(model_points, gt_pose)
    return torch.linalg.norm(pred_pts - gt_pts, dim=-1).mean(dim=-1)


def adds_metric(pred_pose: torch.Tensor, gt_pose: torch.Tensor, model_points: torch.Tensor) -> torch.Tensor:
    pred_pts = transform_points(model_points, pred_pose)
    gt_pts = transform_points(model_points, gt_pose)
    dists = torch.cdist(pred_pts, gt_pts, p=2)
    return dists.min(dim=-1).values.mean(dim=-1)


def translation_error_cm(pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
    diff = pred_pose[:, :3, 3] - gt_pose[:, :3, 3]
    return torch.linalg.norm(diff, dim=-1) * 100.0


def estimate_diameter_from_points(points: torch.Tensor) -> torch.Tensor:
    """Estimate object diameter from a point cloud batch.

    points: [B,P,3] or [P,3]
    returns: [B] or scalar tensor
    """
    if points.ndim == 2:
        mins = points.min(dim=0).values
        maxs = points.max(dim=0).values
        return torch.linalg.norm(maxs - mins)
    if points.ndim != 3:
        raise ValueError(f"Expected points with shape [P,3] or [B,P,3], got {points.shape}")
    mins = points.min(dim=1).values
    maxs = points.max(dim=1).values
    return torch.linalg.norm(maxs - mins, dim=-1)


def pose_metrics(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    model_points: torch.Tensor,
    diameter_m: float | torch.Tensor,
    symmetric: torch.Tensor,
) -> Dict[str, float]:
    rot_err = pose_geodesic_distance_deg(pred_pose, gt_pose)
    trans_err = translation_error_cm(pred_pose, gt_pose)
    add = add_metric(pred_pose, gt_pose, model_points)
    adds = adds_metric(pred_pose, gt_pose, model_points)
    metric = torch.where(symmetric, adds, add)

    if not isinstance(diameter_m, torch.Tensor):
        threshold = torch.full_like(metric, 0.1 * float(diameter_m))
    else:
        threshold = 0.1 * diameter_m.to(device=metric.device, dtype=metric.dtype)
        if threshold.ndim == 0:
            threshold = threshold.expand_as(metric)

    return {
        "ADD(-S)_0.1d": (metric < threshold).float().mean().item(),
        "5deg": (rot_err < 5.0).float().mean().item(),
        "5cm": (trans_err < 5.0).float().mean().item(),
        "rot_err_deg": rot_err.mean().item(),
        "trans_err_cm": trans_err.mean().item(),
        "add_mean_m": metric.mean().item(),
    }
