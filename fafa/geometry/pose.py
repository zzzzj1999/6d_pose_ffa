from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to a 3x3 rotation matrix."""
    if x.shape[-1] != 6:
        raise ValueError(f"Expected last dimension 6, got {x.shape}")
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def invert_pose(pose: torch.Tensor) -> torch.Tensor:
    r = pose[..., :3, :3]
    t = pose[..., :3, 3:4]
    r_inv = r.transpose(-1, -2)
    t_inv = -r_inv @ t
    out = torch.eye(4, device=pose.device, dtype=pose.dtype).expand(*pose.shape[:-2], 4, 4).clone()
    out[..., :3, :3] = r_inv
    out[..., :3, 3:4] = t_inv
    return out


def apply_delta_pose(
    base_pose: torch.Tensor,
    delta_rot6d: torch.Tensor,
    delta_t: torch.Tensor,
    translation_scale: float = 1.0,
) -> torch.Tensor:
    """Update pose with a decoupled relative rotation and additive translation.

    This is a faithful but explicit implementation choice because the paper states that
    a decoupled relative pose [RΔ|tΔ] is predicted and refined iteratively, but does not
    fully specify the translation parameterization in the public PDF.
    """
    if base_pose.ndim != 3 or base_pose.shape[-2:] != (4, 4):
        raise ValueError(f"base_pose must have shape [B,4,4], got {base_pose.shape}")
    d_r = rot6d_to_matrix(delta_rot6d)
    r = base_pose[:, :3, :3]
    t = base_pose[:, :3, 3]
    new_pose = base_pose.clone()
    new_pose[:, :3, :3] = d_r @ r
    new_pose[:, :3, 3] = t + delta_t * translation_scale
    new_pose[:, 3, :] = torch.tensor([0, 0, 0, 1], device=base_pose.device, dtype=base_pose.dtype)
    return new_pose


def transform_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Transform 3D points using a camera-space pose.

    points: [B,P,3] or [P,3]
    pose:   [B,4,4]
    returns [B,P,3]
    """
    if points.ndim == 2:
        points = points.unsqueeze(0).expand(pose.shape[0], -1, -1)
    r = pose[:, :3, :3]
    t = pose[:, :3, 3]
    return (r @ points.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)


def pose_geodesic_distance_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ra = a[:, :3, :3]
    rb = b[:, :3, :3]
    rel = ra @ rb.transpose(1, 2)
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    cos = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    ang = torch.rad2deg(torch.acos(cos))
    return ang
