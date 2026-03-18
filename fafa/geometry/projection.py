from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .pose import invert_pose


@torch.no_grad()
def _pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return xs, ys


def project_points(points_cam: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera-space 3D points into image space.

    points_cam: [B,P,3]
    k:          [B,3,3]
    returns uv [B,P,2], depth [B,P]
    """
    x = points_cam[..., 0]
    y = points_cam[..., 1]
    z = points_cam[..., 2].clamp(min=1e-6)
    fx = k[:, 0, 0].unsqueeze(1)
    fy = k[:, 1, 1].unsqueeze(1)
    cx = k[:, 0, 2].unsqueeze(1)
    cy = k[:, 1, 2].unsqueeze(1)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    uv = torch.stack([u, v], dim=-1)
    return uv, z


def _intrinsics_components(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = k.shape[0]
    fx = k[:, 0, 0].view(b, 1, 1, 1)
    fy = k[:, 1, 1].view(b, 1, 1, 1)
    cx = k[:, 0, 2].view(b, 1, 1, 1)
    cy = k[:, 1, 2].view(b, 1, 1, 1)
    return fx, fy, cx, cy


def shape_constraint_flow_from_depth(
    depth_src: torch.Tensor,
    pose_src: torch.Tensor,
    pose_tgt: torch.Tensor,
    k_src: torch.Tensor,
    k_tgt: torch.Tensor | None = None,
    mask_src: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute synthetic-to-target flow using source depth and two poses.

    depth_src: [B,1,H,W]
    pose_src:  [B,4,4] pose that produced the source synthetic view
    pose_tgt:  [B,4,4] current target pose estimate
    k_src:     [B,3,3] intrinsics of the source synthetic crop/view
    k_tgt:     [B,3,3] intrinsics of the target crop/view. If omitted, ``k_src`` is reused.
    mask_src:  [B,1,H,W] optional source object mask. When provided, background pixels
               are forced invalid even if the stored depth map contains positive sentinel values.

    returns:
      flow_s2t: [B,2,H,W]
      valid:    [B,1,H,W]
    """
    if k_tgt is None:
        k_tgt = k_src

    b, _, h, w = depth_src.shape
    device = depth_src.device
    dtype = depth_src.dtype

    if mask_src is not None:
        mask_src = (mask_src > 0.5).to(dtype=dtype)
        depth_used = depth_src * mask_src
    else:
        depth_used = depth_src

    xs, ys = _pixel_grid(h, w, device, dtype)
    xs = xs.view(1, 1, h, w).expand(b, 1, h, w)
    ys = ys.view(1, 1, h, w).expand(b, 1, h, w)

    z = depth_used.clamp(min=0.0)
    fx_src, fy_src, cx_src, cy_src = _intrinsics_components(k_src)
    fx_tgt, fy_tgt, cx_tgt, cy_tgt = _intrinsics_components(k_tgt)

    x = (xs - cx_src) / fx_src * z
    y = (ys - cy_src) / fy_src * z
    xyz_src_cam = torch.cat([x, y, z], dim=1).reshape(b, 3, -1)

    src_inv = invert_pose(pose_src)
    xyz_obj = src_inv[:, :3, :3] @ xyz_src_cam + src_inv[:, :3, 3:4]

    xyz_tgt_cam = pose_tgt[:, :3, :3] @ xyz_obj + pose_tgt[:, :3, 3:4]
    z_tgt = xyz_tgt_cam[:, 2:3, :]
    z_safe = z_tgt.clamp(min=1e-6)

    u_tgt = fx_tgt.reshape(b, 1, 1) * (xyz_tgt_cam[:, 0:1, :] / z_safe) + cx_tgt.reshape(b, 1, 1)
    v_tgt = fy_tgt.reshape(b, 1, 1) * (xyz_tgt_cam[:, 1:2, :] / z_safe) + cy_tgt.reshape(b, 1, 1)

    u_src = xs.reshape(b, 1, -1)
    v_src = ys.reshape(b, 1, -1)

    flow = torch.cat([u_tgt - u_src, v_tgt - v_src], dim=1).reshape(b, 2, h, w)
    valid = (z > 1e-6).reshape(b, 1, h, w)
    valid = valid & (z_tgt.reshape(b, 1, h, w) > 1e-6)
    valid = valid & (u_tgt.reshape(b, 1, h, w) >= 0) & (u_tgt.reshape(b, 1, h, w) <= (w - 1))
    valid = valid & (v_tgt.reshape(b, 1, h, w) >= 0) & (v_tgt.reshape(b, 1, h, w) <= (h - 1))
    if mask_src is not None:
        valid = valid & (mask_src > 0.5)
    return flow, valid.float()


def resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize optical flow and scale magnitudes consistently."""
    b, c, h, w = flow.shape
    new_h, new_w = size
    out = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    scale_x = new_w / max(1.0, w)
    scale_y = new_h / max(1.0, h)
    out[:, 0] *= scale_x
    out[:, 1] *= scale_y
    return out
