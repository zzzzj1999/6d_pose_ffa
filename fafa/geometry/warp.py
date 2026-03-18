from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def base_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=0)  # [2,H,W]


def _coords_to_grid(coords: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = coords[:, 0]
    y = coords[:, 1]
    x_norm = 2.0 * x / max(width - 1, 1) - 1.0
    y_norm = 2.0 * y / max(height - 1, 1) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def backward_warp_target_to_source(target: torch.Tensor, flow_s2t: torch.Tensor) -> torch.Tensor:
    """Warp a target-frame tensor into the source frame using source->target flow.

    At source pixel x_s, the corresponding target location is x_t = x_s + flow_s2t.
    Therefore we sample the target tensor at x_t.
    """
    b, _, h, w = target.shape
    grid = base_grid(h, w, target.device, target.dtype).unsqueeze(0).expand(b, -1, -1, -1)
    coords = grid + flow_s2t
    sampling_grid = _coords_to_grid(coords, h, w)
    return F.grid_sample(target, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def forward_splat_mask(mask_src: torch.Tensor, flow_s2t: torch.Tensor) -> torch.Tensor:
    """Forward warp a single-channel source mask using bilinear splatting."""
    if mask_src.shape[1] != 1:
        raise ValueError(f"mask_src must have one channel, got {mask_src.shape}")
    b, _, h, w = mask_src.shape
    device = mask_src.device
    dtype = mask_src.dtype
    grid = base_grid(h, w, device, dtype).unsqueeze(0).expand(b, -1, -1, -1)
    coords = grid + flow_s2t
    x = coords[:, 0].reshape(b, -1)
    y = coords[:, 1].reshape(b, -1)
    val = mask_src.reshape(b, -1)

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    out = torch.zeros((b, h * w), device=device, dtype=dtype)

    def scatter(xx: torch.Tensor, yy: torch.Tensor, ww: torch.Tensor) -> None:
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        idx = (yy.long() * w + xx.long()).clamp(min=0, max=h * w - 1)
        contrib = ww * val
        out.scatter_add_(1, idx * valid.long(), contrib * valid.float())

    scatter(x0, y0, wa)
    scatter(x0, y1, wb)
    scatter(x1, y0, wc)
    scatter(x1, y1, wd)

    return out.view(b, 1, h, w).clamp(0.0, 1.0)
