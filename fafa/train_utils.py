from __future__ import annotations

import os
from typing import Any, Dict

import torch

from fafa.common import ensure_dir
from fafa.geometry.metrics import estimate_diameter_from_points
from fafa.modeling import FAFANet


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_model_from_cfg(cfg: Any, teacher: bool = False) -> FAFANet:
    model_cfg = cfg.model
    outer_iters = int(model_cfg.teacher_outer_iters if teacher else model_cfg.student_outer_iters)
    model = FAFANet(
        feature_dim=int(model_cfg.feature_dim),
        hidden_dim=int(model_cfg.hidden_dim),
        outer_iters=outer_iters,
        translation_scale=float(model_cfg.translation_scale),
        geometric_consistency_px=float(model_cfg.geometric_consistency_px),
        max_disp_feat=float(getattr(model_cfg, "max_disp_feat", 32.0)),
        mask_prior_flow=bool(getattr(model_cfg, "mask_prior_flow", True)),
        mask_pred_flow=bool(getattr(model_cfg, "mask_pred_flow", True)),
        masked_pose_pooling=bool(getattr(model_cfg, "masked_pose_pooling", True)),
    )
    return model


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: Any, steps_per_epoch: int) -> tuple[torch.optim.Optimizer, Any]:
    train_cfg = cfg.train
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.max_lr),
        weight_decay=float(train_cfg.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(train_cfg.max_lr),
        epochs=int(train_cfg.epochs),
        steps_per_epoch=max(1, steps_per_epoch),
        pct_start=float(train_cfg.pct_start),
        anneal_strategy=str(train_cfg.anneal_strategy),
        div_factor=float(train_cfg.div_factor),
        final_div_factor=float(train_cfg.final_div_factor),
    )
    return optimizer, scheduler


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    extra: Dict[str, Any] | None = None,
) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {
        "model": model.state_dict(),
        "epoch": int(epoch),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


# Backward-compatible helper retained because eval utilities may import it.
def estimate_diameter_from_points_scalar(points: torch.Tensor) -> float:
    value = estimate_diameter_from_points(points)
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    return float(value)
