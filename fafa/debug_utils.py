from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch

from fafa.common import ensure_dir


def _as_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


@torch.no_grad()
def tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    x = x.detach()
    finite = torch.isfinite(x)
    out: Dict[str, Any] = {
        "shape": list(x.shape),
        "finite_ratio": float(finite.float().mean().cpu()),
    }
    if finite.any():
        vals = x[finite]
        out.update(
            {
                "min": float(vals.min().cpu()),
                "max": float(vals.max().cpu()),
                "mean": float(vals.mean().cpu()),
                "abs_max": float(vals.abs().max().cpu()),
                "p50": float(torch.quantile(vals, 0.5).cpu()),
                "p90": float(torch.quantile(vals, 0.9).cpu()),
                "p95": float(torch.quantile(vals, 0.95).cpu()),
            }
        )
    return out


def cfg_section(cfg: Any, name: str) -> Any:
    return getattr(cfg.train, name, None)


def debug_cfg(cfg: Any) -> Any:
    dbg = cfg_section(cfg, "debug")
    return dbg if dbg is not None else {}


def get_debug_value(cfg: Any, key: str, default: Any) -> Any:
    dbg = debug_cfg(cfg)
    if isinstance(dbg, dict):
        return dbg.get(key, default)
    return getattr(dbg, key, default)


@torch.no_grad()
def collect_model_debug_stats(out: Dict[str, torch.Tensor]) -> Dict[str, float]:
    flows = out["flows"]
    prior_flows = out.get("prior_flows")
    valid = out.get("flow_valid")

    flat_abs = flows.detach().abs().reshape(-1)
    stats: Dict[str, float] = {
        "flow_abs_max_px": float(flat_abs.max().cpu()),
        "flow_abs_p90_px": float(torch.quantile(flat_abs, 0.9).cpu()),
    }
    if prior_flows is not None:
        prior_abs = prior_flows.detach().abs().reshape(-1)
        stats["prior_flow_abs_max_px"] = float(prior_abs.max().cpu())
        stats["prior_flow_abs_p90_px"] = float(torch.quantile(prior_abs, 0.9).cpu())
    if valid is not None:
        stats["flow_valid_ratio"] = float(valid.detach().float().mean().cpu())
    if "debug_mask_ratio" in out:
        stats["mask_ratio"] = float(out["debug_mask_ratio"].detach().cpu())
    return stats


@torch.no_grad()
def should_fail_on_flow_anomaly(cfg: Any, out: Dict[str, torch.Tensor]) -> tuple[bool, str, Dict[str, float]]:
    stats = collect_model_debug_stats(out)
    if not bool(get_debug_value(cfg, "enabled", True)):
        return False, "", stats
    max_abs = stats["flow_abs_max_px"]
    hard_max = float(get_debug_value(cfg, "max_abs_flow_px", 1.0e4))
    if not torch.isfinite(out["flows"]).all():
        return True, "non_finite_flow", stats
    if max_abs > hard_max:
        return True, f"flow_abs_max_px={max_abs:.4f} > {hard_max:.4f}", stats
    return False, "", stats


def dump_anomaly(
    dump_root: str | os.PathLike[str],
    *,
    stage: str,
    epoch: int,
    step: int,
    reason: str,
    batch: Dict[str, Any],
    out: Dict[str, torch.Tensor],
    extra: Dict[str, Any] | None = None,
) -> str:
    dump_root = str(dump_root)
    ensure_dir(dump_root)
    case_dir = Path(dump_root) / f"{stage}_e{epoch:03d}_s{step:06d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "stage": stage,
        "epoch": int(epoch),
        "step": int(step),
        "reason": reason,
        "sample_id": list(batch.get("sample_id", [])),
        "out_stats": {
            "flows": tensor_stats(out["flows"]),
            "flow_valid": tensor_stats(out["flow_valid"]) if "flow_valid" in out else None,
            "prior_flows": tensor_stats(out["prior_flows"]) if "prior_flows" in out else None,
        },
    }
    if "image" in batch:
        meta["batch_image_stats"] = tensor_stats(batch["image"])
    if "context_masks" in batch:
        meta["batch_context_masks_stats"] = tensor_stats(batch["context_masks"])
    if "context_depths" in batch:
        meta["batch_context_depths_stats"] = tensor_stats(batch["context_depths"])
    if extra is not None:
        meta["extra"] = extra

    with open(case_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    tensor_payload: Dict[str, Any] = {}
    for key in [
        "image",
        "context_images",
        "context_masks",
        "context_depths",
        "init_pose",
        "gt_pose",
        "K",
        "context_ks",
        "context_poses",
    ]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            tensor_payload[key] = batch[key].detach().cpu()
    for key in ["flows", "flow_valid", "prior_flows", "prior_valid", "pose"]:
        if key in out and isinstance(out[key], torch.Tensor):
            tensor_payload[key] = out[key].detach().cpu()
    torch.save(tensor_payload, case_dir / "tensors.pt")
    return str(case_dir)
