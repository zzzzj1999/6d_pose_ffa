from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from fafa.common import device_from_config, load_config, seed_everything
from fafa.data.dataset import PreparedContextPoseDataset, build_dataloaders, collate_prepared_context
from fafa.debug_utils import collect_model_debug_stats
from fafa.geometry.metrics import estimate_diameter_from_points, pose_metrics, translation_error_cm
from fafa.geometry.pose import pose_geodesic_distance_deg
from fafa.train_utils import build_model_from_cfg, move_batch_to_device


def build_eval_loader(cfg: Any) -> DataLoader:
    ds = PreparedContextPoseDataset(
        index_path=str(cfg.data.eval_index),
        n_context=int(cfg.data.n_context),
        style_index_path=None,
        zero_background_depth=bool(getattr(cfg.data, "zero_background_depth", True)),
        max_context_depth_m=getattr(cfg.data, "max_context_depth_m", None),
    )
    return DataLoader(
        ds,
        batch_size=min(8, int(cfg.runtime.batch_size)),
        shuffle=False,
        num_workers=int(cfg.runtime.num_workers),
        pin_memory=bool(cfg.runtime.pin_memory),
        collate_fn=collate_prepared_context,
    )


def summarize_pose(prefix: str, pose: torch.Tensor, gt_pose: torch.Tensor, mesh_points: torch.Tensor, symmetric: torch.Tensor) -> None:
    diam = estimate_diameter_from_points(mesh_points)
    metrics = pose_metrics(pose, gt_pose, mesh_points, diam, symmetric)
    rot_err = pose_geodesic_distance_deg(pose, gt_pose)
    trans_err = translation_error_cm(pose, gt_pose)
    print(f"{prefix}_metrics:", metrics)
    print(f"{prefix}_rot_err_deg_p50:", float(torch.quantile(rot_err, 0.5)))
    print(f"{prefix}_rot_err_deg_p90:", float(torch.quantile(rot_err, 0.9)))
    print(f"{prefix}_trans_err_cm_p50:", float(torch.quantile(trans_err, 0.5)))
    print(f"{prefix}_trans_err_cm_p90:", float(torch.quantile(trans_err, 0.9)))


def load_checkpoint_weights(model: torch.nn.Module, checkpoint: str, *, teacher: bool = False, device: torch.device) -> None:
    ckpt = torch.load(checkpoint, map_location=device)
    state = None
    if teacher and "teacher" in ckpt:
        state = ckpt["teacher"]
    elif "model" in ckpt:
        state = ckpt["model"]
    elif "student" in ckpt:
        state = ckpt["student"]
    if state is None:
        raise ValueError(f"Could not find model weights in {checkpoint}")
    model.load_state_dict(state, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe one batch of the refiner and print flow/pose diagnostics.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="eval", choices=["pretrain", "selfsup", "eval"])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--teacher", action="store_true", help="Load teacher weights when probing a selfsup checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.runtime.seed))
    device = device_from_config(cfg)

    if args.mode == "eval":
        loader = build_eval_loader(cfg)
    else:
        train_loader, val_loader = build_dataloaders(cfg, mode=args.mode)
        loader = val_loader if val_loader is not None else train_loader

    model = build_model_from_cfg(cfg, teacher=args.teacher).to(device)
    if args.checkpoint:
        load_checkpoint_weights(model, args.checkpoint, teacher=args.teacher, device=device)
    model.eval()

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        out = model(
            real_image=batch["image"],
            synth_images=batch["context_images"],
            synth_depths=batch["context_depths"],
            synth_masks=batch["context_masks"],
            synth_poses=batch["context_poses"],
            init_pose=batch["init_pose"],
            k=batch["K"],
            context_ks=batch.get("context_ks"),
        )

    print("sample_ids:", batch.get("sample_id", []))
    print("flow_debug:", collect_model_debug_stats(out))
    if "gt_pose" in batch:
        summarize_pose("init", batch["init_pose"], batch["gt_pose"], batch["mesh_points"], batch["symmetric"])
        summarize_pose("pred", out["pose"], batch["gt_pose"], batch["mesh_points"], batch["symmetric"])
        init_rot = pose_geodesic_distance_deg(batch["init_pose"], batch["gt_pose"])
        pred_rot = pose_geodesic_distance_deg(out["pose"], batch["gt_pose"])
        init_trans = translation_error_cm(batch["init_pose"], batch["gt_pose"])
        pred_trans = translation_error_cm(out["pose"], batch["gt_pose"])
        print("rot_improved_ratio:", float((pred_rot < init_rot).float().mean()))
        print("trans_improved_ratio:", float((pred_trans < init_trans).float().mean()))


if __name__ == "__main__":
    main()
