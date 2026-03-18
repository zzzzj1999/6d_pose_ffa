from __future__ import annotations

from typing import Any, Dict

import torch
from tqdm import tqdm

from fafa.geometry.metrics import estimate_diameter_from_points, pose_metrics
from fafa.train_utils import move_batch_to_device


@torch.no_grad()
def evaluate_pose_loader(model: torch.nn.Module, loader: Any, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals = {
        "ADD(-S)_0.1d": 0.0,
        "5deg": 0.0,
        "5cm": 0.0,
        "rot_err_deg": 0.0,
        "trans_err_cm": 0.0,
        "add_mean_m": 0.0,
    }
    total_samples = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        if "gt_pose" not in batch:
            raise ValueError("Evaluation requires gt_pose in the dataset index")
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
        diameter_m = estimate_diameter_from_points(batch["mesh_points"])
        metrics = pose_metrics(
            pred_pose=out["pose"],
            gt_pose=batch["gt_pose"],
            model_points=batch["mesh_points"],
            diameter_m=diameter_m,
            symmetric=batch["symmetric"],
        )
        batch_size = int(batch["image"].shape[0])
        for k, v in metrics.items():
            totals[k] += float(v) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return {k: 0.0 for k in totals}
    return {k: v / total_samples for k, v in totals.items()}
