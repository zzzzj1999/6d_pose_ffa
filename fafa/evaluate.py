from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fafa.common import device_from_config, format_metrics, load_config, save_json
from fafa.data.dataset import PreparedContextPoseDataset, collate_prepared_context
from fafa.eval_utils import evaluate_pose_loader
from fafa.train_utils import build_model_from_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--teacher", action="store_true", help="Use teacher_outer_iters instead of student_outer_iters")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_from_config(cfg)

    dataset = PreparedContextPoseDataset(
        index_path=str(cfg.data.eval_index),
        n_context=int(cfg.data.n_context),
        style_index_path=str(getattr(cfg.data, "real_style_index", "")) if getattr(cfg.data, "real_style_index", None) else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.runtime.batch_size),
        shuffle=False,
        num_workers=int(cfg.runtime.num_workers),
        pin_memory=bool(cfg.runtime.pin_memory),
        collate_fn=collate_prepared_context,
    )

    model = build_model_from_cfg(cfg, teacher=args.teacher).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" in ckpt:
        state = ckpt["model"]
    elif "student" in ckpt:
        state = ckpt["student"]
    else:
        raise ValueError("Checkpoint must contain either 'model' or 'student'")
    model.load_state_dict(state, strict=False)

    metrics = evaluate_pose_loader(model, loader, device)
    print(format_metrics(metrics))
    output_path = Path(args.checkpoint).with_suffix(".metrics.json")
    save_json(metrics, output_path)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
