from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch
from tqdm import tqdm

from fafa.common import AverageMeter, device_from_config, format_metrics, load_config, seed_everything
from fafa.data.dataset import build_dataloaders
from fafa.debug_utils import collect_model_debug_stats, dump_anomaly, get_debug_value, should_fail_on_flow_anomaly
from fafa.eval_utils import evaluate_pose_loader
from fafa.fft import fft_mix_augment
from fafa.geometry.projection import shape_constraint_flow_from_depth
from fafa.losses.core import flow_supervision_loss, point_matching_loss
from fafa.train_utils import build_model_from_cfg, build_optimizer_and_scheduler, load_checkpoint, move_batch_to_device, save_checkpoint


def current_pretrain_flow_weight(cfg: Any, epoch: int) -> float:
    final_weight = float(cfg.loss.pretrain_flow_weight)
    pose_only_epochs = int(getattr(cfg.train, "pose_only_epochs", 0))
    flow_ramp_epochs = int(getattr(cfg.train, "flow_ramp_epochs", 0))
    if epoch < pose_only_epochs:
        return 0.0
    if flow_ramp_epochs <= 0:
        return final_weight
    local_epoch = epoch - pose_only_epochs
    if local_epoch >= flow_ramp_epochs:
        return final_weight
    alpha = float(local_epoch + 1) / float(flow_ramp_epochs)
    return final_weight * alpha


def maybe_handle_anomaly(
    cfg: Any,
    *,
    stage: str,
    epoch: int,
    step: int,
    batch: Dict[str, Any],
    out: Dict[str, torch.Tensor],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    should_fail, reason, stats = should_fail_on_flow_anomaly(cfg, out)
    debug_enabled = bool(get_debug_value(cfg, "enabled", True))
    warn_abs_flow_px = float(get_debug_value(cfg, "warn_abs_flow_px", 500.0))
    if debug_enabled and stats.get("flow_abs_max_px", 0.0) > warn_abs_flow_px:
        print(f"[WARN][{stage}] epoch={epoch} step={step} flow_abs_max_px={stats['flow_abs_max_px']:.4f}")
    if should_fail:
        dump_dir = str(get_debug_value(cfg, "dump_dir", os.path.join(str(cfg.train.output_dir), "debug")))
        case_dir = dump_anomaly(
            dump_dir,
            stage=stage,
            epoch=epoch,
            step=step,
            reason=reason,
            batch=batch,
            out=out,
            extra=extra,
        )
        if bool(get_debug_value(cfg, "fail_on_anomaly", True)):
            raise RuntimeError(f"{reason}. Dumped anomaly package to: {case_dir}")
        print(f"[WARN][{stage}] {reason}. Dumped anomaly package to: {case_dir}")
    return stats


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    cfg: Any,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter("loss")
    flow_meter = AverageMeter("flow")
    pose_meter = AverageMeter("pose")
    flow_abs_p90_meter = AverageMeter("flow_abs_p90_px")
    flow_abs_max_meter = AverageMeter("flow_abs_max_px")

    pose_weight = float(cfg.loss.pretrain_pose_weight)
    flow_weight = current_pretrain_flow_weight(cfg, epoch)

    for step, batch in enumerate(tqdm(loader, desc="pretrain", leave=False)):
        batch = move_batch_to_device(batch, device)
        image = batch["image"]
        if bool(cfg.fft.enabled):
            image = fft_mix_augment(
                image,
                batch["style_image"],
                delta0=float(cfg.fft.delta0),
                beta=float(cfg.fft.beta),
            )

        out = model(
            real_image=image,
            synth_images=batch["context_images"],
            synth_depths=batch["context_depths"],
            synth_masks=batch["context_masks"],
            synth_poses=batch["context_poses"],
            init_pose=batch["init_pose"],
            k=batch["K"],
            context_ks=batch.get("context_ks"),
        )

        debug_stats = maybe_handle_anomaly(
            cfg,
            stage="pretrain",
            epoch=epoch,
            step=step,
            batch=batch,
            out=out,
            extra={"flow_weight": flow_weight},
        )

        _, n = batch["context_images"].shape[:2]
        gt_flow_loss = 0.0
        for i in range(n):
            gt_flow_i, gt_valid_i = shape_constraint_flow_from_depth(
                batch["context_depths"][:, i],
                batch["context_poses"][:, i],
                batch["gt_pose"],
                k_src=batch["context_ks"][:, i],
                k_tgt=batch["K"],
                mask_src=batch["context_masks"][:, i],
            )
            gt_valid_i = gt_valid_i * batch["context_masks"][:, i]
            gt_flow_loss = gt_flow_loss + flow_supervision_loss(out["flows"][:, i], gt_flow_i, gt_valid_i)
        gt_flow_loss = gt_flow_loss / n

        pose_loss = point_matching_loss(out["pose"], batch["gt_pose"], batch["mesh_points"], batch["symmetric"])
        total = pose_weight * pose_loss + flow_weight * gt_flow_loss

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if float(cfg.train.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
        optimizer.step()
        scheduler.step()

        bs = int(image.shape[0])
        loss_meter.update(float(total.detach().cpu()), bs)
        flow_meter.update(float(gt_flow_loss.detach().cpu()), bs)
        pose_meter.update(float(pose_loss.detach().cpu()), bs)
        flow_abs_p90_meter.update(float(debug_stats.get("flow_abs_p90_px", 0.0)), bs)
        flow_abs_max_meter.update(float(debug_stats.get("flow_abs_max_px", 0.0)), bs)

    return {
        "loss": loss_meter.avg,
        "flow": flow_meter.avg,
        "pose": pose_meter.avg,
        "flow_w": flow_weight,
        "flow_abs_p90_px": flow_abs_p90_meter.avg,
        "flow_abs_max_px": flow_abs_max_meter.avg,
        "lr": scheduler.get_last_lr()[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.runtime.seed))
    device = device_from_config(cfg)

    train_loader, val_loader = build_dataloaders(cfg, mode="pretrain")
    model = build_model_from_cfg(cfg, teacher=False).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, steps_per_epoch=len(train_loader))

    start_epoch = 0
    best_score = float("-inf")
    if getattr(cfg.train, "resume_from", None):
        ckpt = load_checkpoint(cfg.train.resume_from, model, optimizer, scheduler, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("extra", {}).get("best_score", best_score))

    best_metric_name = str(cfg.train.best_metric)
    output_dir = str(cfg.train.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg, epoch)
        msg = f"[pretrain][epoch {epoch}] {format_metrics(train_stats)}"
        if val_loader is not None:
            val_stats = evaluate_pose_loader(model, val_loader, device)
            msg += " | val: " + format_metrics(val_stats)
            score = float(val_stats[best_metric_name])
            if score > best_score:
                best_score = score
                save_checkpoint(
                    os.path.join(output_dir, "best_pretrain.pt"),
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    extra={"best_score": best_score, "val": val_stats},
                )
        print(msg)
        save_checkpoint(
            os.path.join(output_dir, "latest_pretrain.pt"),
            model,
            optimizer,
            scheduler,
            epoch,
            extra={"best_score": best_score},
        )


if __name__ == "__main__":
    main()
