from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch
from tqdm import tqdm

from fafa.common import AverageMeter, device_from_config, format_metrics, freeze_bn, load_config, seed_everything
from fafa.data.augment import noisy_student_augment
from fafa.data.dataset import build_dataloaders
from fafa.debug_utils import collect_model_debug_stats, dump_anomaly, get_debug_value, should_fail_on_flow_anomaly
from fafa.eval_utils import evaluate_pose_loader
from fafa.losses.core import self_supervised_loss
from fafa.modeling.ema import update_ema
from fafa.train_utils import build_model_from_cfg, build_optimizer_and_scheduler, move_batch_to_device


def save_selfsup_checkpoint(
    path: str,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_score: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_score": float(best_score),
        },
        path,
    )


def load_selfsup_checkpoint(
    path: str,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    map_location: str | torch.device,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    student.load_state_dict(ckpt["student"], strict=True)
    teacher.load_state_dict(ckpt["teacher"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


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
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    cfg: Any,
    epoch: int,
) -> Dict[str, float]:
    student.train()
    teacher.eval()

    loss_meter = AverageMeter("loss")
    pose_meter = AverageMeter("pose")
    flow_meter = AverageMeter("flow")
    student_flow_abs_max_meter = AverageMeter("student_flow_abs_max_px")
    teacher_valid_ratio_meter = AverageMeter("teacher_valid_ratio")

    gamma1 = float(cfg.loss.gamma1)
    gamma2 = float(cfg.loss.gamma2)
    gamma3 = float(cfg.loss.gamma3)
    gamma4 = float(cfg.loss.gamma4)
    min_teacher_flow_valid_ratio = float(get_debug_value(cfg, "min_teacher_flow_valid_ratio", 0.0))

    for step, batch in enumerate(tqdm(loader, desc="selfsup", leave=False)):
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            teacher_out = teacher(
                real_image=batch["image"],
                synth_images=batch["context_images"],
                synth_depths=batch["context_depths"],
                synth_masks=batch["context_masks"],
                synth_poses=batch["context_poses"],
                init_pose=batch["init_pose"],
                k=batch["K"],
                context_ks=batch.get("context_ks"),
            )
        teacher_stats = maybe_handle_anomaly(
            cfg,
            stage="selfsup_teacher",
            epoch=epoch,
            step=step,
            batch=batch,
            out=teacher_out,
            extra={"kind": "teacher"},
        )
        teacher_valid_ratio = float(teacher_out["flow_valid"].detach().float().mean().cpu())
        if min_teacher_flow_valid_ratio > 0 and teacher_valid_ratio < min_teacher_flow_valid_ratio:
            dump_dir = str(get_debug_value(cfg, "dump_dir", os.path.join(str(cfg.train.output_dir), "debug")))
            case_dir = dump_anomaly(
                dump_dir,
                stage="selfsup_teacher_valid",
                epoch=epoch,
                step=step,
                reason=f"teacher_flow_valid_ratio={teacher_valid_ratio:.6f} < {min_teacher_flow_valid_ratio:.6f}",
                batch=batch,
                out=teacher_out,
                extra={"teacher_stats": teacher_stats},
            )
            raise RuntimeError(
                f"Teacher flow_valid ratio too low ({teacher_valid_ratio:.6f}). Dumped anomaly package to: {case_dir}"
            )

        noisy_image = noisy_student_augment(batch["image"])
        student_out = student(
            real_image=noisy_image,
            synth_images=batch["context_images"],
            synth_depths=batch["context_depths"],
            synth_masks=batch["context_masks"],
            synth_poses=batch["context_poses"],
            init_pose=batch["init_pose"],
            k=batch["K"],
            context_ks=batch.get("context_ks"),
        )
        student_stats = maybe_handle_anomaly(
            cfg,
            stage="selfsup_student",
            epoch=epoch,
            step=step,
            batch=batch,
            out=student_out,
            extra={"kind": "student"},
        )
        total, stats = self_supervised_loss(
            student_out=student_out,
            teacher_out=teacher_out,
            real_image=batch["image"],
            synth_masks=batch["context_masks"],
            model_points=batch["mesh_points"],
            symmetric=batch["symmetric"],
            gamma1=gamma1,
            gamma2=gamma2,
            gamma3=gamma3,
            gamma4=gamma4,
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if float(cfg.train.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.train.grad_clip))
        optimizer.step()
        scheduler.step()
        update_ema(teacher, student, momentum=float(cfg.train.ema_momentum))

        bs = int(batch["image"].shape[0])
        loss_meter.update(stats["loss_total"], bs)
        pose_meter.update(stats["loss_pose"], bs)
        flow_meter.update(stats["loss_flow"], bs)
        student_flow_abs_max_meter.update(float(student_stats.get("flow_abs_max_px", 0.0)), bs)
        teacher_valid_ratio_meter.update(teacher_valid_ratio, bs)

    return {
        "loss": loss_meter.avg,
        "pose": pose_meter.avg,
        "flow": flow_meter.avg,
        "student_flow_abs_max_px": student_flow_abs_max_meter.avg,
        "teacher_valid_ratio": teacher_valid_ratio_meter.avg,
        "lr": scheduler.get_last_lr()[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.runtime.seed))
    device = device_from_config(cfg)

    train_loader, val_loader = build_dataloaders(cfg, mode="selfsup")
    student = build_model_from_cfg(cfg, teacher=False).to(device)
    teacher = build_model_from_cfg(cfg, teacher=True).to(device)

    if getattr(cfg.train, "pretrain_checkpoint", None):
        ckpt = torch.load(cfg.train.pretrain_checkpoint, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt.get("student")
        if state is None:
            raise ValueError("Could not find model weights in pretrain_checkpoint")
        student.load_state_dict(state, strict=False)
        teacher.load_state_dict(student.state_dict(), strict=False)
    else:
        teacher.load_state_dict(student.state_dict(), strict=False)

    if bool(cfg.train.freeze_bn):
        freeze_bn(student)
        freeze_bn(teacher)

    optimizer, scheduler = build_optimizer_and_scheduler(student, cfg, steps_per_epoch=len(train_loader))

    start_epoch = 0
    best_score = float("-inf")
    if getattr(cfg.train, "resume_from", None):
        ckpt = load_selfsup_checkpoint(cfg.train.resume_from, student, teacher, optimizer, scheduler, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", best_score))

    best_metric_name = str(cfg.train.best_metric)
    output_dir = str(cfg.train.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        train_stats = train_one_epoch(student, teacher, train_loader, optimizer, scheduler, device, cfg, epoch)
        msg = f"[selfsup][epoch {epoch}] {format_metrics(train_stats)}"
        if val_loader is not None:
            val_stats = evaluate_pose_loader(student, val_loader, device)
            msg += " | val: " + format_metrics(val_stats)
            score = float(val_stats[best_metric_name])
            if score > best_score:
                best_score = score
                save_selfsup_checkpoint(
                    os.path.join(output_dir, "best_selfsup.pt"),
                    student,
                    teacher,
                    optimizer,
                    scheduler,
                    epoch,
                    best_score,
                )
        print(msg)
        save_selfsup_checkpoint(
            os.path.join(output_dir, "latest_selfsup.pt"),
            student,
            teacher,
            optimizer,
            scheduler,
            epoch,
            best_score,
        )


if __name__ == "__main__":
    main()
