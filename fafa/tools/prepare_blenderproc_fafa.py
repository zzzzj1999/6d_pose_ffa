from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from fafa.common import ensure_dir, write_jsonl
from fafa.tools.bop_utils import (
    choose_bbox,
    crop_and_resize_with_intrinsics,
    iter_bop_records,
    load_pose_predictions,
    parse_scene_ids,
    perturb_pose,
    read_depth,
    read_image_rgb,
    read_mask,
    relpath_str,
    resolve_bop_scene_root,
    save_depth_npy,
    save_mask_png,
    save_rgb_png,
    scene_object_id,
)


def _record_to_jsonable_pose(pose: np.ndarray | None) -> Any:
    return None if pose is None else np.asarray(pose, dtype=np.float32).tolist()


def load_pose(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape == (3, 4):
        arr = np.concatenate([arr, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected pose to be 3x4 or 4x4, got {arr.shape}")
    return arr


def rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3] @ b[:3, :3].T
    cos = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def translation_distance_m(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def attach_context_records(
    samples: Sequence[Dict[str, Any]],
    bank: Sequence[Dict[str, Any]],
    *,
    n_context: int = 4,
    rot_weight: float = 1.0,
    trans_weight: float = 100.0,
    exclude_same_id: bool = False,
    exclude_same_frame: bool = False,
    min_rot_distance_deg: float = 0.0,
    min_trans_distance_m: float = 0.0,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sample in samples:
        init_pose = load_pose(sample["init_pose"])
        object_id = sample.get("object_id")
        sample_id = sample.get("id")

        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for view in bank:
            if object_id is not None and view.get("object_id") is not None and view.get("object_id") != object_id:
                continue
            if exclude_same_id and sample_id is not None and sample_id == view.get("id"):
                continue
            if exclude_same_frame:
                if (
                    sample.get("scene_id") == view.get("scene_id")
                    and sample.get("frame_id") == view.get("frame_id")
                    and sample.get("inst_id", 0) == view.get("inst_id", 0)
                ):
                    continue
            pose = load_pose(view["pose"])
            rot_dist = rotation_distance_deg(init_pose, pose)
            trans_dist = translation_distance_m(init_pose, pose)
            if rot_dist < min_rot_distance_deg:
                continue
            if trans_dist < min_trans_distance_m:
                continue
            score = rot_weight * rot_dist + trans_weight * trans_dist
            candidates.append((score, view))

        if len(candidates) < n_context:
            raise ValueError(
                f"Sample {sample.get('id', '<unknown>')} only has {len(candidates)} candidate context views. "
                f"Need at least {n_context}."
            )

        candidates.sort(key=lambda x: x[0])
        context: List[Dict[str, Any]] = []
        for _, view in candidates[:n_context]:
            entry = {
                "image": view["image"],
                "depth": view["depth"],
                "mask": view["mask"],
                "pose": load_pose(view["pose"]).tolist(),
            }
            if "K" in view:
                entry["K"] = view["K"]
            context.append(entry)
        rec = dict(sample)
        rec["context"] = context
        out.append(rec)
    return out


def split_records_by_scene(
    records: Sequence[Dict[str, Any]],
    *,
    val_scene_ids: Optional[Sequence[int]] = None,
    val_ratio: float = 0.1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records:
        return [], []
    scenes = sorted({int(r["scene_id"]) for r in records if "scene_id" in r})
    if val_scene_ids:
        val_set = {int(s) for s in val_scene_ids}
    else:
        if len(scenes) == 1 or val_ratio <= 0:
            val_set = set()
        else:
            num_val = max(1, int(round(len(scenes) * float(val_ratio))))
            num_val = min(num_val, max(1, len(scenes) - 1))
            val_set = set(scenes[-num_val:])
    train, val = [], []
    for rec in records:
        scene_id = int(rec.get("scene_id", -1))
        if scene_id in val_set:
            val.append(rec)
        else:
            train.append(rec)
    return train, val


def split_records_by_frame(
    records: Sequence[Dict[str, Any]],
    *,
    val_ratio: float = 0.1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records:
        return [], []
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(int(rec.get("scene_id", -1)), []).append(rec)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for scene_id, scene_records in grouped.items():
        frame_ids = sorted({int(r.get("frame_id", -1)) for r in scene_records})
        if len(frame_ids) <= 1 or val_ratio <= 0:
            train.extend(scene_records)
            continue
        num_val = max(1, int(round(len(frame_ids) * float(val_ratio))))
        num_val = min(num_val, max(1, len(frame_ids) - 1))
        val_frame_set = set(frame_ids[-num_val:])
        for rec in scene_records:
            if int(rec.get("frame_id", -1)) in val_frame_set:
                val.append(rec)
            else:
                train.append(rec)
    return train, val


def build_target_records(
    *,
    dataset_root: str,
    output_root: Path,
    index_base: Path,
    mesh_points: str,
    symmetric: bool,
    scene_ids: Optional[Sequence[int]],
    image_subdir: str,
    mask_subdir: str,
    depth_subdir: str,
    translation_scale: float,
    crop_source: str,
    pad_scale: float,
    square: bool,
    output_size: int,
    min_size: int,
    init_pose_source: str,
    init_pose_jsonl: Optional[str],
    init_rot_noise_deg: float,
    init_trans_noise_m: float,
    seed: int,
    drop_gt_pose: bool,
) -> List[Dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    pred_map = {}
    if init_pose_source == "predictions":
        if not init_pose_jsonl:
            raise ValueError("init_pose_jsonl is required when init_pose_source=predictions")
        pred_map = load_pose_predictions(init_pose_jsonl)

    records: List[Dict[str, Any]] = []
    for rec in iter_bop_records(
        dataset_root,
        scene_ids=scene_ids,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        depth_subdir=depth_subdir,
        translation_scale=translation_scale,
    ):
        if rec.pose_m2c is None:
            continue
        rgb = read_image_rgb(rec.rgb_path)
        bbox = choose_bbox(rec, crop_source=crop_source)
        crop = crop_and_resize_with_intrinsics(
            rgb,
            rec.k,
            bbox,
            output_size=(output_size, output_size),
            pad_scale=pad_scale,
            square=square,
            min_size=min_size,
            interpolation=cv2.INTER_LINEAR,
        )

        sample_id = scene_object_id(rec.scene_id, rec.frame_id, rec.inst_id)
        rgb_out = output_root / "images" / f"{sample_id}.png"
        save_rgb_png(rgb_out, crop.image)

        gt_pose = rec.pose_m2c
        if init_pose_source == "gt":
            init_pose = gt_pose.copy()
        else:
            key = (rec.scene_id, rec.frame_id, rec.inst_id)
            if key not in pred_map:
                raise KeyError(f"Missing init pose for {key} in {init_pose_jsonl}")
            init_pose = pred_map[key].copy()

        if init_rot_noise_deg > 0 or init_trans_noise_m > 0:
            sample_seed = int(seed) + rec.scene_id * 10_000_000 + rec.frame_id * 10 + rec.inst_id
            init_pose = perturb_pose(
                init_pose,
                rot_deg=init_rot_noise_deg,
                trans_m=init_trans_noise_m,
                seed=sample_seed,
            )

        record: Dict[str, Any] = {
            "id": sample_id,
            "scene_id": rec.scene_id,
            "frame_id": rec.frame_id,
            "inst_id": rec.inst_id,
            "object_id": rec.object_id,
            "image": relpath_str(rgb_out, index_base),
            "K": crop.k.tolist(),
            "init_pose": _record_to_jsonable_pose(init_pose),
            "mesh_points": relpath_str(Path(mesh_points), index_base),
            "symmetric": bool(symmetric),
            "bbox_source": crop_source,
            "crop_window_xyxy": [crop.window.x0, crop.window.y0, crop.window.x1, crop.window.y1],
        }
        if not drop_gt_pose:
            record["gt_pose"] = _record_to_jsonable_pose(gt_pose)
        records.append(record)
    return records


def build_view_bank_records(
    *,
    dataset_root: str,
    output_root: Path,
    index_base: Path,
    scene_ids: Optional[Sequence[int]],
    image_subdir: str,
    mask_subdir: str,
    depth_subdir: str,
    translation_scale: float,
    raw_depth_to_meter: Optional[float],
    crop_source: str,
    pad_scale: float,
    square: bool,
    output_size: int,
    min_size: int,
) -> List[Dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for rec in iter_bop_records(
        dataset_root,
        scene_ids=scene_ids,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        depth_subdir=depth_subdir,
        translation_scale=translation_scale,
    ):
        if rec.pose_m2c is None:
            continue
        if rec.mask_path is None:
            raise FileNotFoundError(
                f"Missing mask for scene={rec.scene_id}, frame={rec.frame_id}, inst={rec.inst_id}."
            )
        if rec.depth_path is None:
            raise FileNotFoundError(
                f"Missing depth for scene={rec.scene_id}, frame={rec.frame_id}, inst={rec.inst_id}."
            )

        rgb = read_image_rgb(rec.rgb_path)
        mask = read_mask(rec.mask_path)
        depth_factor_m = rec.depth_factor_m if raw_depth_to_meter is None else raw_depth_to_meter
        depth_m = read_depth(rec.depth_path, raw_depth_to_meter=float(depth_factor_m))
        bbox = choose_bbox(rec, crop_source=crop_source)

        rgb_crop = crop_and_resize_with_intrinsics(
            rgb,
            rec.k,
            bbox,
            output_size=(output_size, output_size),
            pad_scale=pad_scale,
            square=square,
            min_size=min_size,
            interpolation=cv2.INTER_LINEAR,
        )
        mask_crop = crop_and_resize_with_intrinsics(
            mask,
            rec.k,
            bbox,
            output_size=(output_size, output_size),
            pad_scale=pad_scale,
            square=square,
            min_size=min_size,
            interpolation=cv2.INTER_NEAREST,
        )
        depth_crop = crop_and_resize_with_intrinsics(
            depth_m,
            rec.k,
            bbox,
            output_size=(output_size, output_size),
            pad_scale=pad_scale,
            square=square,
            min_size=min_size,
            interpolation=cv2.INTER_NEAREST,
        )
        depth_crop_image = np.asarray(depth_crop.image, dtype=np.float32).copy()
        depth_crop_image[mask_crop.image <= 0] = 0.0

        sample_id = scene_object_id(rec.scene_id, rec.frame_id, rec.inst_id)
        rgb_out = output_root / "images" / f"{sample_id}.png"
        mask_out = output_root / "masks" / f"{sample_id}.png"
        depth_out = output_root / "depth" / f"{sample_id}.npy"
        save_rgb_png(rgb_out, rgb_crop.image)
        save_mask_png(mask_out, mask_crop.image)
        save_depth_npy(depth_out, depth_crop_image)

        records.append(
            {
                "id": sample_id,
                "scene_id": rec.scene_id,
                "frame_id": rec.frame_id,
                "inst_id": rec.inst_id,
                "object_id": rec.object_id,
                "image": relpath_str(rgb_out, index_base),
                "depth": relpath_str(depth_out, index_base),
                "mask": relpath_str(mask_out, index_base),
                "pose": _record_to_jsonable_pose(rec.pose_m2c),
                "K": rgb_crop.k.tolist(),
                "bbox_source": crop_source,
                "crop_window_xyxy": [rgb_crop.window.x0, rgb_crop.window.y0, rgb_crop.window.x1, rgb_crop.window.y1],
            }
        )
    return records


def write_style_index_from_records(records: Iterable[Dict[str, Any]], path: Path) -> None:
    style_records = []
    seen = set()
    for rec in records:
        img = rec["image"]
        if img in seen:
            continue
        seen.add(img)
        style_records.append({"image": img})
    write_jsonl(style_records, path)


def write_training_configs(
    config_dir: Path,
    *,
    pretrain_train_index: Path,
    pretrain_val_index: Path,
    selfsup_train_index: Optional[Path],
    selfsup_val_index: Optional[Path],
    eval_index: Optional[Path],
    real_style_index: Optional[Path],
    output_root: Path,
) -> None:
    ensure_dir(config_dir)

    common_model = {
        "feature_dim": 128,
        "hidden_dim": 128,
        "student_outer_iters": 1,
        "teacher_outer_iters": 2,
        "translation_scale": 0.02,
        "geometric_consistency_px": 3.0,
        "max_disp_feat": 32.0,
        "mask_prior_flow": True,
        "mask_pred_flow": True,
        "masked_pose_pooling": True,
    }
    common_data = {
        "real_style_index": str(real_style_index) if real_style_index else None,
        "n_context": 4,
        "zero_background_depth": True,
        "max_context_depth_m": None,
    }
    common_debug = {
        "enabled": True,
        "warn_abs_flow_px": 500.0,
        "max_abs_flow_px": 10000.0,
        "fail_on_anomaly": True,
    }

    pretrain_cfg = {
        "runtime": {
            "seed": 42,
            "device": "cuda:0",
            "batch_size": 16,
            "num_workers": 4,
            "pin_memory": True,
        },
        "data": {
            **common_data,
            "pretrain_train_index": str(pretrain_train_index),
            "pretrain_val_index": str(pretrain_val_index),
        },
        "model": dict(common_model),
        "fft": {
            "enabled": bool(real_style_index),
            "delta0": 0.5,
            "beta": 1.0,
        },
        "loss": {
            "pretrain_pose_weight": 1.0,
            "pretrain_flow_weight": 1.0e-4,
            "gamma1": 0.1,
            "gamma2": 0.1,
            "gamma3": 10.0,
            "gamma4": 10.0,
        },
        "train": {
            "epochs": 20,
            "max_lr": 1.0e-4,
            "weight_decay": 1.0e-4,
            "pct_start": 0.1,
            "anneal_strategy": "cos",
            "div_factor": 25.0,
            "final_div_factor": 1000.0,
            "grad_clip": 1.0,
            "output_dir": str(output_root / "pretrain"),
            "best_metric": "5cm",
            "resume_from": None,
            "pretrain_checkpoint": None,
            "ema_momentum": 0.999,
            "freeze_bn": False,
            "pose_only_epochs": 5,
            "flow_ramp_epochs": 10,
            "debug": {
                **common_debug,
                "dump_dir": str(output_root / "pretrain" / "debug"),
            },
        },
    }
    with open(config_dir / "blenderproc_pretrain.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(pretrain_cfg, f, sort_keys=False, allow_unicode=True)

    if selfsup_train_index is not None and selfsup_val_index is not None:
        selfsup_cfg = {
            "runtime": {
                "seed": 42,
                "device": "cuda:0",
                "batch_size": 16,
                "num_workers": 4,
                "pin_memory": True,
            },
            "data": {
                **common_data,
                "selfsup_train_index": str(selfsup_train_index),
                "selfsup_val_index": str(selfsup_val_index),
            },
            "model": dict(common_model),
            "fft": {
                "enabled": False,
                "delta0": 0.5,
                "beta": 1.0,
            },
            "loss": {
                "pretrain_pose_weight": 1.0,
                "pretrain_flow_weight": 1.0e-4,
                "gamma1": 0.1,
                "gamma2": 0.1,
                "gamma3": 10.0,
                "gamma4": 10.0,
            },
            "train": {
                "epochs": 10,
                "max_lr": 1.0e-4,
                "weight_decay": 1.0e-4,
                "pct_start": 0.1,
                "anneal_strategy": "cos",
                "div_factor": 25.0,
                "final_div_factor": 1000.0,
                "grad_clip": 1.0,
                "output_dir": str(output_root / "selfsup"),
                "best_metric": "5cm",
                "resume_from": None,
                "pretrain_checkpoint": str(output_root / "pretrain" / "best_pretrain.pt"),
                "ema_momentum": 0.999,
                "freeze_bn": True,
                "debug": {
                    **common_debug,
                    "dump_dir": str(output_root / "selfsup" / "debug"),
                    "min_teacher_flow_valid_ratio": 0.005,
                },
            },
        }
        with open(config_dir / "blenderproc_selfsup.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(selfsup_cfg, f, sort_keys=False, allow_unicode=True)

    if eval_index is not None:
        eval_cfg = {
            "runtime": {
                "seed": 42,
                "device": "cuda:0",
                "batch_size": 16,
                "num_workers": 4,
                "pin_memory": True,
            },
            "data": {
                **common_data,
                "eval_index": str(eval_index),
            },
            "model": dict(common_model),
            "fft": {
                "enabled": False,
                "delta0": 0.5,
                "beta": 1.0,
            },
            "loss": {
                "pretrain_pose_weight": 1.0,
                "pretrain_flow_weight": 1.0e-4,
                "gamma1": 0.1,
                "gamma2": 0.1,
                "gamma3": 10.0,
                "gamma4": 10.0,
            },
            "train": {
                "epochs": 1,
                "max_lr": 1.0e-4,
                "weight_decay": 1.0e-4,
                "pct_start": 0.1,
                "anneal_strategy": "cos",
                "div_factor": 25.0,
                "final_div_factor": 1000.0,
                "grad_clip": 1.0,
                "output_dir": str(output_root / "eval"),
                "best_metric": "5cm",
                "resume_from": None,
                "pretrain_checkpoint": None,
                "ema_momentum": 0.999,
                "freeze_bn": True,
            },
        }
        with open(config_dir / "blenderproc_eval.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(eval_cfg, f, sort_keys=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FAFA pretrain/self-supervised indices directly from BlenderProc/BOP-style synthetic data."
    )
    parser.add_argument("--synth-root", type=str, required=True, help="BlenderProc synthetic root. Can point to output_root, dataset_name, or train_pbr.")
    parser.add_argument("--real-root", type=str, default="", help="Optional real BOP root for self-supervision/evaluation.")
    parser.add_argument("--workdir", type=str, required=True, help="Directory to write prepared crops, JSONL indices, and configs.")
    parser.add_argument("--mesh-points", type=str, required=True, help="Path to object point cloud .npy used for pose loss/eval.")
    parser.add_argument("--symmetric", action="store_true", help="Use ADD-S style metrics and symmetric point matching.")

    parser.add_argument("--synth-scene-ids", type=str, default="", help="Optional synthetic scene ids to include.")
    parser.add_argument("--synth-val-scene-ids", type=str, default="", help="Optional synthetic validation scene ids.")
    parser.add_argument("--real-train-scene-ids", type=str, default="", help="Optional real scene ids for self-supervised training.")
    parser.add_argument("--real-eval-scene-ids", type=str, default="", help="Optional real scene ids for validation/evaluation.")
    parser.add_argument("--synth-val-ratio", type=float, default=0.1)
    parser.add_argument("--real-eval-ratio", type=float, default=0.1)

    parser.add_argument("--image-subdir", type=str, default="rgb")
    parser.add_argument("--mask-subdir", type=str, default="auto")
    parser.add_argument("--depth-subdir", type=str, default="depth")
    parser.add_argument("--translation-scale", type=float, default=0.001, help="Pose translation unit -> meter, usually 0.001 for mm.")
    parser.add_argument("--raw-depth-to-meter", type=float, default=None, help="Override raw depth -> meter conversion. Defaults to scene_camera depth_scale * translation_scale.")
    parser.add_argument("--crop-source-synth", type=str, default="mask_bbox", choices=["bbox_visib", "bbox_obj", "mask_bbox"])
    parser.add_argument("--crop-source-real", type=str, default="bbox_visib", choices=["bbox_visib", "bbox_obj", "mask_bbox"])
    parser.add_argument("--pad-scale", type=float, default=1.3)
    parser.add_argument("--no-square", action="store_true")
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--min-size", type=int, default=32)

    parser.add_argument("--synth-init-pose-source", type=str, default="gt", choices=["gt", "predictions"])
    parser.add_argument("--synth-init-pose-jsonl", type=str, default="")
    parser.add_argument("--synth-init-rot-noise-deg", type=float, default=8.0, help="Recommended non-zero noise so pretraining learns refinement.")
    parser.add_argument("--synth-init-trans-noise-m", type=float, default=0.03)

    parser.add_argument("--real-init-pose-source", type=str, default="gt", choices=["gt", "predictions"])
    parser.add_argument("--real-init-pose-jsonl", type=str, default="")
    parser.add_argument("--real-init-rot-noise-deg", type=float, default=8.0)
    parser.add_argument("--real-init-trans-noise-m", type=float, default=0.03)

    parser.add_argument("--n-context", type=int, default=4)
    parser.add_argument("--rot-weight", type=float, default=1.0)
    parser.add_argument("--trans-weight", type=float, default=100.0)
    parser.add_argument("--allow-self-context", action="store_true", help="By default synthetic pretraining excludes the identical rendered view. Set this flag to allow self-context.")
    parser.add_argument("--min-context-rot-distance-deg", type=float, default=2.0)
    parser.add_argument("--min-context-trans-distance-m", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    synth_root = str(resolve_bop_scene_root(args.synth_root))
    real_root = str(resolve_bop_scene_root(args.real_root)) if args.real_root else ""

    workdir = Path(args.workdir).resolve()
    index_dir = workdir / "indices"
    prepared_dir = workdir / "prepared"
    config_dir = workdir / "configs"
    output_dir = workdir / "outputs"
    ensure_dir(index_dir)
    ensure_dir(prepared_dir)
    ensure_dir(config_dir)
    ensure_dir(output_dir)

    synth_scene_ids = parse_scene_ids(args.synth_scene_ids)
    synth_val_scene_ids = parse_scene_ids(args.synth_val_scene_ids)
    real_train_scene_ids = parse_scene_ids(args.real_train_scene_ids)
    real_eval_scene_ids = parse_scene_ids(args.real_eval_scene_ids)
    square = not bool(args.no_square)

    print(f"[INFO] Synthetic scene root: {synth_root}")
    if real_root:
        print(f"[INFO] Real scene root: {real_root}")

    # 1) Synthetic view bank from BlenderProc train_pbr.
    synth_view_bank_records = build_view_bank_records(
        dataset_root=synth_root,
        output_root=prepared_dir / "synth_view_bank",
        index_base=index_dir,
        scene_ids=synth_scene_ids,
        image_subdir=args.image_subdir,
        mask_subdir=args.mask_subdir,
        depth_subdir=args.depth_subdir,
        translation_scale=float(args.translation_scale),
        raw_depth_to_meter=args.raw_depth_to_meter,
        crop_source=args.crop_source_synth,
        pad_scale=float(args.pad_scale),
        square=square,
        output_size=int(args.output_size),
        min_size=int(args.min_size),
    )
    synth_view_bank_index = index_dir / "synth_view_bank.jsonl"
    write_jsonl(synth_view_bank_records, synth_view_bank_index)
    print(f"[DONE] Wrote synthetic view bank: {synth_view_bank_index} ({len(synth_view_bank_records)} views)")

    # 2) Synthetic pretrain targets from the same BlenderProc dataset.
    synth_targets_records = build_target_records(
        dataset_root=synth_root,
        output_root=prepared_dir / "synth_targets",
        index_base=index_dir,
        mesh_points=args.mesh_points,
        symmetric=bool(args.symmetric),
        scene_ids=synth_scene_ids,
        image_subdir=args.image_subdir,
        mask_subdir=args.mask_subdir,
        depth_subdir=args.depth_subdir,
        translation_scale=float(args.translation_scale),
        crop_source=args.crop_source_synth,
        pad_scale=float(args.pad_scale),
        square=square,
        output_size=int(args.output_size),
        min_size=int(args.min_size),
        init_pose_source=args.synth_init_pose_source,
        init_pose_jsonl=args.synth_init_pose_jsonl or None,
        init_rot_noise_deg=float(args.synth_init_rot_noise_deg),
        init_trans_noise_m=float(args.synth_init_trans_noise_m),
        seed=int(args.seed),
        drop_gt_pose=False,
    )
    synth_targets_raw_index = index_dir / "synth_pretrain_targets_raw.jsonl"
    write_jsonl(synth_targets_records, synth_targets_raw_index)

    synth_pretrain_records = attach_context_records(
        synth_targets_records,
        synth_view_bank_records,
        n_context=int(args.n_context),
        rot_weight=float(args.rot_weight),
        trans_weight=float(args.trans_weight),
        exclude_same_id=not bool(args.allow_self_context),
        exclude_same_frame=not bool(args.allow_self_context),
        min_rot_distance_deg=float(args.min_context_rot_distance_deg),
        min_trans_distance_m=float(args.min_context_trans_distance_m),
    )
    synth_pretrain_all = index_dir / "synth_pretrain_all.jsonl"
    write_jsonl(synth_pretrain_records, synth_pretrain_all)

    synth_pretrain_train_records, synth_pretrain_val_records = split_records_by_scene(
        synth_pretrain_records,
        val_scene_ids=synth_val_scene_ids,
        val_ratio=float(args.synth_val_ratio),
    )
    synth_pretrain_train_index = index_dir / "synth_pretrain_train.jsonl"
    synth_pretrain_val_index = index_dir / "synth_pretrain_val.jsonl"
    write_jsonl(synth_pretrain_train_records, synth_pretrain_train_index)
    write_jsonl(synth_pretrain_val_records, synth_pretrain_val_index)
    print(
        f"[DONE] Synthetic pretrain indices: train={len(synth_pretrain_train_records)}, "
        f"val={len(synth_pretrain_val_records)}"
    )

    real_style_index: Optional[Path] = None
    selfsup_train_index: Optional[Path] = None
    selfsup_val_index: Optional[Path] = None
    eval_index: Optional[Path] = None

    if real_root:
        real_all_records = build_target_records(
            dataset_root=real_root,
            output_root=prepared_dir / "real_targets_all",
            index_base=index_dir,
            mesh_points=args.mesh_points,
            symmetric=bool(args.symmetric),
            scene_ids=None,
            image_subdir=args.image_subdir,
            mask_subdir=args.mask_subdir,
            depth_subdir=args.depth_subdir,
            translation_scale=float(args.translation_scale),
            crop_source=args.crop_source_real,
            pad_scale=float(args.pad_scale),
            square=square,
            output_size=int(args.output_size),
            min_size=int(args.min_size),
            init_pose_source=args.real_init_pose_source,
            init_pose_jsonl=args.real_init_pose_jsonl or None,
            init_rot_noise_deg=float(args.real_init_rot_noise_deg),
            init_trans_noise_m=float(args.real_init_trans_noise_m),
            seed=int(args.seed),
            drop_gt_pose=False,
        )
        if not real_all_records:
            raise ValueError(f"No usable real records were found under {real_root}")

        all_real_scene_ids = sorted({int(r["scene_id"]) for r in real_all_records})
        if real_train_scene_ids or real_eval_scene_ids:
            if real_train_scene_ids:
                train_scene_set = {int(s) for s in real_train_scene_ids}
            else:
                train_scene_set = set(all_real_scene_ids)
            if real_eval_scene_ids:
                eval_scene_set = {int(s) for s in real_eval_scene_ids}
            else:
                eval_scene_set = set(all_real_scene_ids) - train_scene_set
            if not train_scene_set:
                train_scene_set = set(all_real_scene_ids) - eval_scene_set
            if not eval_scene_set:
                eval_scene_set = set(all_real_scene_ids) - train_scene_set
            real_train_records_full = [r for r in real_all_records if int(r["scene_id"]) in train_scene_set]
            real_eval_records = [r for r in real_all_records if int(r["scene_id"]) in eval_scene_set]
        else:
            real_train_records_full, real_eval_records = split_records_by_scene(
                real_all_records,
                val_scene_ids=None,
                val_ratio=float(args.real_eval_ratio),
            )

        if not real_train_records_full or not real_eval_records:
            # Fall back to an intra-scene frame split so that one-scene datasets do not
            # silently reuse the same real images for both selfsup train and eval.
            real_train_records_full, real_eval_records = split_records_by_frame(
                real_all_records,
                val_ratio=float(args.real_eval_ratio),
            )
        if not real_train_records_full:
            real_train_records_full = list(real_eval_records)
        if not real_eval_records:
            real_eval_records = list(real_train_records_full)

        real_train_records = []
        for rec in real_train_records_full:
            train_rec = dict(rec)
            train_rec.pop("gt_pose", None)
            real_train_records.append(train_rec)

        real_selfsup_records = attach_context_records(
            real_train_records,
            synth_view_bank_records,
            n_context=int(args.n_context),
            rot_weight=float(args.rot_weight),
            trans_weight=float(args.trans_weight),
            exclude_same_id=False,
            exclude_same_frame=False,
            min_rot_distance_deg=0.0,
            min_trans_distance_m=0.0,
        )
        selfsup_train_index = index_dir / "real_selfsup_train.jsonl"
        write_jsonl(real_selfsup_records, selfsup_train_index)
        print(f"[DONE] Real self-supervised train index: {selfsup_train_index} ({len(real_selfsup_records)} samples)")

        real_eval_attached = attach_context_records(
            real_eval_records,
            synth_view_bank_records,
            n_context=int(args.n_context),
            rot_weight=float(args.rot_weight),
            trans_weight=float(args.trans_weight),
            exclude_same_id=False,
            exclude_same_frame=False,
            min_rot_distance_deg=0.0,
            min_trans_distance_m=0.0,
        )
        selfsup_val_index = index_dir / "real_selfsup_val.jsonl"
        eval_index = index_dir / "real_eval.jsonl"
        write_jsonl(real_eval_attached, selfsup_val_index)
        write_jsonl(real_eval_attached, eval_index)
        print(f"[DONE] Real eval index: {eval_index} ({len(real_eval_attached)} samples)")

        real_style_index = index_dir / "real_style.jsonl"
        write_style_index_from_records(real_all_records, real_style_index)
        print(f"[DONE] Real FFT style index: {real_style_index}")

    write_training_configs(
        config_dir,
        pretrain_train_index=synth_pretrain_train_index,
        pretrain_val_index=synth_pretrain_val_index,
        selfsup_train_index=selfsup_train_index,
        selfsup_val_index=selfsup_val_index,
        eval_index=eval_index,
        real_style_index=real_style_index,
        output_root=output_dir,
    )
    print(f"[DONE] Configs written under: {config_dir}")
    print("[NEXT] Pretrain config:", config_dir / "blenderproc_pretrain.yaml")
    if selfsup_train_index is not None:
        print("[NEXT] Selfsup config:", config_dir / "blenderproc_selfsup.yaml")
        print("[NEXT] Eval config:", config_dir / "blenderproc_eval.yaml")


if __name__ == "__main__":
    main()
