from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2

from fafa.common import write_jsonl
from fafa.tools.bop_utils import (
    choose_bbox,
    crop_and_resize_with_intrinsics,
    iter_bop_records,
    load_pose_predictions,
    parse_scene_ids,
    perturb_pose,
    read_image_rgb,
    relpath_str,
    save_rgb_png,
    scene_object_id,
)


def _record_to_jsonable_pose(pose):
    return pose.tolist() if pose is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAFA target-sample JSONL from a BOP-style dataset.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root containing scene folders like 000000/")
    parser.add_argument("--output-root", type=str, required=True, help="Directory to write cropped RGB targets")
    parser.add_argument("--output-index", type=str, required=True, help="Output JSONL index path")
    parser.add_argument("--mesh-points", type=str, required=True, help="Path to mesh point cloud .npy")
    parser.add_argument("--symmetric", action="store_true", help="Mark the object as symmetric for ADD-S")
    parser.add_argument("--scene-ids", type=str, default="", help="Optional comma-separated scene ids")
    parser.add_argument("--image-subdir", type=str, default="rgb")
    parser.add_argument("--mask-subdir", type=str, default="auto", help="mask or mask_visib, or auto to choose whichever exists")
    parser.add_argument("--depth-subdir", type=str, default="depth")
    parser.add_argument("--translation-scale", type=float, default=0.001, help="Convert pose translation units to meters")
    parser.add_argument("--crop-source", type=str, default="bbox_visib", choices=["bbox_visib", "bbox_obj", "mask_bbox"])
    parser.add_argument("--pad-scale", type=float, default=1.3)
    parser.add_argument("--no-square", action="store_true", help="Do not force square crops")
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--min-size", type=int, default=32)
    parser.add_argument(
        "--init-pose-source",
        type=str,
        default="gt",
        choices=["gt", "predictions"],
        help="How to populate init_pose",
    )
    parser.add_argument("--init-pose-jsonl", type=str, default="", help="JSONL predictions when --init-pose-source=predictions")
    parser.add_argument("--init-rot-noise-deg", type=float, default=0.0, help="Optional perturbation applied to init_pose")
    parser.add_argument("--init-trans-noise-m", type=float, default=0.0, help="Optional perturbation applied to init_pose")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-gt-pose", action="store_true", help="Omit gt_pose from the output index")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index_base = Path(args.output_index).resolve().parent
    index_base.mkdir(parents=True, exist_ok=True)

    pred_map = {}
    if args.init_pose_source == "predictions":
        if not args.init_pose_jsonl:
            raise ValueError("--init-pose-jsonl is required when --init-pose-source=predictions")
        pred_map = load_pose_predictions(args.init_pose_jsonl)

    records: List[Dict[str, Any]] = []
    scene_ids = parse_scene_ids(args.scene_ids)
    for rec in iter_bop_records(
        args.dataset_root,
        scene_ids=scene_ids,
        image_subdir=args.image_subdir,
        mask_subdir=args.mask_subdir,
        depth_subdir=args.depth_subdir,
        translation_scale=float(args.translation_scale),
    ):
        if rec.pose_m2c is None:
            continue
        rgb = read_image_rgb(rec.rgb_path)
        bbox = choose_bbox(rec, crop_source=args.crop_source)
        crop = crop_and_resize_with_intrinsics(
            rgb,
            rec.k,
            bbox,
            output_size=(int(args.output_size), int(args.output_size)),
            pad_scale=float(args.pad_scale),
            square=not bool(args.no_square),
            min_size=int(args.min_size),
            interpolation=cv2.INTER_LINEAR,
        )

        sample_id = scene_object_id(rec.scene_id, rec.frame_id, rec.inst_id)
        rgb_out = output_root / "images" / f"{sample_id}.png"
        save_rgb_png(rgb_out, crop.image)

        gt_pose = rec.pose_m2c
        if args.init_pose_source == "gt":
            init_pose = gt_pose.copy()
        else:
            key = (rec.scene_id, rec.frame_id, rec.inst_id)
            if key not in pred_map:
                raise KeyError(f"Missing init pose for {key} in {args.init_pose_jsonl}")
            init_pose = pred_map[key].copy()

        if float(args.init_rot_noise_deg) > 0 or float(args.init_trans_noise_m) > 0:
            seed = int(args.seed) + rec.scene_id * 10_000_000 + rec.frame_id * 10 + rec.inst_id
            init_pose = perturb_pose(
                init_pose,
                rot_deg=float(args.init_rot_noise_deg),
                trans_m=float(args.init_trans_noise_m),
                seed=seed,
            )

        record: Dict[str, Any] = {
            "id": sample_id,
            "scene_id": rec.scene_id,
            "frame_id": rec.frame_id,
            "inst_id": rec.inst_id,
            "object_id": rec.object_id,
            "image": os.path.relpath(rgb_out, index_base),
            "K": crop.k.tolist(),
            "init_pose": _record_to_jsonable_pose(init_pose),
            "mesh_points": os.path.relpath(Path(args.mesh_points).resolve(), index_base),
            "symmetric": bool(args.symmetric),
            "bbox_source": args.crop_source,
            "crop_window_xyxy": [crop.window.x0, crop.window.y0, crop.window.x1, crop.window.y1],
        }
        if not bool(args.drop_gt_pose):
            record["gt_pose"] = _record_to_jsonable_pose(gt_pose)
        records.append(record)

    write_jsonl(records, args.output_index)
    print(f"Wrote {len(records)} target records to {args.output_index}")


if __name__ == "__main__":
    main()
