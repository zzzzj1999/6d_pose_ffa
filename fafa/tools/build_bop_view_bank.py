from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from fafa.common import write_jsonl
from fafa.tools.bop_utils import (
    choose_bbox,
    crop_and_resize_with_intrinsics,
    iter_bop_records,
    parse_scene_ids,
    read_depth,
    read_image_rgb,
    read_mask,
    save_depth_npy,
    save_mask_png,
    save_rgb_png,
    scene_object_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAFA synthetic context-view bank from a BOP-style dataset.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root containing scene folders like 000000/")
    parser.add_argument("--output-root", type=str, required=True, help="Directory to write cropped RGB/depth/mask views")
    parser.add_argument("--output-index", type=str, required=True, help="Output JSONL view-bank path")
    parser.add_argument("--scene-ids", type=str, default="", help="Optional comma-separated scene ids")
    parser.add_argument("--image-subdir", type=str, default="rgb")
    parser.add_argument("--mask-subdir", type=str, default="auto", help="mask or mask_visib, or auto to choose whichever exists")
    parser.add_argument("--depth-subdir", type=str, default="depth")
    parser.add_argument("--translation-scale", type=float, default=0.001, help="Convert pose translation units to meters")
    parser.add_argument("--raw-depth-to-meter", type=float, default=None, help="Override raw depth -> meter conversion. Defaults to scene_camera depth_scale * translation_scale when omitted.")
    parser.add_argument("--crop-source", type=str, default="mask_bbox", choices=["bbox_visib", "bbox_obj", "mask_bbox"])
    parser.add_argument("--pad-scale", type=float, default=1.3)
    parser.add_argument("--no-square", action="store_true", help="Do not force square crops")
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--min-size", type=int, default=32)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index_base = Path(args.output_index).resolve().parent
    index_base.mkdir(parents=True, exist_ok=True)

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
        if rec.mask_path is None:
            raise FileNotFoundError(
                f"Synthetic view bank requires masks. Missing mask for scene={rec.scene_id}, frame={rec.frame_id}, inst={rec.inst_id}"
            )
        if rec.depth_path is None:
            raise FileNotFoundError(
                f"Synthetic view bank requires depth. Missing depth for scene={rec.scene_id}, frame={rec.frame_id}, inst={rec.inst_id}"
            )

        rgb = read_image_rgb(rec.rgb_path)
        mask = read_mask(rec.mask_path)
        depth_factor_m = rec.depth_factor_m if args.raw_depth_to_meter is None else float(args.raw_depth_to_meter)
        depth_m = read_depth(rec.depth_path, raw_depth_to_meter=float(depth_factor_m))
        bbox = choose_bbox(rec, crop_source=args.crop_source)

        rgb_crop = crop_and_resize_with_intrinsics(
            rgb,
            rec.k,
            bbox,
            output_size=(int(args.output_size), int(args.output_size)),
            pad_scale=float(args.pad_scale),
            square=not bool(args.no_square),
            min_size=int(args.min_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_crop = crop_and_resize_with_intrinsics(
            mask,
            rec.k,
            bbox,
            output_size=(int(args.output_size), int(args.output_size)),
            pad_scale=float(args.pad_scale),
            square=not bool(args.no_square),
            min_size=int(args.min_size),
            interpolation=cv2.INTER_NEAREST,
        )
        depth_crop = crop_and_resize_with_intrinsics(
            depth_m,
            rec.k,
            bbox,
            output_size=(int(args.output_size), int(args.output_size)),
            pad_scale=float(args.pad_scale),
            square=not bool(args.no_square),
            min_size=int(args.min_size),
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

        record: Dict[str, Any] = {
            "id": sample_id,
            "scene_id": rec.scene_id,
            "frame_id": rec.frame_id,
            "inst_id": rec.inst_id,
            "object_id": rec.object_id,
            "image": os.path.relpath(rgb_out, index_base),
            "depth": os.path.relpath(depth_out, index_base),
            "mask": os.path.relpath(mask_out, index_base),
            "pose": rec.pose_m2c.tolist(),
            "K": rgb_crop.k.tolist(),
            "bbox_source": args.crop_source,
            "crop_window_xyxy": [rgb_crop.window.x0, rgb_crop.window.y0, rgb_crop.window.x1, rgb_crop.window.y1],
        }
        records.append(record)

    write_jsonl(records, args.output_index)
    print(f"Wrote {len(records)} view-bank records to {args.output_index}")


if __name__ == "__main__":
    main()
