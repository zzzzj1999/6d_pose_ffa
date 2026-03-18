from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from fafa.common import device_from_config, ensure_dir, load_config, resolve_path, seed_everything
from fafa.data.dataset import PreparedContextPoseDataset
from fafa.geometry.pose import pose_geodesic_distance_deg
from fafa.geometry.metrics import translation_error_cm
from fafa.train_utils import build_model_from_cfg
from fafa.tools.bop_utils import IMAGE_EXTS, intrinsics_from_scene_camera, read_json, resolve_bop_scene_root


def load_checkpoint_weights(model: torch.nn.Module, checkpoint: str, *, teacher: bool, device: torch.device) -> None:
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


def _to_single_batch(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, bool):
            batch[k] = torch.tensor([v], dtype=torch.bool, device=device)
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = v
    return batch


def _find_rgb_file(folder: Path, frame_id: int) -> Path:
    stem = f"{frame_id:06d}"
    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not find RGB for frame {frame_id:06d} in {folder}")


def _load_full_image_and_k(bop_root: str, scene_id: int, frame_id: int, image_subdir: str = "rgb") -> tuple[np.ndarray, np.ndarray]:
    root = resolve_bop_scene_root(bop_root)
    scene_dir = root / f"{int(scene_id):06d}"
    scene_camera = read_json(scene_dir / "scene_camera.json")
    entry = scene_camera.get(str(int(frame_id)), scene_camera.get(f"{int(frame_id):06d}"))
    if entry is None:
        raise KeyError(f"Missing scene_camera entry for scene={scene_id}, frame={frame_id}")
    rgb_path = _find_rgb_file(scene_dir / image_subdir, int(frame_id))
    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {rgb_path}")
    k = intrinsics_from_scene_camera(entry)
    return img, k


def _tensor_image_to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _compute_bbox_corners(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    x0, y0, z0 = mins.tolist()
    x1, y1, z1 = maxs.tolist()
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    return corners


BOX_EDGES: Sequence[Tuple[int, int]] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _project_points(points_obj: np.ndarray, pose: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = pose[:3, :3]
    t = pose[:3, 3]
    cam = (r @ points_obj.T).T + t[None, :]
    z = cam[:, 2]
    valid = z > 1e-6
    uv = np.full((points_obj.shape[0], 2), -1.0, dtype=np.float32)
    if valid.any():
        xyz = cam[valid]
        uv_valid = (k @ xyz.T).T
        uv_valid = uv_valid[:, :2] / uv_valid[:, 2:3]
        uv[valid] = uv_valid.astype(np.float32)
    return uv, valid


def _draw_bbox(image: np.ndarray, uv: np.ndarray, valid: np.ndarray, color: tuple[int, int, int], line_width: int) -> None:
    for i, j in BOX_EDGES:
        if not (valid[i] and valid[j]):
            continue
        p1 = tuple(np.round(uv[i]).astype(int).tolist())
        p2 = tuple(np.round(uv[j]).astype(int).tolist())
        cv2.line(image, p1, p2, color, line_width, lineType=cv2.LINE_AA)


def _draw_axes(
    image: np.ndarray,
    pose: np.ndarray,
    k: np.ndarray,
    axis_len: float,
    color: tuple[int, int, int],
    line_width: int,
    label_prefix: str,
) -> None:
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        dtype=np.float32,
    )
    uv, valid = _project_points(pts, pose, k)
    if not valid[0]:
        return
    o = tuple(np.round(uv[0]).astype(int).tolist())
    axis_labels = ("X", "Y", "Z")
    for idx in range(1, 4):
        if not valid[idx]:
            continue
        p = tuple(np.round(uv[idx]).astype(int).tolist())
        cv2.line(image, o, p, color, line_width, lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            f"{label_prefix}-{axis_labels[idx-1]}",
            (p[0] + 2, p[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.circle(image, o, max(2, line_width + 1), color, -1, lineType=cv2.LINE_AA)


def _auto_axis_length(points: np.ndarray) -> float:
    size = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(size) * 0.2)


def _select_indices(dataset: PreparedContextPoseDataset, num_samples: int, start_index: int, sample_ids: Sequence[str]) -> list[int]:
    if sample_ids:
        wanted = set(sample_ids)
        out = []
        for idx, rec in enumerate(dataset.records):
            if str(rec.get("id", idx)) in wanted:
                out.append(idx)
        if not out:
            raise ValueError(f"None of the requested sample_ids were found: {sample_ids}")
        return out
    end = min(len(dataset), start_index + num_samples)
    return list(range(start_index, end))


def _default_index_from_mode(cfg: Any, mode: str) -> str:
    if mode == "eval":
        return str(cfg.data.eval_index)
    if mode == "selfsup":
        if getattr(cfg.data, "selfsup_val_index", None):
            return str(cfg.data.selfsup_val_index)
        return str(cfg.data.selfsup_train_index)
    if mode == "pretrain":
        if getattr(cfg.data, "pretrain_val_index", None):
            return str(cfg.data.pretrain_val_index)
        return str(cfg.data.pretrain_train_index)
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GT and predicted 6D pose with 3D bbox + axes on ROV/BOP images.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "selfsup", "pretrain"])
    parser.add_argument("--index", type=str, default="", help="Optional explicit JSONL index path. Defaults to config-dependent index.")
    parser.add_argument("--teacher", action="store_true", help="Load teacher weights when probing a selfsup checkpoint.")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-ids", type=str, default="", help="Comma-separated sample ids to render.")
    parser.add_argument("--render-space", type=str, default="crop", choices=["crop", "full"], help="Draw on cropped prepared images or original full BOP frames.")
    parser.add_argument("--bop-root", type=str, default="", help="Required for --render-space full. Root of the original real BOP/ROV dataset split.")
    parser.add_argument("--image-subdir", type=str, default="rgb")
    parser.add_argument("--draw-init", action="store_true", help="Also draw init pose in blue for debugging.")
    parser.add_argument("--axis-length", type=float, default=-1.0, help="Axis length in object coordinates. Negative means auto.")
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--save-jsonl", action="store_true", help="Save per-sample metadata/metrics JSONL.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.runtime.seed))
    device = device_from_config(cfg)

    index_path = args.index if args.index else _default_index_from_mode(cfg, args.mode)
    dataset = PreparedContextPoseDataset(
        index_path=index_path,
        n_context=int(cfg.data.n_context),
        style_index_path=None,
        zero_background_depth=bool(getattr(cfg.data, "zero_background_depth", True)),
        max_context_depth_m=getattr(cfg.data, "max_context_depth_m", None),
    )

    model = build_model_from_cfg(cfg, teacher=args.teacher).to(device)
    load_checkpoint_weights(model, args.checkpoint, teacher=args.teacher, device=device)
    model.eval()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    sample_ids = [x.strip() for x in args.sample_ids.split(",") if x.strip()]
    indices = _select_indices(dataset, args.num_samples, args.start_index, sample_ids)

    summary: List[Dict[str, Any]] = []
    for vis_idx in indices:
        rec = dataset.records[vis_idx]
        sample = dataset[vis_idx]
        batch = _to_single_batch(sample, device)
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

        pred_pose = out["pose"][0].detach().cpu().numpy().astype(np.float32)
        gt_pose = None
        if "gt_pose" in sample:
            gt_pose = sample["gt_pose"].detach().cpu().numpy().astype(np.float32)
        init_pose = sample["init_pose"].detach().cpu().numpy().astype(np.float32)
        mesh_points = sample["mesh_points"].detach().cpu().numpy().astype(np.float32)
        bbox_corners = _compute_bbox_corners(mesh_points)
        axis_len = float(args.axis_length) if args.axis_length > 0 else _auto_axis_length(mesh_points)

        if args.render_space == "crop":
            image = _tensor_image_to_bgr_uint8(sample["image"])
            k = sample["K"].detach().cpu().numpy().astype(np.float32)
        else:
            if not args.bop_root:
                raise ValueError("--bop-root is required when --render-space=full")
            scene_id = int(rec["scene_id"])
            frame_id = int(rec["frame_id"])
            image, k = _load_full_image_and_k(args.bop_root, scene_id, frame_id, image_subdir=args.image_subdir)

        canvas = image.copy()
        # GT in green
        if gt_pose is not None:
            uv_gt, valid_gt = _project_points(bbox_corners, gt_pose, k)
            _draw_bbox(canvas, uv_gt, valid_gt, color=(0, 255, 0), line_width=int(args.line_width))
            _draw_axes(canvas, gt_pose, k, axis_len, color=(0, 255, 0), line_width=int(args.line_width), label_prefix="GT")

        # Pred in red
        uv_pred, valid_pred = _project_points(bbox_corners, pred_pose, k)
        _draw_bbox(canvas, uv_pred, valid_pred, color=(0, 0, 255), line_width=int(args.line_width))
        _draw_axes(canvas, pred_pose, k, axis_len, color=(0, 0, 255), line_width=int(args.line_width), label_prefix="P")

        # Optional init in blue
        if args.draw_init:
            uv_init, valid_init = _project_points(bbox_corners, init_pose, k)
            _draw_bbox(canvas, uv_init, valid_init, color=(255, 0, 0), line_width=max(1, int(args.line_width) - 1))
            _draw_axes(canvas, init_pose, k, axis_len, color=(255, 0, 0), line_width=max(1, int(args.line_width) - 1), label_prefix="I")

        sample_id = str(rec.get("id", vis_idx))
        # Info text
        y = 20
        cv2.putText(canvas, f"sample: {sample_id}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, f"sample: {sample_id}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, lineType=cv2.LINE_AA)
        y += 22
        cv2.putText(canvas, "GT box/axes: green | Pred box/axes: red", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "GT box/axes: green | Pred box/axes: red", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, lineType=cv2.LINE_AA)

        metrics_info: Dict[str, Any] = {
            "sample_id": sample_id,
            "scene_id": rec.get("scene_id"),
            "frame_id": rec.get("frame_id"),
            "inst_id": rec.get("inst_id"),
            "render_space": args.render_space,
            "image_out": str(output_dir / f"{sample_id}.png"),
        }
        if gt_pose is not None:
            pred_pose_t = torch.from_numpy(pred_pose).unsqueeze(0)
            gt_pose_t = torch.from_numpy(gt_pose).unsqueeze(0)
            rot_err = float(pose_geodesic_distance_deg(pred_pose_t, gt_pose_t)[0].item())
            trans_err = float(translation_error_cm(pred_pose_t, gt_pose_t)[0].item())
            metrics_info["pred_rot_err_deg"] = rot_err
            metrics_info["pred_trans_err_cm"] = trans_err
            y += 22
            text = f"pred vs gt: rot={rot_err:.2f} deg, trans={trans_err:.2f} cm"
            cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, lineType=cv2.LINE_AA)

        out_path = output_dir / f"{sample_id}.png"
        cv2.imwrite(str(out_path), canvas)
        summary.append(metrics_info)
        print(f"saved: {out_path}")

    if args.save_jsonl:
        jsonl_path = output_dir / "visualization_summary.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in summary:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"saved summary: {jsonl_path}")


if __name__ == "__main__":
    main()

