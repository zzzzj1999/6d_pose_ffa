from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DEPTH_EXTS = (".npy", ".png", ".tif", ".tiff", ".exr")
BOP_SPLIT_CANDIDATES = ("train_pbr", "train_real", "test_pool", "test", "train")
MASK_SUBDIR_CANDIDATES = ("mask", "mask_visib")


@dataclass(frozen=True)
class SceneObjectRecord:
    scene_id: int
    frame_id: int
    inst_id: int
    object_id: int
    rgb_path: Path
    mask_path: Path | None
    depth_path: Path | None
    k: np.ndarray
    pose_m2c: np.ndarray | None
    bbox_obj: tuple[int, int, int, int] | None
    bbox_visib: tuple[int, int, int, int] | None
    depth_factor_m: float | None = None


@dataclass(frozen=True)
class CropWindow:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class CropResult:
    image: np.ndarray
    k: np.ndarray
    window: CropWindow


def _is_scene_dir(path: Path) -> bool:
    return path.is_dir() and path.name.isdigit()


def _contains_scene_dirs(path: Path) -> bool:
    return path.is_dir() and any(_is_scene_dir(child) for child in path.iterdir())


def resolve_bop_scene_root(dataset_root: str | Path, preferred_split: str | None = None) -> Path:
    """Resolve a user-provided root to the directory that directly contains BOP scene folders.

    Supported inputs:
    - /path/to/dataset/train_pbr
    - /path/to/dataset            (contains train_pbr)
    - /path/to/output_root        (contains a single dataset child with train_pbr)
    """
    root = Path(dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"BOP dataset root does not exist: {root}")
    if _contains_scene_dirs(root):
        return root

    split_names = []
    if preferred_split:
        split_names.append(str(preferred_split))
    split_names.extend([s for s in BOP_SPLIT_CANDIDATES if s not in split_names])

    for split in split_names:
        candidate = root / split
        if _contains_scene_dirs(candidate):
            return candidate

    nested_candidates: list[Path] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if _contains_scene_dirs(child):
            nested_candidates.append(child)
            continue
        for split in split_names:
            candidate = child / split
            if _contains_scene_dirs(candidate):
                nested_candidates.append(candidate)

    deduped = []
    seen = set()
    for candidate in nested_candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)

    if len(deduped) == 1:
        return deduped[0]
    if len(deduped) > 1:
        options = ", ".join(str(p) for p in deduped)
        raise ValueError(
            f"Ambiguous BOP root {root}. Multiple scene roots were found: {options}. "
            "Please point --dataset-root directly to the desired split such as <dataset>/train_pbr."
        )

    raise FileNotFoundError(
        f"Could not resolve a BOP scene root from {root}. Expected scene folders like 000000/ or a known split like train_pbr/."
    )


def sorted_scene_dirs(dataset_root: Path, scene_ids: Sequence[int] | None = None) -> list[Path]:
    dataset_root = resolve_bop_scene_root(dataset_root)
    if scene_ids is None or len(scene_ids) == 0:
        candidates = [p for p in dataset_root.iterdir() if _is_scene_dir(p)]
        return sorted(candidates, key=lambda p: int(p.name))
    wanted = {int(s) for s in scene_ids}
    out: list[Path] = []
    for sid in sorted(wanted):
        path = dataset_root / f"{sid:06d}"
        if not path.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {path}")
        out.append(path)
    return out


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def pose_from_bop(entry: Mapping[str, Any], translation_scale: float = 0.001) -> np.ndarray:
    rot = np.asarray(entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    trans = np.asarray(entry["cam_t_m2c"], dtype=np.float32).reshape(3) * float(translation_scale)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose


def intrinsics_from_scene_camera(entry: Mapping[str, Any]) -> np.ndarray:
    k = np.asarray(entry["cam_K"], dtype=np.float32)
    return k.reshape(3, 3)


def depth_factor_m_from_scene_camera(entry: Mapping[str, Any], translation_scale: float = 0.001) -> float:
    # In BOP-style exports, raw depth values are typically scaled by scene_camera.depth_scale
    # and use the same metric family as pose annotations (commonly mm). translation_scale
    # converts that annotation unit to meters.
    depth_scale = float(entry.get("depth_scale", 1.0))
    return float(translation_scale) * depth_scale


def _find_file(stem: str, folder: Path, extensions: Sequence[str]) -> Path | None:
    for ext in extensions:
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _mask_name(frame_id: int, inst_id: int) -> str:
    return f"{frame_id:06d}_{inst_id:06d}.png"


def _bbox_tuple(x: Any) -> tuple[int, int, int, int] | None:
    if x is None:
        return None
    arr = [int(round(float(v))) for v in x]
    if len(arr) != 4:
        raise ValueError(f"Expected bbox of length 4, got {arr}")
    return (arr[0], arr[1], arr[2], arr[3])


def _scene_keys(*mappings: Mapping[str, Any] | None) -> list[str]:
    keys: set[str] = set()
    for m in mappings:
        if m is None:
            continue
        keys.update(str(k) for k in m.keys())
    return sorted(keys, key=lambda x: int(x))


def _resolve_mask_dir(scene_dir: Path, mask_subdir: str) -> Path | None:
    if mask_subdir != "auto":
        path = scene_dir / mask_subdir
        return path if path.is_dir() else None
    for candidate in MASK_SUBDIR_CANDIDATES:
        path = scene_dir / candidate
        if path.is_dir():
            return path
    return None


def iter_bop_records(
    dataset_root: str | Path,
    *,
    scene_ids: Sequence[int] | None = None,
    image_subdir: str = "rgb",
    mask_subdir: str = "auto",
    depth_subdir: str = "depth",
    translation_scale: float = 0.001,
    preferred_split: str | None = None,
) -> Iterator[SceneObjectRecord]:
    root = resolve_bop_scene_root(dataset_root, preferred_split=preferred_split)
    for scene_dir in sorted_scene_dirs(root, scene_ids):
        scene_id = int(scene_dir.name)
        scene_camera = read_json(scene_dir / "scene_camera.json")
        scene_gt = read_json(scene_dir / "scene_gt.json") if (scene_dir / "scene_gt.json").is_file() else None
        scene_gt_info = read_json(scene_dir / "scene_gt_info.json") if (scene_dir / "scene_gt_info.json").is_file() else None

        rgb_dir = scene_dir / image_subdir
        mask_dir = _resolve_mask_dir(scene_dir, mask_subdir)
        depth_dir = scene_dir / depth_subdir

        for key in _scene_keys(scene_camera, scene_gt, scene_gt_info):
            frame_id = int(key)
            rgb_path = _find_file(f"{frame_id:06d}", rgb_dir, IMAGE_EXTS)
            if rgb_path is None:
                raise FileNotFoundError(f"Could not find RGB file for scene={scene_id}, frame={frame_id}")
            depth_path = _find_file(f"{frame_id:06d}", depth_dir, DEPTH_EXTS) if depth_dir.is_dir() else None

            camera_entry = scene_camera[str(frame_id)] if str(frame_id) in scene_camera else scene_camera[key]
            k = intrinsics_from_scene_camera(camera_entry)
            depth_factor_m = depth_factor_m_from_scene_camera(camera_entry, translation_scale=translation_scale)
            gt_list = [] if scene_gt is None else scene_gt.get(str(frame_id), scene_gt.get(key, []))
            gt_info_list = [] if scene_gt_info is None else scene_gt_info.get(str(frame_id), scene_gt_info.get(key, []))

            if len(gt_list) == 0:
                # Some purely-unlabeled BOP-style exports may omit scene_gt.json.
                yield SceneObjectRecord(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    inst_id=0,
                    object_id=-1,
                    rgb_path=rgb_path,
                    mask_path=None,
                    depth_path=depth_path,
                    k=k,
                    pose_m2c=None,
                    bbox_obj=None,
                    bbox_visib=None,
                    depth_factor_m=depth_factor_m,
                )
                continue

            for inst_id, gt_entry in enumerate(gt_list):
                gt_info_entry = gt_info_list[inst_id] if inst_id < len(gt_info_list) else None
                pose_m2c = pose_from_bop(gt_entry, translation_scale=translation_scale)
                mask_path = None
                if mask_dir is not None:
                    candidate = mask_dir / _mask_name(frame_id, inst_id)
                    if candidate.is_file():
                        mask_path = candidate
                yield SceneObjectRecord(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    inst_id=inst_id,
                    object_id=int(gt_entry.get("obj_id", -1)),
                    rgb_path=rgb_path,
                    mask_path=mask_path,
                    depth_path=depth_path,
                    k=k.copy(),
                    pose_m2c=pose_m2c,
                    bbox_obj=_bbox_tuple(gt_info_entry.get("bbox_obj")) if gt_info_entry is not None else None,
                    bbox_visib=_bbox_tuple(gt_info_entry.get("bbox_visib")) if gt_info_entry is not None else None,
                    depth_factor_m=depth_factor_m,
                )


def read_image_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: str | Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8) * 255


def read_depth(path: str | Path, raw_depth_to_meter: float = 0.001) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth file: {path}")
        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
    return depth * float(raw_depth_to_meter)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def choose_bbox(
    record: SceneObjectRecord,
    *,
    crop_source: str = "bbox_visib",
) -> tuple[int, int, int, int]:
    if crop_source == "bbox_visib":
        if record.bbox_visib is None:
            raise ValueError(f"bbox_visib missing for scene={record.scene_id}, frame={record.frame_id}, inst={record.inst_id}")
        return record.bbox_visib
    if crop_source == "bbox_obj":
        if record.bbox_obj is None:
            raise ValueError(f"bbox_obj missing for scene={record.scene_id}, frame={record.frame_id}, inst={record.inst_id}")
        return record.bbox_obj
    if crop_source == "mask_bbox":
        if record.mask_path is None:
            raise ValueError(f"mask required for crop_source=mask_bbox at scene={record.scene_id}, frame={record.frame_id}, inst={record.inst_id}")
        bbox = bbox_from_mask(read_mask(record.mask_path))
        if bbox is None:
            raise ValueError(f"empty mask for scene={record.scene_id}, frame={record.frame_id}, inst={record.inst_id}")
        return bbox
    raise ValueError(f"Unsupported crop_source: {crop_source}")


def make_crop_window(
    bbox_xywh: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    *,
    pad_scale: float = 1.2,
    square: bool = True,
    min_size: int = 32,
) -> CropWindow:
    x, y, w, h = bbox_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    crop_w = max(float(min_size), float(w) * float(pad_scale))
    crop_h = max(float(min_size), float(h) * float(pad_scale))
    if square:
        side = max(crop_w, crop_h)
        crop_w = crop_h = side

    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    x1 = int(round(cx + crop_w / 2.0))
    y1 = int(round(cy + crop_h / 2.0))

    if x0 < 0:
        x1 += -x0
        x0 = 0
    if y0 < 0:
        y1 += -y0
        y0 = 0
    if x1 > image_w:
        shift = x1 - image_w
        x0 -= shift
        x1 = image_w
    if y1 > image_h:
        shift = y1 - image_h
        y0 -= shift
        y1 = image_h

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image_w, x1)
    y1 = min(image_h, y1)

    if x1 <= x0:
        x1 = min(image_w, x0 + max(1, min_size))
    if y1 <= y0:
        y1 = min(image_h, y0 + max(1, min_size))
    return CropWindow(x0=x0, y0=y0, x1=x1, y1=y1)


def adjust_intrinsics_for_crop_resize(
    k: np.ndarray,
    window: CropWindow,
    output_size: tuple[int, int],
) -> np.ndarray:
    out_h, out_w = output_size
    crop_w = max(1, window.w)
    crop_h = max(1, window.h)
    sx = float(out_w) / float(crop_w)
    sy = float(out_h) / float(crop_h)

    out = np.asarray(k, dtype=np.float32).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = (out[0, 2] - float(window.x0)) * sx
    out[1, 2] = (out[1, 2] - float(window.y0)) * sy
    return out


def crop_and_resize_array(
    arr: np.ndarray,
    window: CropWindow,
    output_size: tuple[int, int],
    *,
    interpolation: int,
) -> np.ndarray:
    cropped = arr[window.y0 : window.y1, window.x0 : window.x1]
    out_h, out_w = output_size
    if cropped.shape[0] == out_h and cropped.shape[1] == out_w:
        return cropped
    return cv2.resize(cropped, (out_w, out_h), interpolation=interpolation)


def crop_and_resize_with_intrinsics(
    arr: np.ndarray,
    k: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    output_size: tuple[int, int],
    *,
    pad_scale: float = 1.2,
    square: bool = True,
    min_size: int = 32,
    interpolation: int = cv2.INTER_LINEAR,
) -> CropResult:
    image_h, image_w = arr.shape[:2]
    window = make_crop_window(
        bbox_xywh,
        image_w=image_w,
        image_h=image_h,
        pad_scale=pad_scale,
        square=square,
        min_size=min_size,
    )
    cropped = crop_and_resize_array(arr, window, output_size, interpolation=interpolation)
    new_k = adjust_intrinsics_for_crop_resize(k, window, output_size)
    return CropResult(image=cropped, k=new_k, window=window)


def save_rgb_png(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(path)


def save_depth_npy(path: str | Path, depth_m: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(depth_m, dtype=np.float32))


def relpath_str(path: str | Path, start: str | Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(start).resolve())


def scene_object_id(scene_id: int, frame_id: int, inst_id: int) -> str:
    return f"s{scene_id:06d}_f{frame_id:06d}_i{inst_id:06d}"


def parse_scene_ids(text: str | None) -> list[int] | None:
    if text is None or text == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _random_axis(rng: random.Random) -> np.ndarray:
    v = np.asarray([rng.uniform(-1.0, 1.0) for _ in range(3)], dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return v / norm


def axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.asarray(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float32,
    )


def perturb_pose(
    pose: np.ndarray,
    *,
    rot_deg: float = 0.0,
    trans_m: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    rng = random.Random(seed)
    out = np.asarray(pose, dtype=np.float32).copy()
    if rot_deg > 0:
        angle = math.radians(rng.uniform(-float(rot_deg), float(rot_deg)))
        axis = _random_axis(rng)
        delta_r = axis_angle_to_matrix(axis, angle)
        out[:3, :3] = delta_r @ out[:3, :3]
    if trans_m > 0:
        delta_t = np.asarray([rng.uniform(-trans_m, trans_m) for _ in range(3)], dtype=np.float32)
        out[:3, 3] += delta_t
    return out


def load_pose_predictions(path: str | Path) -> dict[tuple[int, int, int], np.ndarray]:
    mapping: dict[tuple[int, int, int], np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scene_id = int(rec["scene_id"])
            frame_id = int(rec["frame_id"])
            inst_id = int(rec.get("inst_id", 0))
            pose_key = "init_pose" if "init_pose" in rec else "pose"
            pose = np.asarray(rec[pose_key], dtype=np.float32)
            if pose.shape == (3, 4):
                pose = np.concatenate([pose, np.asarray([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
            mapping[(scene_id, frame_id, inst_id)] = pose
    return mapping
