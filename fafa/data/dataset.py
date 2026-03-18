from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fafa.common import read_jsonl, resolve_path


def _read_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _read_mask(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr > 0).astype(np.float32)
    return arr


def _read_depth(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot read depth map: {path}")
        depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def _load_pose(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape == (3, 4):
        arr = np.concatenate([arr, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
    if arr.shape != (4, 4):
        raise ValueError(f"Pose must be 3x4 or 4x4, got {arr.shape}")
    return arr


def _load_k(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape == (9,):
        arr = arr.reshape(3, 3)
    if arr.shape != (3, 3):
        raise ValueError(f"Camera intrinsics must be 3x3 or flat-9, got {arr.shape}")
    return arr


def sanitize_depth_with_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    zero_background_depth: bool = True,
    max_valid_depth_m: float | None = None,
) -> np.ndarray:
    out = np.asarray(depth, dtype=np.float32).copy()
    mask_bin = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    if zero_background_depth:
        out *= mask_bin
    if max_valid_depth_m is not None and max_valid_depth_m > 0:
        out[(out > max_valid_depth_m)] = 0.0
    out[~np.isfinite(out)] = 0.0
    out[out < 0] = 0.0
    return out


class MeshPointCache:
    def __init__(self) -> None:
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, path: str) -> np.ndarray:
        if path not in self._cache:
            points = np.load(path).astype(np.float32)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Mesh point cloud must have shape [P,3], got {points.shape} from {path}")
            self._cache[path] = points
        return self._cache[path]


class PreparedContextPoseDataset(Dataset):
    """Dataset for FAFA training/evaluation."""

    def __init__(
        self,
        index_path: str,
        n_context: int = 4,
        style_index_path: Optional[str] = None,
        *,
        zero_background_depth: bool = True,
        max_context_depth_m: float | None = None,
    ) -> None:
        super().__init__()
        self.index_path = index_path
        self.base_dir = os.path.dirname(index_path)
        self.records = read_jsonl(index_path)
        self.n_context = int(n_context)
        self.style_records: List[Dict[str, Any]] = []
        self.style_base_dir = self.base_dir
        if style_index_path is not None:
            self.style_records = read_jsonl(style_index_path)
            self.style_base_dir = os.path.dirname(style_index_path)
        self.mesh_points = MeshPointCache()
        self.zero_background_depth = bool(zero_background_depth)
        self.max_context_depth_m = None if max_context_depth_m is None else float(max_context_depth_m)

    def __len__(self) -> int:
        return len(self.records)

    def _load_style_image(self, target_hw: Tuple[int, int]) -> np.ndarray:
        if not self.style_records:
            h, w = target_hw
            return np.zeros((h, w, 3), dtype=np.float32)
        record = random.choice(self.style_records)
        style_path = resolve_path(self.style_base_dir, record["image"])
        img = _read_image(style_path)
        h, w = target_hw
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | bool | str]:
        record = self.records[idx]
        image_path = resolve_path(self.base_dir, record["image"])
        img = _read_image(image_path)
        h, w, _ = img.shape

        target_k = _load_k(record["K"])
        context = record.get("context", [])
        if len(context) < self.n_context:
            raise ValueError(
                f"Sample {idx} only has {len(context)} context views, but n_context={self.n_context}"
            )
        context = context[: self.n_context]

        ctx_imgs: List[np.ndarray] = []
        ctx_depths: List[np.ndarray] = []
        ctx_masks: List[np.ndarray] = []
        ctx_poses: List[np.ndarray] = []
        ctx_ks: List[np.ndarray] = []
        for c in context:
            ctx_imgs.append(_read_image(resolve_path(self.base_dir, c["image"])))
            ctx_depths.append(_read_depth(resolve_path(self.base_dir, c["depth"])))
            ctx_masks.append(_read_mask(resolve_path(self.base_dir, c["mask"])))
            ctx_poses.append(_load_pose(c["pose"]))
            ctx_ks.append(_load_k(c.get("K", target_k)))

        ctx_imgs = [cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in ctx_imgs]
        ctx_masks = [cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in ctx_masks]
        ctx_masks = [(x > 0.5).astype(np.float32) for x in ctx_masks]
        ctx_depths = [cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in ctx_depths]
        ctx_depths = [
            sanitize_depth_with_mask(
                depth,
                mask,
                zero_background_depth=self.zero_background_depth,
                max_valid_depth_m=self.max_context_depth_m,
            )
            for depth, mask in zip(ctx_depths, ctx_masks)
        ]

        gt_pose = record.get("gt_pose")
        mesh_points_path = resolve_path(self.base_dir, record["mesh_points"])
        points = self.mesh_points.get(mesh_points_path)
        style_img = self._load_style_image((h, w))

        output: Dict[str, torch.Tensor | bool | str] = {
            "image": torch.from_numpy(img).permute(2, 0, 1).float(),
            "style_image": torch.from_numpy(style_img).permute(2, 0, 1).float(),
            "K": torch.from_numpy(target_k),
            "init_pose": torch.from_numpy(_load_pose(record["init_pose"])),
            "context_images": torch.from_numpy(np.stack(ctx_imgs, axis=0)).permute(0, 3, 1, 2).float(),
            "context_depths": torch.from_numpy(np.stack(ctx_depths, axis=0)[:, None, ...]).float(),
            "context_masks": torch.from_numpy(np.stack(ctx_masks, axis=0)[:, None, ...]).float(),
            "context_poses": torch.from_numpy(np.stack(ctx_poses, axis=0)).float(),
            "context_ks": torch.from_numpy(np.stack(ctx_ks, axis=0)).float(),
            "mesh_points": torch.from_numpy(points).float(),
            "symmetric": bool(record.get("symmetric", False)),
            "sample_id": str(record.get("id", idx)),
        }
        if gt_pose is not None:
            output["gt_pose"] = torch.from_numpy(_load_pose(gt_pose)).float()
        return output


def _collate_bool(batch: List[Dict[str, Any]], key: str) -> torch.Tensor:
    return torch.tensor([bool(x[key]) for x in batch], dtype=torch.bool)


def collate_prepared_context(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for key in keys:
        value = batch[0][key]
        if isinstance(value, torch.Tensor):
            out[key] = torch.stack([x[key] for x in batch], dim=0)
        elif isinstance(value, bool):
            out[key] = _collate_bool(batch, key)
        else:
            out[key] = [x[key] for x in batch]
    return out


def build_dataloaders(cfg: Any, mode: str) -> Tuple[DataLoader, Optional[DataLoader]]:
    data_cfg = cfg.data
    runtime_cfg = cfg.runtime
    train_index = getattr(data_cfg, f"{mode}_train_index", None)
    val_index = getattr(data_cfg, f"{mode}_val_index", None)
    style_index = getattr(data_cfg, "real_style_index", None)
    if train_index is None:
        raise ValueError(f"Missing data.{mode}_train_index in config")

    ds_kwargs = {
        "n_context": int(data_cfg.n_context),
        "style_index_path": str(style_index) if style_index else None,
        "zero_background_depth": bool(getattr(data_cfg, "zero_background_depth", True)),
        "max_context_depth_m": getattr(data_cfg, "max_context_depth_m", None),
    }

    train_ds = PreparedContextPoseDataset(
        index_path=str(train_index),
        **ds_kwargs,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(runtime_cfg.batch_size),
        shuffle=True,
        num_workers=int(runtime_cfg.num_workers),
        pin_memory=bool(runtime_cfg.pin_memory),
        collate_fn=collate_prepared_context,
    )

    val_loader: Optional[DataLoader] = None
    if val_index:
        val_ds = PreparedContextPoseDataset(
            index_path=str(val_index),
            **ds_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(runtime_cfg.batch_size),
            shuffle=False,
            num_workers=int(runtime_cfg.num_workers),
            pin_memory=bool(runtime_cfg.pin_memory),
            collate_fn=collate_prepared_context,
        )
    return train_loader, val_loader
