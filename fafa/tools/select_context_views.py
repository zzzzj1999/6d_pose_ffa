from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from fafa.common import read_jsonl, write_jsonl


def load_pose(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape == (3, 4):
        arr = np.concatenate([arr, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected pose to be 3x4 or 4x4, got {arr.shape}")
    return arr


def rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    ra = a[:3, :3]
    rb = b[:3, :3]
    rel = ra @ rb.T
    trace = np.trace(rel)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def translation_distance_m(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def pose_distance_score(a: np.ndarray, b: np.ndarray, rot_weight: float, trans_weight: float) -> float:
    rot = rotation_distance_deg(a, b)
    trans = translation_distance_m(a, b)
    return rot_weight * rot + trans_weight * trans


def _same_sample(sample: Dict[str, Any], view: Dict[str, Any]) -> bool:
    sid_a = sample.get("scene_id")
    fid_a = sample.get("frame_id")
    iid_a = sample.get("inst_id", 0)
    sid_b = view.get("scene_id")
    fid_b = view.get("frame_id")
    iid_b = view.get("inst_id", 0)
    return sid_a == sid_b and fid_a == fid_b and iid_a == iid_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the nearest synthetic context views around init_pose.")
    parser.add_argument("--sample-index", type=str, required=True, help="JSONL with target crops and init_pose")
    parser.add_argument("--view-bank", type=str, required=True, help="JSONL with synthetic views: image/depth/mask/pose")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-context", type=int, default=4)
    parser.add_argument("--rot-weight", type=float, default=1.0)
    parser.add_argument("--trans-weight", type=float, default=100.0, help="Weight on translation distance in meters")
    parser.add_argument("--exclude-same-id", action="store_true", help="Exclude a context view when sample.id == view.id")
    parser.add_argument("--exclude-same-frame", action="store_true", help="Exclude a context view when scene/frame/inst match")
    parser.add_argument("--min-rot-distance-deg", type=float, default=0.0, help="Reject context views closer than this rotation distance")
    parser.add_argument("--min-trans-distance-m", type=float, default=0.0, help="Reject context views closer than this translation distance")
    args = parser.parse_args()

    samples = read_jsonl(args.sample_index)
    bank = read_jsonl(args.view_bank)

    if not bank:
        raise ValueError("View bank is empty")

    processed: List[Dict[str, Any]] = []
    for sample in samples:
        init_pose = load_pose(sample["init_pose"])
        object_id = sample.get("object_id")
        sample_id = sample.get("id")

        candidates: List[tuple[float, Dict[str, Any]]] = []
        for view in bank:
            if object_id is not None and view.get("object_id") is not None and view.get("object_id") != object_id:
                continue
            if bool(args.exclude_same_id) and sample_id is not None and sample_id == view.get("id"):
                continue
            if bool(args.exclude_same_frame) and _same_sample(sample, view):
                continue

            pose = load_pose(view["pose"])
            rot_dist = rotation_distance_deg(init_pose, pose)
            trans_dist = translation_distance_m(init_pose, pose)
            if rot_dist < float(args.min_rot_distance_deg):
                continue
            if trans_dist < float(args.min_trans_distance_m):
                continue

            score = args.rot_weight * rot_dist + args.trans_weight * trans_dist
            candidates.append((score, view))

        if len(candidates) < args.n_context:
            raise ValueError(
                f"Sample {sample.get('id', '<unknown>')} only has {len(candidates)} candidate views, "
                f"but n_context={args.n_context}. Reduce the exclusion thresholds or enlarge the view bank."
            )

        candidates.sort(key=lambda x: x[0])
        context = []
        for _, view in candidates[: args.n_context]:
            entry = {
                "image": view["image"],
                "depth": view["depth"],
                "mask": view["mask"],
                "pose": load_pose(view["pose"]).tolist(),
            }
            if "K" in view:
                entry["K"] = view["K"]
            context.append(entry)

        new_sample = dict(sample)
        new_sample["context"] = context
        processed.append(new_sample)

    write_jsonl(processed, args.output)
    print(f"Wrote {len(processed)} records to {args.output}")


if __name__ == "__main__":
    main()
