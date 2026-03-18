from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from fafa.common import read_jsonl


REQUIRED_TOP_LEVEL = ["image", "K", "init_pose", "mesh_points", "context"]
REQUIRED_CONTEXT = ["image", "depth", "mask", "pose"]


def check_pose(x: Any) -> None:
    arr = np.asarray(x)
    if arr.shape not in {(3, 4), (4, 4)}:
        raise ValueError(f"Pose must be 3x4 or 4x4, got {arr.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check a FAFA JSONL index file.")
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.index)
    base_dir = Path(args.index).resolve().parent
    if not records:
        raise ValueError("Index is empty")

    for i, record in enumerate(records):
        for key in REQUIRED_TOP_LEVEL:
            if key not in record:
                raise KeyError(f"Record {i} is missing required key '{key}'")
        np.asarray(record["K"], dtype=np.float32).reshape(3, 3)
        check_pose(record["init_pose"])
        if "gt_pose" in record:
            check_pose(record["gt_pose"])
        for j, ctx in enumerate(record["context"]):
            for key in REQUIRED_CONTEXT:
                if key not in ctx:
                    raise KeyError(f"Record {i}, context {j} is missing key '{key}'")
            check_pose(ctx["pose"])
            if "K" in ctx:
                np.asarray(ctx["K"], dtype=np.float32).reshape(3, 3)
        if args.check_files:
            for rel in [record["image"], record["mesh_points"]]:
                if not (base_dir / rel).exists():
                    raise FileNotFoundError(base_dir / rel)
            for ctx in record["context"]:
                for rel in [ctx["image"], ctx["depth"], ctx["mask"]]:
                    if not (base_dir / rel).exists():
                        raise FileNotFoundError(base_dir / rel)

    print(f"Index looks valid: {args.index} ({len(records)} records)")


if __name__ == "__main__":
    main()
