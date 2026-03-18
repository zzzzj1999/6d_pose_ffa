from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a point cloud from a mesh for ADD / ADD-S training and evaluation.")
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected a triangular mesh")

    points, _ = trimesh.sample.sample_surface(mesh, args.num_points)
    points = points.astype(np.float32)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, points)
    print(f"Saved {points.shape[0]} points to {args.output}")


if __name__ == "__main__":
    main()
