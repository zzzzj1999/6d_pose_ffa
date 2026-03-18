from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fafa.common import load_config
from fafa.data.dataset import PreparedContextPoseDataset, build_dataloaders


def _write_toy_scene(scene_dir: Path, *, scene_id: int, nframes: int) -> None:
    for sub in ["rgb", "depth", "mask", "mask_visib"]:
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
    scene_camera = {}
    scene_gt = {}
    scene_gt_info = {}
    for fid in range(nframes):
        width, height = 320, 240
        x0, y0 = 70 + fid * 8, 50
        x1, y1 = 240 + fid * 4, 180

        rgb = Image.new("RGB", (width, height), (20 + 10 * scene_id, 30 + 5 * fid, 40))
        ImageDraw.Draw(rgb).rectangle([x0, y0, x1, y1], fill=(200, 100, 50))
        rgb.save(scene_dir / "rgb" / f"{fid:06d}.png")

        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
        mask.save(scene_dir / "mask" / f"{fid:06d}_000000.png")
        mask.save(scene_dir / "mask_visib" / f"{fid:06d}_000000.png")

        depth = np.zeros((height, width), dtype=np.uint16)
        depth[y0 : y1 + 1, x0 : x1 + 1] = 1000 + 10 * fid + 20 * scene_id
        Image.fromarray(depth).save(scene_dir / "depth" / f"{fid:06d}.png")

        scene_camera[str(fid)] = {
            "cam_K": [300.0, 0.0, width / 2.0, 0.0, 300.0, height / 2.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        scene_gt[str(fid)] = [
            {
                "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "cam_t_m2c": [0.0, 0.0, 1000.0 + 40 * fid + 25 * scene_id],
                "obj_id": 1,
            }
        ]
        scene_gt_info[str(fid)] = [{"bbox_obj": [x0, y0, x1 - x0, y1 - y0], "bbox_visib": [x0, y0, x1 - x0, y1 - y0]}]

    (scene_dir / "scene_camera.json").write_text(json.dumps(scene_camera), encoding="utf-8")
    (scene_dir / "scene_gt.json").write_text(json.dumps(scene_gt), encoding="utf-8")
    (scene_dir / "scene_gt_info.json").write_text(json.dumps(scene_gt_info), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fafa_bproc_smoke_") as tmp:
        tmpdir = Path(tmp)
        synth_root = tmpdir / "synth_dataset" / "train_pbr"
        real_root = tmpdir / "real_dataset" / "test_pool"
        for sid in [0, 1]:
            _write_toy_scene(synth_root / f"{sid:06d}", scene_id=sid, nframes=3)
        for sid in [10, 11]:
            _write_toy_scene(real_root / f"{sid:06d}", scene_id=sid, nframes=3)

        mesh_points = tmpdir / "mesh_points.npy"
        np.save(mesh_points, np.random.randn(128, 3).astype(np.float32))

        workdir = tmpdir / "prepared"
        cmd = [
            sys.executable,
            "-m",
            "fafa.tools.prepare_blenderproc_fafa",
            "--synth-root",
            str(tmpdir / "synth_dataset"),
            "--real-root",
            str(tmpdir / "real_dataset"),
            "--workdir",
            str(workdir),
            "--mesh-points",
            str(mesh_points),
            "--symmetric",
            "--synth-val-ratio",
            "0.5",
            "--real-eval-ratio",
            "0.5",
            "--n-context",
            "4",
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT))

        pretrain_cfg = load_config(workdir / "configs" / "blenderproc_pretrain.yaml")
        pretrain_loader, val_loader = build_dataloaders(pretrain_cfg, "pretrain")
        assert len(pretrain_loader.dataset) > 0
        assert val_loader is not None and len(val_loader.dataset) > 0

        selfsup_cfg = load_config(workdir / "configs" / "blenderproc_selfsup.yaml")
        selfsup_loader, selfsup_val_loader = build_dataloaders(selfsup_cfg, "selfsup")
        assert len(selfsup_loader.dataset) > 0
        assert selfsup_val_loader is not None and len(selfsup_val_loader.dataset) > 0

        eval_ds = PreparedContextPoseDataset(str(workdir / "indices" / "real_eval.jsonl"), n_context=4)
        sample = eval_ds[0]
        assert "gt_pose" in sample
        assert sample["context_images"].shape[0] == 4
        print("BlenderProc FAFA prep smoke test passed.")


if __name__ == "__main__":
    main()
