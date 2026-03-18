from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from fafa.losses.core import self_supervised_loss
from fafa.modeling import FAFANet


def make_pose_batch(batch: int) -> torch.Tensor:
    pose = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    pose[:, 2, 3] = 1.0
    return pose


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    b, n, h, w = 2, 4, 64, 64
    real = torch.rand(b, 3, h, w)
    synth = torch.rand(b, n, 3, h, w)
    depth = torch.ones(b, n, 1, h, w)
    masks = torch.ones(b, n, 1, h, w)
    synth_poses = make_pose_batch(b).unsqueeze(1).repeat(1, n, 1, 1)
    init_pose = make_pose_batch(b)
    k = torch.tensor(
        [
            [[50.0, 0.0, w / 2], [0.0, 50.0, h / 2], [0.0, 0.0, 1.0]],
            [[50.0, 0.0, w / 2], [0.0, 50.0, h / 2], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    points = torch.rand(b, 256, 3) - 0.5
    symmetric = torch.tensor([False, True])

    student = FAFANet(feature_dim=64, hidden_dim=64, outer_iters=2)
    teacher = FAFANet(feature_dim=64, hidden_dim=64, outer_iters=3)
    teacher.load_state_dict(student.state_dict(), strict=False)

    with torch.no_grad():
        teacher_out = teacher(real, synth, depth, masks, synth_poses, init_pose, k)
    student_out = student(real, synth, depth, masks, synth_poses, init_pose, k)

    total, stats = self_supervised_loss(
        student_out=student_out,
        teacher_out=teacher_out,
        real_image=real,
        synth_masks=masks,
        model_points=points,
        symmetric=symmetric,
        gamma1=0.1,
        gamma2=0.1,
        gamma3=10.0,
        gamma4=10.0,
    )

    assert torch.isfinite(total).item(), "Total loss must be finite"
    assert student_out["pose"].shape == (b, 4, 4)
    assert student_out["flows"].shape == (b, n, 2, h, w)
    assert "loss_pose" in stats
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
