from __future__ import annotations

import torch


def update_ema(teacher: torch.nn.Module, student: torch.nn.Module, momentum: float = 0.999) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
        for b_t, b_s in zip(teacher.buffers(), student.buffers()):
            b_t.copy_(b_s)
