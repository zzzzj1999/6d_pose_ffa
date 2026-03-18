from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml


@dataclass
class AverageMeter:
    name: str
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


class DotDict(dict):
    """Dictionary with attribute-style access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def recursive_dotdict(x: Any) -> Any:
    if isinstance(x, dict):
        return DotDict({k: recursive_dotdict(v) for k, v in x.items()})
    if isinstance(x, list):
        return [recursive_dotdict(v) for v in x]
    return x


def load_config(path: str | os.PathLike[str]) -> DotDict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return recursive_dotdict(cfg)


def save_json(obj: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_tensor(x: np.ndarray | torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if dtype is None else x.to(dtype=dtype)
    t = torch.from_numpy(x)
    return t if dtype is None else t.to(dtype=dtype)


def format_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def resolve_path(base_dir: str | os.PathLike[str], maybe_rel: str) -> str:
    maybe_rel_path = Path(maybe_rel)
    if maybe_rel_path.is_absolute():
        return str(maybe_rel_path)
    return str(Path(base_dir) / maybe_rel_path)


def stack_if_list(values: List[torch.Tensor]) -> torch.Tensor:
    if len(values) == 0:
        raise ValueError("Cannot stack an empty list")
    return torch.stack(values, dim=0)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def freeze_bn(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def device_from_config(cfg: DotDict) -> torch.device:
    requested = str(cfg.runtime.device)
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")
