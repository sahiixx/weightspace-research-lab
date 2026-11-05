"""Independent synthetic validation and holdout-test samples."""
from __future__ import annotations

from dataclasses import dataclass
import torch

SEED_OFFSETS = {"validation": 101, "test": 202, "audit": 303}


@dataclass(frozen=True)
class Samples:
    x: torch.Tensor
    y: torch.Tensor
    groups: torch.Tensor
    split: str


def samples(split: str, seed: int = 42, count: int = 2048) -> Samples:
    if split not in SEED_OFFSETS:
        raise ValueError(f"Unsupported split {split!r}. Expected: {sorted(SEED_OFFSETS)}")
    if count < 20:
        raise ValueError("count must be >= 20")
    generator = torch.Generator(device="cpu").manual_seed(seed + SEED_OFFSETS[split])
    x = torch.randn(count, 2, generator=generator, dtype=torch.float32)
    y = (2.0 * x[:, 0] + x[:, 1] > 0).long()
    groups = (x[:, 0] >= 0).long()
    return Samples(x, y, groups, split)
