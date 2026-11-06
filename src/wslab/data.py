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


def dataset_manifest(draw: Samples, *, seed: int, count: int) -> dict:
    """Binds deterministic protocol AND actual tensor bytes to one SHA-256."""
    import hashlib
    import json
    if draw.split not in SEED_OFFSETS or len(draw.x) != count or len(draw.y) != count or len(draw.groups) != count:
        raise ValueError("Dataset split or count does not match declared protocol")
    tensors = {}
    for name in ("x", "y", "groups"):
        value = getattr(draw, name).detach().cpu().contiguous()
        tensors[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                         "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}
    body = {"schema_version": 1, "rule": "toy-linear-repair-y=(2*x0+x1>0)",
            "split": draw.split, "seed": seed, "seed_offset": SEED_OFFSETS[draw.split],
            "count": count, "tensors": tensors}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
                                                 separators=(",", ":")).encode()).hexdigest()
    return body


def manifest_for_split(split: str, seed: int, count: int) -> dict:
    return dataset_manifest(samples(split, seed, count), seed=seed, count=count)
