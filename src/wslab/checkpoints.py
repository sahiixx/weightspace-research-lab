"""Load only tensor state dictionaries and hash their canonical bytes.

A SHA256 fingerprint detects accidental mismatch; it is NOT a digital signature.
Do not open arbitrary non-weights-only checkpoints from untrusted sources.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from .models import ToyClassifier


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone().cpu() for key, value in state.items()}


def fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Only tensor state dictionaries are supported")
        item = tensor.detach().cpu().contiguous()
        if item.layout != torch.strided:
            raise TypeError("Only dense strided tensors are supported")
        metadata = json.dumps({"name": name, "dtype": str(item.dtype),
                               "shape": list(item.shape)}, sort_keys=True).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(item.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path: str | Path, state: dict[str, torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clone_state(state), path)


def load_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                                        for k, v in obj.items()):
        raise ValueError("Checkpoint must contain a tensor-only state_dict")
    model = ToyClassifier()
    model.load_state_dict(obj, strict=True)
    if any(not torch.isfinite(v).all().item() for v in obj.values()):
        raise ValueError("Checkpoint contains NaN or Infinity")
    return clone_state(obj)


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value} is forbidden")
    obj = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj
