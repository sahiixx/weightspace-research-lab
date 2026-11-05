"""Explicit local patch constraints plus held-out validation checks.

These are useful *guardrails*, not a security sandbox or model alignment proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from .benchmarks import Metrics
from .checkpoints import clone_state, fingerprint


class PatchRejected(ValueError):
    pass


@dataclass(frozen=True)
class PatchPolicy:
    allowed_tensors: tuple[str, ...] = ("linear.weight",)
    max_edits: int = 1
    max_changed_fraction: float = 0.25
    max_abs_delta: float = 6.0
    max_abs_result: float = 8.0
    min_validation_gain: float = 0.05
    max_group_accuracy_drop: float = 0.02


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PatchRejected(f"{field} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise PatchRejected(f"{field} must be a finite number") from None
    if not math.isfinite(result):
        raise PatchRejected(f"{field} must be finite")
    return result


def apply_checked(base: dict[str, torch.Tensor], patch: dict,
                  policy: PatchPolicy = PatchPolicy()) -> dict[str, torch.Tensor]:
    """Return an isolated candidate copy, or reject before committing anything."""
    if type(patch.get("schema_version")) is not int or patch.get("schema_version") != 1 or patch.get("task") != "toy-linear-repair":
        raise PatchRejected("Unsupported patch schema/task")
    if patch.get("base_sha256") != fingerprint(base):
        raise PatchRejected("Checkpoint fingerprint mismatch: stale or wrong patch")
    edits = patch.get("edits")
    if not isinstance(edits, list) or not (1 <= len(edits) <= policy.max_edits):
        raise PatchRejected("Invalid number of edits")
    total = sum(t.numel() for t in base.values())
    if len(edits) / total > policy.max_changed_fraction:
        raise PatchRejected("Patch modifies too large a fraction of the parameters")
    result = clone_state(base)
    changed = set()
    for edit in edits:
        if not isinstance(edit, dict):
            raise PatchRejected("Each edit must be a JSON object")
        name = edit.get("tensor")
        if not isinstance(name, str) or name not in policy.allowed_tensors or name not in result:
            raise PatchRejected(f"Tensor {name!r} is not allowlisted")
        index = edit.get("index")
        target = result[name]
        if (not isinstance(index, list) or len(index) != target.ndim or
                any(type(n) is not int or n < 0 or n >= target.shape[d]
                    for d, n in enumerate(index))):
            raise PatchRejected(f"Index {index!r} invalid for {name}")
        key = (name, tuple(index))
        if key in changed:
            raise PatchRejected("Duplicate edit coordinates")
        changed.add(key)
        old = _finite_number(edit.get("from"), "from")
        new = _finite_number(edit.get("to"), "to")
        current = float(target[tuple(index)].item())
        if current != old:
            raise PatchRejected("Old value mismatch")
        if new == old:
            raise PatchRejected("No-op edits are disallowed")
        if abs(new-old) > policy.max_abs_delta or abs(new) > policy.max_abs_result:
            raise PatchRejected("Proposed delta or result exceeds patch policy")
        target[tuple(index)] = new
    if any(not torch.isfinite(t).all().item() for t in result.values()):
        raise PatchRejected("Candidate contains non-finite weights")
    return result


def validation_gate(before: Metrics, after: Metrics,
                    policy: PatchPolicy = PatchPolicy()) -> None:
    if before.split != "validation" or after.split != "validation":
        raise PatchRejected("Validation gate must use the validation split")
    if after.accuracy + 1e-9 < before.accuracy + policy.min_validation_gain:
        raise PatchRejected("Insufficient validation accuracy improvement")
    for name, old_acc in before.group_accuracy.items():
        if name not in after.group_accuracy:
            raise PatchRejected("Missing validation subgroup")
        if after.group_accuracy[name] + policy.max_group_accuracy_drop + 1e-9 < old_acc:
            raise PatchRejected(f"Regression in subgroup {name}")
