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
    if type(value) not in (int, float):
        raise PatchRejected(f"{field} must be a finite JSON number (not bool/string)")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise PatchRejected(f"{field} is out of range") from None
    if not math.isfinite(result):
        raise PatchRejected(f"{field} must be finite")
    return result


def validate_patch_schema(patch: object, *, require_dtype: bool = False) -> dict:
    """Strict JSON-object contract; authenticity is checked by a separate layer."""
    if type(patch) is not dict:
        raise PatchRejected("Patch must be a JSON object")
    permitted = {"schema_version", "task", "base_sha256", "method", "edits"}
    if set(patch) - permitted or not {"schema_version", "task", "base_sha256", "edits"} <= set(patch):
        raise PatchRejected("Unknown or missing patch fields")
    if type(patch["schema_version"]) is not int or patch["schema_version"] != 1 or patch["task"] != "toy-linear-repair":
        raise PatchRejected("Unsupported patch schema/task")
    if type(patch["base_sha256"]) is not str or len(patch["base_sha256"]) != 64 or any(c not in '0123456789abcdef' for c in patch["base_sha256"]):
        raise PatchRejected("base_sha256 must be a lowercase hexadecimal SHA-256")
    if "method" in patch and (type(patch["method"]) is not str or len(patch["method"]) > 120):
        raise PatchRejected("Invalid method")
    edits = patch["edits"]
    if type(edits) is not list or not edits or len(edits) > 1000:
        raise PatchRejected("Edits must be a nonempty bounded list")
    for edit in edits:
        if type(edit) is not dict:
            raise PatchRejected("Each edit must be an object")
        required = {"tensor", "index", "from", "to"}
        optional = {"reason", "dtype", "device"}
        if set(edit) - required - optional or not required <= set(edit):
            raise PatchRejected("Unknown or missing edit fields")
        if type(edit["tensor"]) is not str or not edit["tensor"]:
            raise PatchRejected("Invalid tensor name")
        if type(edit["index"]) is not list or any(type(v) is not int or v < 0 for v in edit["index"]):
            raise PatchRejected("Index must contain nonnegative integer coordinates")
        for number in ("from", "to"):
            _finite_number(edit[number], number)
        if "reason" in edit and (type(edit["reason"]) is not str or len(edit["reason"]) > 250):
            raise PatchRejected("Invalid reason")
        if require_dtype and ("dtype" not in edit or "device" not in edit):
            raise PatchRejected("Signed edits must declare tensor dtype and device")
        if "dtype" in edit and (type(edit["dtype"]) is not str or edit["dtype"] not in ("torch.float32",)):
            raise PatchRejected("Only explicit torch.float32 is supported")
        if "device" in edit and (type(edit["device"]) is not str or edit["device"] != "cpu"):
            raise PatchRejected("Only explicit cpu device is supported")
    return patch


def apply_checked(base: dict[str, torch.Tensor], patch: dict,
                  policy: PatchPolicy = PatchPolicy()) -> dict[str, torch.Tensor]:
    """Arithmetic/shape guardrail ONLY. Not an authorization or authenticity check."""
    patch = validate_patch_schema(patch)
    if patch["base_sha256"] != fingerprint(base):
        raise PatchRejected("Checkpoint fingerprint mismatch: stale or wrong patch")
    edits = patch["edits"]
    if not (1 <= len(edits) <= policy.max_edits):
        raise PatchRejected("Invalid number of edits")
    if any(t.device.type != "cpu" or t.layout != torch.strided or t.dtype != torch.float32
           for t in base.values()):
        raise PatchRejected("Only CPU dense float32 tensors are supported")
    total = sum(t.numel() for t in base.values())
    result = clone_state(base)
    changed: set[tuple[str, tuple[int, ...]]] = set()
    for edit in edits:
        name = edit["tensor"]
        if name not in policy.allowed_tensors or name not in result:
            raise PatchRejected(f"Tensor {name!r} is not allowlisted")
        index = edit["index"]
        target = result[name]
        if len(index) != target.ndim or any(n >= target.shape[d] for d, n in enumerate(index)):
            raise PatchRejected(f"Index {index!r} invalid for {name}")
        key = (name, tuple(index))
        if key in changed:
            raise PatchRejected("Duplicate edit coordinates")
        changed.add(key)
        if edit.get("dtype", "torch.float32") != str(target.dtype):
            raise PatchRejected("Tensor dtype mismatch")
        if edit.get("device", "cpu") != str(target.device):
            raise PatchRejected("Tensor device mismatch")
        old = _finite_number(edit["from"], "from")
        new = _finite_number(edit["to"], "to")
        current = float(target[tuple(index)].item())
        if current != old:
            raise PatchRejected("Old value mismatch")
        if new == old:
            raise PatchRejected("No-op edits are disallowed")
        if abs(new-old) > policy.max_abs_delta or abs(new) > policy.max_abs_result:
            raise PatchRejected("Proposed delta or result exceeds patch policy")
        if float(torch.tensor(new, dtype=target.dtype).item()) != new:
            raise PatchRejected("New value is not exactly representable in target dtype")
        target[tuple(index)] = new
    # Count actual unique changed coordinates, not merely edit-record length.
    changed_count = sum(int((base[k] != result[k]).sum().item()) for k in base)
    if changed_count != len(changed):
        raise PatchRejected("Unexpected changed coordinate count")
    if changed_count / total > policy.max_changed_fraction:
        raise PatchRejected("Patch modifies too large a fraction of parameters")
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
