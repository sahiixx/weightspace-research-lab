import copy
import math
import pytest
import torch
from wslab.models import corrupted_model
from wslab.checkpoints import fingerprint
from wslab.safety import PatchRejected, PatchPolicy, apply_checked, validation_gate
from wslab.benchmarks import evaluate
from wslab.data import samples
from wslab.models import model_from_state


def baseline():
    return corrupted_model().state_dict()


def valid_patch(state=None):
    if state is None:
        state = baseline()
    return {"schema_version": 1, "task": "toy-linear-repair",
            "base_sha256": fingerprint(state), "edits": [
                {"tensor": "linear.weight", "index": [1, 0], "from": -2.0, "to": 2.0}]}


def test_edit_is_copy_with_exactly_one_scalar_change():
    base = baseline()
    sha = fingerprint(base)
    edited = apply_checked(base, valid_patch(base))
    assert fingerprint(base) == sha
    assert float(base["linear.weight"][1, 0]) == -2.0
    assert float(edited["linear.weight"][1, 0]) == 2.0
    assert torch.equal(base["linear.bias"], edited["linear.bias"])
    assert (base["linear.weight"] != edited["linear.weight"]).sum().item() == 1


@pytest.mark.parametrize("change", [
    {"base_sha256": "deadbeef"},
    {"schema_version": 99},
    {"edits": []},
    {"edits": [{"tensor": "linear.bias", "index": [0], "from": 0.0, "to": 1.0}]},
    {"edits": [{"tensor": "linear.weight", "index": [8, 0], "from": -2.0, "to": 2.0}]},
    {"edits": [{"tensor": "linear.weight", "index": [True, 0], "from": -2.0, "to": 2.0}]},
    {"edits": [{"tensor": "linear.weight", "index": [1, 0], "from": 3.0, "to": 2.0}]},
    {"edits": [{"tensor": "linear.weight", "index": [1, 0], "from": -2.0, "to": 30.0}]},
    {"edits": [{"tensor": "linear.weight", "index": [1, 0], "from": -2.0, "to": float("nan")}]},
    {"edits": [{"tensor": "linear.weight", "index": [1, 0], "from": -2.0, "to": -2.0}]},
])
def test_rejects_malformed_unsafe_or_stale_patches(change):
    original = baseline()
    p = valid_patch(original)
    p.update(change)
    with pytest.raises(PatchRejected):
        apply_checked(original, p)
    assert float(original["linear.weight"][1, 0]) == -2.0


def test_rejects_duplicate_coordinates():
    original = baseline()
    patch = valid_patch(original)
    patch["edits"].append(dict(patch["edits"][0]))
    with pytest.raises(PatchRejected):
        apply_checked(original, patch, PatchPolicy(max_edits=2, max_changed_fraction=0.5))


def test_rejects_regressive_patch():
    original = baseline()
    patch = valid_patch(original)
    patch["edits"][0]["to"] = -3.0
    modified = apply_checked(original, patch)
    val = samples("validation")
    with pytest.raises(PatchRejected):
        validation_gate(evaluate(model_from_state(original), val),
                        evaluate(model_from_state(modified), val))
