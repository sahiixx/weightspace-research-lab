"""Finite coordinate search for a single weight, with a fixed validation budget.

This is gradient-free programmatic search, *not* the published Astra editor.
The coordinate is given to the algorithm; it does not discover a causal circuit.
"""
from __future__ import annotations

import torch
from .benchmarks import evaluate, Metrics
from .checkpoints import clone_state, fingerprint
from .data import Samples
from .models import model_from_state
from .safety import PatchRejected, PatchPolicy, apply_checked, validation_gate

TARGET_TENSOR = "linear.weight"
TARGET_INDEX = [1, 0]
CANDIDATES = tuple(float(i) / 4.0 for i in range(-16, 17))


def propose(base: dict[str, torch.Tensor], validation: Samples,
            policy: PatchPolicy = PatchPolicy()) -> tuple[dict, dict]:
    if validation.split != "validation":
        raise ValueError("Search may inspect only the validation split")
    baseline = evaluate(model_from_state(base), validation)
    before = float(base[TARGET_TENSOR][tuple(TARGET_INDEX)].item())
    best_patch = None
    best_metrics = None
    # Tie-breaking prefers the smallest possible absolute edit.
    best_key = (baseline.accuracy, float("-inf"))
    evaluated = 0
    for value in CANDIDATES:
        if value == before:
            continue
        patch = {
            "schema_version": 1,
            "task": "toy-linear-repair",
            "base_sha256": fingerprint(base),
            "method": "bounded-coordinate-grid-search-not-llm",
            "edits": [{"tensor": TARGET_TENSOR, "index": TARGET_INDEX,
                       "from": before, "to": value, "dtype": "torch.float32", "device": "cpu",
                       "reason": "Candidate assessed on synthetic validation samples"}],
        }
        try:
            candidate = apply_checked(base, patch, policy)
        except PatchRejected:
            continue
        evaluated += 1
        metric = evaluate(model_from_state(candidate), validation)
        key = (metric.accuracy, -abs(value-before))
        if key > best_key:
            best_key, best_patch, best_metrics = key, patch, metric
    if best_patch is None or best_metrics is None:
        raise PatchRejected("No permitted candidate improves the validation benchmark")
    validation_gate(baseline, best_metrics, policy)
    return best_patch, {
        "candidates_evaluated": evaluated,
        "baseline_validation": baseline.to_dict(),
        "candidate_validation": best_metrics.to_dict(),
        "selection_split": "validation",
        "search_algorithm": "fixed grid over one known weight; no gradients, no LLM",
    }
