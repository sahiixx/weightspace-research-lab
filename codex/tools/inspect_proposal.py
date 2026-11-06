#!/usr/bin/env python3
"""Non-authorizing toy validator for a proposed patch. No secrets, no writes."""
from __future__ import annotations
import argparse
import json
from wslab.checkpoints import load_checkpoint, read_json
from wslab.safety import apply_checked, validation_gate, PatchRejected
from wslab.data import samples
from wslab.benchmarks import evaluate
from wslab.models import model_from_state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count", type=int, default=2048)
    args = p.parse_args(argv)
    try:
        base = load_checkpoint(args.checkpoint)
        patch = read_json(args.patch)
        candidate = apply_checked(base, patch)
        validation = samples("validation", args.seed, args.count)
        before = evaluate(model_from_state(base), validation)
        after = evaluate(model_from_state(candidate), validation)
        validation_gate(before, after)
        result = {"validation_gate": "passed", "authorization": "NOT_APPROVED",
                  "before_accuracy": before.accuracy, "after_accuracy": after.accuracy}
    except (PatchRejected, OSError, RuntimeError, ValueError) as exc:
        result = {"validation_gate": "rejected", "authorization": "NOT_APPROVED", "error": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["validation_gate"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
