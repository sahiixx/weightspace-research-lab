#!/usr/bin/env python3
"""Compare real Qwen evaluation runs and, separately, sign/verify the result.

An unsigned comparison is evidence only. A signed record means a designated
key approved the exact report bytes, not that the model is safe or a GPU ran.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def read_report(path: Path) -> dict:
    result = json.loads(path.read_text("utf-8"))
    if type(result) is not dict:
        raise ValueError("Expected JSON object")
    return result


def compare(base: dict, edited: dict, *, weight_report: dict | None = None) -> dict:
    if base.get("kind") != "actual_gpu_inference_evaluation" or edited.get("kind") != "actual_gpu_inference_evaluation":
        raise ValueError("Both inputs must come from actual GPU inference; no preflight or published-log reports")
    if base.get("model_tag") != "base" or edited.get("model_tag") != "edited":
        raise ValueError("Model arm labels mismatched")
    if base.get("dataset_sha256") != edited.get("dataset_sha256"):
        raise ValueError("Evaluation dataset bytes do not match")
    if base.get("protocol") != edited.get("protocol"):
        raise ValueError("Inference settings differ")
    if base.get("source_pins") != edited.get("source_pins"):
        raise ValueError("Pinned source manifests differ")
    a, b = base.get("samples"), edited.get("samples")
    if type(a) is not list or type(b) is not list or not a or len(a) != len(b):
        raise ValueError("Mismatch in sample counts")
    keys_a = [(r.get("id"), r.get("sample_index")) for r in a]
    keys_b = [(r.get("id"), r.get("sample_index")) for r in b]
    if keys_a != keys_b or len(set(keys_a)) != len(keys_a):
        raise ValueError("Sample IDs and indices are not exactly aligned")
    if any(type(row.get("correct")) is not bool or type(row.get("limit_hit")) is not bool for row in a + b):
        raise ValueError("Invalid sample metrics")
    if weight_report is not None and weight_report.get("kind") != "real_checkpoint_bitwise_reconstruction_no_inference":
        raise ValueError("Weight report must contain the independently reconstructed checkpoint check")
    gain = sum(r["correct"] for r in b) - sum(r["correct"] for r in a)
    return {"kind": "paired_gpu_evaluation_comparison", "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": base["protocol"]["mode"],
            "dataset_sha256": base["dataset_sha256"], "source_pins": base.get("source_pins"),
            "protocol": base["protocol"], "number_of_samples": len(a),
            "base_accuracy_percent": round(100 * sum(r["correct"] for r in a) / len(a), 4),
            "edited_accuracy_percent": round(100 * sum(r["correct"] for r in b) / len(b), 4),
            "delta_accuracy_points": round(100 * gain / len(a), 4),
            "base_truncated": sum(r["limit_hit"] for r in a),
            "edited_truncated": sum(r["limit_hit"] for r in b),
            "corrected_samples": sum(not x["correct"] and y["correct"] for x, y in zip(a, b)),
            "regressed_samples": sum(x["correct"] and not y["correct"] for x, y in zip(a, b)),
            "weight_report_sha256": hashlib.sha256(json.dumps(weight_report, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if weight_report else None,
            "limitations": ["Smoke-mode scores cannot be compared to full published benchmarks",
                            "A signed report proves key endorsement of bytes, not correctness of measurements"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--base", type=Path, required=True)
    c.add_argument("--edited", type=Path, required=True)
    c.add_argument("--weight-report", type=Path)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("sign")
    s.add_argument("--report", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--private-key", type=Path, required=True)
    s.add_argument("--signer-id", required=True)
    s.add_argument("--source-commit", required=True)
    s.add_argument("--run-id", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--envelope", type=Path, required=True)
    v.add_argument("--policy", type=Path, required=True)
    v.add_argument("--report", type=Path, required=True)
    v.add_argument("--source-commit", required=True)
    a = p.parse_args()
    try:
        if a.cmd == "compare":
            obj = compare(read_report(a.base), read_report(a.edited),
                          weight_report=read_report(a.weight_report) if a.weight_report else None)
            with a.out.open("x", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, sort_keys=True); f.write("\n")
            print("Comparison written:", a.out)
        elif a.cmd == "sign":
            from sahiixx_trust_core import canonical_bytes, content_sha256, sign_envelope
            payload = read_report(a.report)
            if payload.get("kind") != "paired_gpu_evaluation_comparison":
                raise ValueError("Only paired GPU comparison reports can be approved")
            prov = {"actor": a.signer_id, "source_repo": "https://github.com/sahiixx/weightspace-research-lab",
                    "source_commit": a.source_commit, "run_id": a.run_id,
                    "artifact_sha256": content_sha256(canonical_bytes(payload)),
                    "created_at_utc": datetime.now(timezone.utc).isoformat()}
            env = sign_envelope(payload, prov, private_key_pem=a.private_key.read_bytes(),
                                signer_id=a.signer_id, purpose="qwen_evaluation")
            with a.out.open("x", encoding="utf-8") as f:
                json.dump(env, f, indent=2, sort_keys=True); f.write("\n")
            print("Operator-signed evidence written:", a.out)
        else:
            from sahiixx_trust_core import canonical_bytes, content_sha256, verify_envelope
            payload = read_report(a.report)
            env = read_report(a.envelope)
            value = verify_envelope(env, read_report(a.policy), purpose="qwen_evaluation",
                                    expected_source_commit=a.source_commit,
                                    expected_artifact_sha256=content_sha256(canonical_bytes(payload)))
            if canonical_bytes(payload) != canonical_bytes(value):
                raise ValueError("Approved report differs from current report")
            print("Signed report verified under supplied trusted policy")
        return 0
    except (OSError, ValueError, ImportError, KeyError) as exc:
        print("Evidence operation rejected:", exc, file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
