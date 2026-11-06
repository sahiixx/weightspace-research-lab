"""CLI: repeatable, inspectable experiments with no hidden network calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch
from .benchmarks import evaluate
from .checkpoints import fingerprint, load_checkpoint, read_json, save_checkpoint, write_json
from .data import samples
from .models import corrupted_model, model_from_state
from .pipeline import run_pipeline, verify_exact
from .safety import PatchRejected, apply_checked, validation_gate
from .search import propose
from .audit import run_sweep, run_negative_controls
from .data import manifest_for_split
from .trust import (create_key_pair, provenance, sign_patch, verify_signed_patch,
                    load_private_key, sign_audit, verify_signed_audit)
from .evidence import verify_audit_manifest
from .io_safety import UnsafePath
import secrets


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="wslab", description="Gradient-free weight editing research lab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pipeline", help="Run complete offline demo, produce benchmarks and a patch")
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count", type=int, default=2048)

    p = sub.add_parser("init", help="Create a deliberately corrupted initial checkpoint")
    p.add_argument("--out", default="runs/manual/base.pt")

    p = sub.add_parser("propose", help="Search one known parameter using validation data only")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/manual/patch.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count", type=int, default=2048)

    p = sub.add_parser("apply", help="Validate patch and apply it to a new checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patch", required=True)
    p.add_argument("--out", default="runs/manual/patched.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count", type=int, default=2048)
    p.add_argument("--trusted-keys", help="Required trusted Ed25519 public-key policy JSON")
    p.add_argument("--root", help="Allowed output root (default current directory)")

    p = sub.add_parser("benchmark", help="Score checkpoint on validation/test/audit (no fitting)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("validation", "test", "audit"), default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count", type=int, default=2048)
    p.add_argument("--out", help="Optional JSON results path")

    p = sub.add_parser("verify", help="Verify patched checkpoint matches exact approved patch")
    p.add_argument("--base", required=True)
    p.add_argument("--patched", required=True)
    p.add_argument("--patch", required=True)

    p = sub.add_parser("sweep", help="Run independent-seed robustness experiments (toy task)")
    p.add_argument("--out", default="runs/audit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--count", type=int, default=1024)

    p = sub.add_parser("negative-controls", help="Attempt and reject unsafe or regressive edits")
    p.add_argument("--out", default="runs/audit")

    p = sub.add_parser("inspect", help="Show tensor names, shapes and fingerprint")
    p.add_argument("--checkpoint", required=True)

    p = sub.add_parser("keygen", help="Generate an Ed25519 approver key (keep private key OUTSIDE repo)")
    p.add_argument("--private-key", required=True)
    p.add_argument("--trust-store", required=True)
    p.add_argument("--key-id", required=True)
    p.add_argument("--source-repo", required=True)

    p = sub.add_parser("sign", help="Create a signed, provenance-bound patch envelope")
    p.add_argument("--patch", required=True)
    p.add_argument("--private-key", required=True)
    p.add_argument("--key-id", required=True)
    p.add_argument("--actor", required=True)
    p.add_argument("--source-repo", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count", type=int, default=2048)
    p.add_argument("--out", required=True)

    p = sub.add_parser("audit-sign", help="Sign exact bytes of a completed audit manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--private-key", required=True)
    p.add_argument("--key-id", required=True)
    p.add_argument("--source-repo", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("audit-verify", help="Verify signed audit and all artifact hashes")
    p.add_argument("--manifest", required=True)
    p.add_argument("--attestation", required=True)
    p.add_argument("--trusted-keys", required=True)
    p.add_argument("--root", required=True)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "pipeline":
            report = run_pipeline(args.out, seed=args.seed, count=args.count)
            _print({"status": "success", "out": str(Path(args.out).resolve()),
                    "validation_base": report["base"]["validation"]["accuracy"],
                    "validation_patched": report["patched"]["validation"]["accuracy"],
                    "test_base": report["base"]["test"]["accuracy"],
                    "test_patched": report["patched"]["test"]["accuracy"],
                    "changed_elements": report["verification"]["changed_elements"]})
        elif args.cmd == "init":
            state = corrupted_model().state_dict()
            save_checkpoint(args.out, state)
            _print({"checkpoint": str(args.out), "sha256": fingerprint(state)})
        elif args.cmd == "propose":
            state = load_checkpoint(args.checkpoint)
            patch, evidence = propose(state, samples("validation", args.seed, args.count))
            write_json(args.out, patch)
            write_json(Path(args.out).with_suffix(".search.json"), evidence)
            _print({"patch": args.out, "edit": patch["edits"],
                    "candidate_accuracy": evidence["candidate_validation"]["accuracy"]})
        elif args.cmd == "apply":
            if Path(args.checkpoint).resolve() == Path(args.out).resolve():
                raise PatchRejected("Refusing to overwrite the original checkpoint")
            if Path(args.patch).resolve() == Path(args.out).resolve():
                raise PatchRejected("Refusing to overwrite the approved patch manifest")
            if not args.trusted_keys:
                raise PatchRejected("Unsigned patches are not authorized; provide --trusted-keys")
            # Verify signature, signer role, allowlisted source and dataset before
            # opening checkpoint or performing any candidate evaluation.
            signed = read_json(args.patch)
            policy = read_json(args.trusted_keys)
            expected_dataset = manifest_for_split("validation", args.seed, args.count)["sha256"]
            patch = verify_signed_patch(signed, policy, expected_dataset_sha256=expected_dataset)
            base = load_checkpoint(args.checkpoint)
            candidate = apply_checked(base, patch)
            validation = samples("validation", args.seed, args.count)
            before = evaluate(model_from_state(base), validation)
            after = evaluate(model_from_state(candidate), validation)
            validation_gate(before, after)
            output_root = Path(args.root) if args.root else Path.cwd()
            save_checkpoint(args.out, candidate, root=output_root)
            _print({"saved": args.out, "authenticated_signer": signed["signature"]["key_id"],
                    "verified": verify_exact(base, load_checkpoint(args.out), patch),
                    "validation_accuracy_before": before.accuracy,
                    "validation_accuracy_after": after.accuracy})
        elif args.cmd == "benchmark":
            base = load_checkpoint(args.checkpoint)
            result = evaluate(model_from_state(base), samples(args.split, args.seed, args.count)).to_dict()
            result.update({"checkpoint": args.checkpoint, "checkpoint_sha256": fingerprint(base),
                           "seed": args.seed, "count": args.count})
            if args.out:
                write_json(args.out, result)
            _print(result)
        elif args.cmd == "verify":
            result = verify_exact(load_checkpoint(args.base), load_checkpoint(args.patched),
                                  read_json(args.patch))
            _print(result)
        elif args.cmd == "sweep":
            result = run_sweep(args.out, base_seed=args.seed, seeds=args.seeds, count=args.count)
            _print({"status": "success", "out": str(Path(args.out).resolve()),
                    "seeds": result["seeds"], "summary": result["summary"]})
        elif args.cmd == "negative-controls":
            result = run_negative_controls(args.out)
            _print({"status": "success" if result["passed"] else "failed",
                    "rejected": result["rejected"], "controls": result["controls"],
                    "base_unchanged": result["base_unchanged"]})
            if not result["passed"]:
                raise PatchRejected("Some negative controls were incorrectly accepted")
        elif args.cmd == "keygen":
            create_key_pair(args.private_key, args.trust_store, args.key_id,
                            source_repo=args.source_repo)
            _print({"trust_store": args.trust_store, "private_key": args.private_key,
                    "warning": "Protect the key and trust store outside the repository; never commit private material"})
        elif args.cmd == "sign":
            unsigned = read_json(args.patch)
            ds_sha = manifest_for_split("validation", args.seed, args.count)["sha256"]
            meta = provenance(actor=args.actor, source_repo=args.source_repo,
                              source_commit=args.source_commit, base_sha256=unsigned["base_sha256"],
                              dataset_sha256=ds_sha, run_id=secrets.token_hex(12))
            signed = sign_patch(unsigned, meta, load_private_key(args.private_key), args.key_id)
            write_json(args.out, signed)
            _print({"signed": args.out, "key_id": args.key_id, "validation_sha256": ds_sha})
        elif args.cmd == "audit-sign":
            audit = read_json(args.manifest)
            if not verify_audit_manifest(Path(args.manifest).parent, audit):
                raise PatchRejected("Audit manifest integrity check failed")
            attestation = sign_audit(audit, load_private_key(args.private_key), args.key_id,
                                     args.source_repo, args.source_commit)
            write_json(args.out, attestation)
            _print({"attestation": args.out, "run_id": audit["run_id"]})
        elif args.cmd == "audit-verify":
            audit = read_json(args.manifest)
            attestation = read_json(args.attestation)
            verify_signed_audit(attestation, audit, read_json(args.trusted_keys))
            verify_audit_manifest(args.root, audit)
            _print({"verified": True, "run_id": audit["run_id"],
                    "signer": attestation["signature"]["key_id"]})
        elif args.cmd == "inspect":
            state = load_checkpoint(args.checkpoint)
            _print({"checkpoint": args.checkpoint, "sha256": fingerprint(state),
                    "parameters": sum(t.numel() for t in state.values()),
                    "tensors": {k: {"shape": list(t.shape), "dtype": str(t.dtype),
                                    "min": float(t.min()), "max": float(t.max())}
                                for k, t in sorted(state.items())}})
    except (PatchRejected, ValueError, FileNotFoundError, RuntimeError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2) from e


if __name__ == "__main__":
    main()
