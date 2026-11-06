"""Tests for authenticity, stricter shape/type limits and evidence tampering."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from wslab.checkpoints import (read_json, write_json, save_checkpoint, load_checkpoint, fingerprint)
from wslab.data import samples, dataset_manifest, manifest_for_split
from wslab.evidence import verify_audit_manifest
from wslab.io_safety import safe_read_bytes, safe_write_bytes, UnsafePath
from wslab.models import corrupted_model
from wslab.pipeline import run_pipeline
from wslab.safety import PatchRejected, PatchPolicy, apply_checked
from wslab.search import propose
from wslab.trust import (create_key_pair, load_private_key, provenance, sign_patch,
                         verify_signed_patch, sign_audit, verify_signed_audit)

REPO = "https://github.com/sahiixx/weightspace-research-lab"
COMMIT = "a" * 40


def signer(tmp_path):
    secret = tmp_path / "outside" / "approver.pem"
    trust = tmp_path / "outside" / "trusted.json"
    create_key_pair(secret, trust, "reviewer-1", source_repo=REPO)
    return secret, trust


def envelope(tmp_path, count=128):
    original = corrupted_model().state_dict()
    patch, _ = propose(original, samples("validation", count=count))
    secret, store = signer(tmp_path)
    meta = provenance(actor="test-operator", source_repo=REPO, source_commit=COMMIT,
                      base_sha256=fingerprint(original),
                      dataset_sha256=manifest_for_split("validation", 42, count)["sha256"],
                      run_id="test-run")
    signed = sign_patch(patch, meta, load_private_key(secret), "reviewer-1")
    return original, signed, read_json(store)


def test_signed_patch_roundtrip_and_authenticated_payload(tmp_path):
    base, signed, store = envelope(tmp_path)
    parsed = verify_signed_patch(signed, store,
                                  expected_dataset_sha256=signed["provenance"]["dataset_sha256"])
    result = apply_checked(base, parsed)
    assert fingerprint(result) != fingerprint(base)
    for field, replacement in (("source_repo", "https://github.com/attacker/forged"),
                               ("base_sha256", "0" * 64),
                               ("generated_by", "unknown")):
        mutated = copy.deepcopy(signed)
        mutated["provenance"][field] = replacement
        with pytest.raises(PatchRejected, match="Signature"):
            verify_signed_patch(mutated, store)
    mutated = copy.deepcopy(signed)
    mutated["patch"]["edits"][0]["to"] = 1.5
    with pytest.raises(PatchRejected, match="Signature"):
        verify_signed_patch(mutated, store)


def test_unknown_revoked_unauthorized_and_dataset_mismatch(tmp_path):
    _, signed, store = envelope(tmp_path)
    bad = copy.deepcopy(store)
    bad["signers"][0]["active"] = False
    with pytest.raises(PatchRejected, match="active"):
        verify_signed_patch(signed, bad)
    bad = copy.deepcopy(store)
    bad["signers"][0]["source_repos"] = ["https://github.com/example/untrusted"]
    with pytest.raises(PatchRejected, match="allowlisted"):
        verify_signed_patch(signed, bad)
    bad = copy.deepcopy(store)
    bad["signers"][0]["key_id"] = "wrong"
    with pytest.raises(PatchRejected, match="Signer"):
        verify_signed_patch(signed, bad)
    with pytest.raises(PatchRejected, match="dataset"):
        verify_signed_patch(signed, store, expected_dataset_sha256="1" * 64)
    bad = copy.deepcopy(signed)
    bad["signature"]["value_base64"] = "Z" * 88
    with pytest.raises(PatchRejected):
        verify_signed_patch(bad, store)


def test_unsigned_cli_apply_is_never_authorized(tmp_path):
    state = corrupted_model().state_dict()
    patch, _ = propose(state, samples("validation", count=128))
    ckpt = tmp_path / "base.pt"
    untrusted = tmp_path / "patch.json"
    save_checkpoint(ckpt, state)
    write_json(untrusted, patch)
    out = tmp_path / "patched.pt"
    run = subprocess.run([sys.executable, "-m", "wslab", "apply", "--checkpoint", str(ckpt),
                          "--patch", str(untrusted), "--out", str(out)],
                         text=True, capture_output=True)
    assert run.returncode != 0
    assert "unsigned" in run.stderr.lower()
    assert not out.exists()


def test_cli_signed_apply_and_signed_audit_roundtrip(tmp_path):
    root = tmp_path / "output"
    run_pipeline(root, seed=42, count=256)
    secret, trusted = signer(tmp_path)
    patch = read_json(root / "patch.json")
    meta = provenance(actor="test-operator", source_repo=REPO, source_commit=COMMIT,
                      base_sha256=patch["base_sha256"],
                      dataset_sha256=manifest_for_split("validation", 42, 256)["sha256"],
                      run_id="verified-256")
    signed = sign_patch(patch, meta, load_private_key(secret), "reviewer-1")
    write_json(root / "signed_patch.json", signed, root=root)
    output = root / "approved.pt"
    run = subprocess.run([sys.executable, "-m", "wslab", "apply",
                          "--checkpoint", str(root / "base.pt"),
                          "--patch", str(root / "signed_patch.json"),
                          "--trusted-keys", str(trusted),
                          "--out", str(output), "--root", str(root),
                          "--seed", "42", "--count", "256"], text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert fingerprint(load_checkpoint(output)) == fingerprint(load_checkpoint(root / "patched.pt"))
    audit = read_json(root / "audit_manifest.json")
    assert verify_audit_manifest(root, audit)
    attested = sign_audit(audit, load_private_key(secret), "reviewer-1", REPO, COMMIT)
    write_json(root / "audit_signed.json", attested)
    assert verify_signed_audit(read_json(root / "audit_signed.json"), audit, read_json(trusted))
    bad = copy.deepcopy(audit)
    bad["artifacts"][0]["sha256"] = "f" * 64
    with pytest.raises(PatchRejected, match="does not bind"):
        verify_signed_audit(attested, bad, read_json(trusted))
    # Deliberately corrupt one artifact after the attestation was created.
    (root / "report.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact"):
        verify_audit_manifest(root, audit)


@pytest.mark.parametrize("edit", [
    {"index": ["1", 0]}, {"index": [-1, 0]},
    {"to": True}, {"to": "0.1"}, {"to": " NaN "}, {"to": float("inf")},
    {"dtype": "torch.float64"}, {"device": "cuda:0"},
    {"to": 0.1}, {"to": 10**400}, {"extra": 1},
])
def test_strict_edit_schema_and_float32_roundtrip(edit):
    base = corrupted_model().state_dict()
    patch, _ = propose(base, samples("validation", count=128))
    patch["edits"][0].update(edit)
    with pytest.raises(PatchRejected):
        apply_checked(base, patch)


def test_actual_fraction_bound_and_not_record_count():
    base = corrupted_model().state_dict()
    patch, _ = propose(base, samples("validation", count=128))
    with pytest.raises(PatchRejected, match="fraction"):
        apply_checked(base, patch, PatchPolicy(max_changed_fraction=0.01))
    assert apply_checked(base, patch, PatchPolicy(max_changed_fraction=0.25))


def test_schema_unknown_keys_and_bool_version_rejected():
    base = corrupted_model().state_dict()
    patch, _ = propose(base, samples("validation", count=128))
    patch["admin_override"] = True
    with pytest.raises(PatchRejected, match="Unknown"):
        apply_checked(base, patch)


def test_dataset_composition_is_hashed():
    sample = samples("validation", count=128)
    manifest = dataset_manifest(sample, seed=42, count=128)
    assert manifest == dataset_manifest(samples("validation", count=128), seed=42, count=128)
    assert manifest != manifest_for_split("validation", 42, 129)
    sample.x[0, 0] += 0.5
    assert dataset_manifest(sample, seed=42, count=128)["sha256"] != manifest["sha256"]


def test_atomic_write_no_overwrite_symlink_or_traversal(tmp_path):
    root = tmp_path / "safe"
    root.mkdir()
    output = root / "check.json"
    write_json(output, {"ok": True}, root=root)
    assert read_json(output, root=root) == {"ok": True}
    with pytest.raises(UnsafePath, match="overwrite"):
        write_json(output, {"ok": False}, root=root)
    assert read_json(output, root=root) == {"ok": True}
    assert (output.stat().st_mode & 0o077) == 0
    with pytest.raises(UnsafePath):
        write_json(root / ".." / "escape.json", {"no": "way"}, root=root)
    (root / "redirect").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises((UnsafePath, OSError)):
        write_json(root / "redirect" / "escape.json", {"no": "way"}, root=root)
    (root / "alias.json").symlink_to(output)
    with pytest.raises((UnsafePath, OSError)):
        read_json(root / "alias.json", root=root)
    with pytest.raises((UnsafePath, OSError)):
        write_json(root / "alias.json", {"ok": False}, root=root)
    assert not (tmp_path / "escape.json").exists()


def test_duplicate_json_keys_and_nan_forbidden(tmp_path):
    path = tmp_path / "duplicate.json"
    safe_write_bytes(path, b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(path)
    path2 = tmp_path / "nan.json"
    safe_write_bytes(path2, b'{"a":NaN}')
    with pytest.raises(ValueError, match="Non-finite"):
        read_json(path2)


def test_external_download_helper_requires_trusted_commit_pin():
    helper = Path(__file__).resolve().parents[1] / "scripts" / "get_official_resources.sh"
    env = os.environ.copy()
    env.pop("OFFICIAL_LOGS_COMMIT", None)
    result = subprocess.run(["bash", str(helper)], capture_output=True, text=True, env=env)
    assert result.returncode == 2
    assert "OFFICIAL_LOGS_COMMIT" in result.stderr
    env["OFFICIAL_LOGS_COMMIT"] = "not-a-valid-sha"
    result = subprocess.run(["bash", str(helper)], capture_output=True, text=True, env=env)
    assert result.returncode == 2
