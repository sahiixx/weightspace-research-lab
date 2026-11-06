"""Independent, reusable trust core should require no torch imports."""
import base64
import copy
from datetime import datetime, timezone, timedelta
import json
import sys
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/sahiixx_trust_core/src"))
from sahiixx_trust_core import (
    TrustCoreError, canonical_bytes, content_sha256, sign_envelope,
    verify_envelope, append_event, verify_event_chain)

REPO = "https://github.com/sahiixx/weightspace-research-lab"
COMMIT = "a" * 40


def fixture(now=None):
    now = now or datetime.now(timezone.utc)
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.Raw,
                                          serialization.PublicFormat.Raw)
    policy = {"version": 1, "signers": [{
        "signer_id": "operator", "public_key_base64": base64.b64encode(pub).decode(),
        "active": True, "roles": ["model-patch", "config-change"], "source_repos": [REPO]
    }]}
    payload = {"operation": "review", "target": "test", "sha256": "f" * 64}
    meta = {"actor": "operator@example", "source_repo": REPO,
            "source_commit": COMMIT, "run_id": "test-1",
            "artifact_sha256": content_sha256(canonical_bytes(payload)),
            "created_at_utc": now.isoformat()}
    return key, pem, policy, payload, meta, now


def test_distinct_purposes_and_authenticated_payload():
    _, pem, policy, payload, meta, now = fixture()
    signed = sign_envelope(payload, meta, private_key_pem=pem, signer_id="operator", purpose="model-patch")
    assert verify_envelope(signed, policy, purpose="model-patch", now=now) == payload
    with pytest.raises(TrustCoreError, match="purpose"):
        verify_envelope(signed, policy, purpose="config-change", now=now)
    forged = copy.deepcopy(signed)
    forged["payload"]["operation"] = "deploy"
    with pytest.raises(TrustCoreError, match="signature"):
        verify_envelope(forged, policy, purpose="model-patch", now=now)


def test_key_revocation_and_source_allowlist():
    _, pem, policy, payload, meta, now = fixture()
    signed = sign_envelope(payload, meta, private_key_pem=pem, signer_id="operator", purpose="model-patch")
    revoked = copy.deepcopy(policy)
    revoked["signers"][0]["active"] = False
    with pytest.raises(TrustCoreError, match="authorized"):
        verify_envelope(signed, revoked, purpose="model-patch", now=now)
    other = copy.deepcopy(policy)
    other["signers"][0]["source_repos"] = ["https://example.net/other"]
    with pytest.raises(TrustCoreError, match="allowlisted"):
        verify_envelope(signed, other, purpose="model-patch", now=now)
    with pytest.raises(TrustCoreError, match="commit"):
        verify_envelope(signed, policy, purpose="model-patch", expected_source_commit="b" * 40, now=now)


def test_expired_and_nonfinite_rejected():
    _, pem, policy, payload, meta, now = fixture(datetime.now(timezone.utc) - timedelta(days=8))
    signed = sign_envelope(payload, meta, private_key_pem=pem, signer_id="operator", purpose="model-patch")
    with pytest.raises(TrustCoreError, match="Expired"):
        verify_envelope(signed, policy, purpose="model-patch")
    with pytest.raises(TrustCoreError):
        canonical_bytes({"bad": float("nan")})
    with pytest.raises(TrustCoreError):
        canonical_bytes({"bad": {1: "nonstring"}})


def test_audit_chain_detects_reordering_and_tampering():
    t = datetime.now(timezone.utc).isoformat()
    events = append_event([], action="proposed", details={"count": 1}, timestamp=t)
    events = append_event(events, action="validated", details={"ok": True}, timestamp=t)
    assert verify_event_chain(events) is True
    changed = copy.deepcopy(events)
    changed[0]["details"]["count"] = 2
    with pytest.raises(TrustCoreError, match="hash"):
        verify_event_chain(changed)
    with pytest.raises(TrustCoreError, match="chain"):
        verify_event_chain(events[::-1])


def test_sdk_module_has_no_torch_dependency():
    import ast
    root = Path(__file__).resolve().parents[1] / "packages/sahiixx_trust_core/src/sahiixx_trust_core/core.py"
    imports = [n for n in ast.walk(ast.parse(root.read_text())) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("torch" in ast.unparse(n) for n in imports)
