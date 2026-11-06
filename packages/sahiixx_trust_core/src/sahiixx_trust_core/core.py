"""Signed approval envelopes and audit chains; no ML or network dependencies.

Important: source repo and actor are *signed assertions*, not verified identities.
The trust store, private key and source code require independent protection.
"""
from __future__ import annotations
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
import re
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

DOMAIN = "sahiixx.trust-core.approval.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
KEY_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
PROVENANCE_KEYS = {"actor", "source_repo", "source_commit", "run_id", "artifact_sha256", "created_at_utc"}

class TrustCoreError(ValueError):
    """Invalid, unauthorized or unverifiable approval."""


def canonical_bytes(data: Any) -> bytes:
    """Stable UTF-8 JSON without NaN, non-string keys or ambiguous types."""
    def check(node: Any) -> None:
        if node is None or type(node) in (str, int, bool):
            return
        if type(node) is float:
            import math
            if math.isfinite(node):
                return
        if type(node) is list:
            for v in node: check(v)
            return
        if type(node) is dict:
            for k, v in node.items():
                if type(k) is not str: raise TrustCoreError("Object keys must be strings")
                check(v)
            return
        raise TrustCoreError("Non-JSON or non-finite value")
    check(data)
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TrustCoreError("Cannot canonicalize JSON") from exc


def content_sha256(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TrustCoreError("Artifact hash input must be bytes")
    return hashlib.sha256(data).hexdigest()


def _utc(timestamp: str) -> datetime:
    if type(timestamp) is not str:
        raise TrustCoreError("created_at_utc must be a timestamp")
    try:
        result = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrustCoreError("Malformed timestamp") from exc
    if result.tzinfo is None:
        raise TrustCoreError("Timezone required")
    return result.astimezone(timezone.utc)


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    if type(provenance) is not dict or set(provenance) != PROVENANCE_KEYS:
        raise TrustCoreError("Invalid provenance schema")
    for field in PROVENANCE_KEYS:
        if type(provenance[field]) is not str or not 1 <= len(provenance[field]) <= 256:
            raise TrustCoreError(f"Invalid provenance field: {field}")
    if not COMMIT.fullmatch(provenance["source_commit"]):
        raise TrustCoreError("source_commit requires full lowercase Git SHA-1")
    if not SHA256.fullmatch(provenance["artifact_sha256"]):
        raise TrustCoreError("Invalid artifact_sha256")
    if not provenance["source_repo"].startswith("https://"):
        raise TrustCoreError("source_repo must use https")
    _utc(provenance["created_at_utc"])


def sign_envelope(payload: dict, provenance: dict, *, private_key_pem: bytes,
                  signer_id: str, purpose: str) -> dict:
    """Operator-side only. Never expose private_key_pem to an autonomous agent."""
    if type(payload) is not dict:
        raise TrustCoreError("Payload must be an object")
    if type(purpose) is not str or not 1 <= len(purpose) <= 80:
        raise TrustCoreError("Invalid purpose")
    if not KEY_ID.fullmatch(signer_id):
        raise TrustCoreError("Invalid signer ID")
    _validate_provenance(provenance)
    if content_sha256(canonical_bytes(payload)) != provenance["artifact_sha256"]:
        raise TrustCoreError("Provenance digest does not match payload")
    try:
        key = serialization.load_pem_private_key(private_key_pem, password=None)
    except (TypeError, ValueError) as exc:
        raise TrustCoreError("Invalid private Ed25519 key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise TrustCoreError("Only Ed25519 keys supported")
    body = {"domain": DOMAIN, "version": 1, "purpose": purpose,
            "payload": payload, "provenance": provenance}
    sig = key.sign(canonical_bytes(body))
    return {**body, "signature": {"algorithm": "Ed25519", "signer_id": signer_id,
            "value_base64": base64.b64encode(sig).decode("ascii")}}


def verify_envelope(envelope: Any, trusted_policy: dict, *, purpose: str,
                    expected_artifact_sha256: str | None = None,
                    expected_source_commit: str | None = None,
                    now: datetime | None = None,
                    max_age: timedelta = timedelta(days=7)) -> dict:
    """Verify a signed envelope against an independently provisioned trusted policy.

    Policy: {"version":1,"signers":[{"signer_id", "public_key_base64",
    "active", "roles", "source_repos"}]}. 
    """
    required = {"domain", "version", "purpose", "payload", "provenance", "signature"}
    if type(envelope) is not dict or set(envelope) != required:
        raise TrustCoreError("Missing/unknown signed envelope fields")
    if envelope["domain"] != DOMAIN or type(envelope["version"]) is not int or envelope["version"] != 1:
        raise TrustCoreError("Wrong signature domain/version")
    if envelope["purpose"] != purpose:
        raise TrustCoreError("Wrong approval purpose")
    sig = envelope["signature"]
    if type(sig) is not dict or set(sig) != {"algorithm", "signer_id", "value_base64"}:
        raise TrustCoreError("Malformed signature structure")
    if sig["algorithm"] != "Ed25519" or type(sig["signer_id"]) is not str or not KEY_ID.fullmatch(sig["signer_id"]):
        raise TrustCoreError("Unknown signature algorithm or signer")
    if type(trusted_policy) is not dict or set(trusted_policy) != {"version", "signers"} or type(trusted_policy["version"]) is not int or trusted_policy["version"] != 1 or type(trusted_policy["signers"]) is not list:
        raise TrustCoreError("Invalid trusted policy")
    signers = [s for s in trusted_policy["signers"] if type(s) is dict and s.get("signer_id") == sig["signer_id"]]
    if len(signers) != 1:
        raise TrustCoreError("Unknown or duplicate signer")
    signer = signers[0]
    if set(signer) != {"signer_id", "public_key_base64", "active", "roles", "source_repos"} or signer["active"] is not True or type(signer["roles"]) is not list or purpose not in signer["roles"] or type(signer["source_repos"]) is not list:
        raise TrustCoreError("Signer not authorized for purpose")
    def b64(value: Any, length: int) -> bytes:
        if type(value) is not str:
            raise TrustCoreError("Invalid key/signature encoding")
        try: result = base64.b64decode(value, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise TrustCoreError("Invalid base64") from exc
        if len(result) != length: raise TrustCoreError("Invalid key/signature size")
        return result
    try:
        Ed25519PublicKey.from_public_bytes(b64(signer["public_key_base64"],32)).verify(
            b64(sig["value_base64"],64),canonical_bytes({k: envelope[k] for k in required - {"signature"}}))
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise TrustCoreError("Invalid signature") from exc
    meta = envelope["provenance"]
    _validate_provenance(meta)
    if meta["source_repo"] not in signer["source_repos"]:
        raise TrustCoreError("Source repository not allowlisted")
    if expected_source_commit is not None and meta["source_commit"] != expected_source_commit:
        raise TrustCoreError("Source commit mismatch")
    digest = content_sha256(canonical_bytes(envelope["payload"]))
    if digest != meta["artifact_sha256"] or (expected_artifact_sha256 is not None and digest != expected_artifact_sha256):
        raise TrustCoreError("Artifact hash mismatch")
    t = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    created = _utc(meta["created_at_utc"])
    if created > t + timedelta(minutes=5) or t - created > max_age:
        raise TrustCoreError("Expired or future signature")
    return envelope["payload"]


def append_event(events: list[dict], *, action: str, details: dict,
                 timestamp: str) -> list[dict]:
    """Pure, hash-chained events (not immutable storage)."""
    verify_event_chain(events)
    if type(action) is not str or not 1 <= len(action) <= 80:
        raise TrustCoreError("Invalid event action")
    _utc(timestamp)
    previous = events[-1]["event_sha256"] if events else "0" * 64
    entry = {"index": len(events), "previous_sha256": previous, "action": action,
             "details": details, "timestamp": timestamp}
    entry["event_sha256"] = content_sha256(canonical_bytes(entry))
    return [*events, entry]


def verify_event_chain(events: Any) -> bool:
    if type(events) is not list:
        raise TrustCoreError("Events must be an array")
    prior = "0" * 64
    for index, entry in enumerate(events):
        if type(entry) is not dict or set(entry) != {"index", "previous_sha256", "action", "details", "timestamp", "event_sha256"}:
            raise TrustCoreError("Malformed audit event")
        if type(entry["index"]) is not int or entry["index"] != index or entry["previous_sha256"] != prior:
            raise TrustCoreError("Broken audit chain")
        if type(entry["details"]) is not dict or type(entry["action"]) is not str:
            raise TrustCoreError("Invalid audit details")
        _utc(entry["timestamp"])
        raw = {k: entry[k] for k in entry if k != "event_sha256"}
        if content_sha256(canonical_bytes(raw)) != entry["event_sha256"]:
            raise TrustCoreError("Event hash mismatch")
        prior = entry["event_sha256"]
    return True
