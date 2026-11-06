"""Ed25519 patch approvals and local trust-store policy.

A signed manifest binds bytes to a *key in a protected trust store*. It cannot
prove that the machine, signer or evaluation data was uncompromised.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone, timedelta
import json
import re
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .checkpoints import read_json, write_json
from .io_safety import safe_read_bytes, safe_write_bytes
from .safety import PatchRejected, validate_patch_schema

COMMIT = re.compile(r"^[0-9a-f]{40}$")
KEYID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _b64decode(value: object, size: int) -> bytes:
    if type(value) is not str:
        raise PatchRejected("Expected a Base64 string")
    try:
        result = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise PatchRejected("Invalid Base64 encoding") from exc
    if len(result) != size:
        raise PatchRejected("Incorrect cryptographic key or signature length")
    return result


def _timestamp(value: object) -> datetime:
    if type(value) is not str:
        raise PatchRejected("Timestamp must be a string")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PatchRejected("Invalid RFC3339 timestamp") from exc
    if timestamp.tzinfo is None:
        raise PatchRejected("Timestamp must carry a UTC offset")
    return timestamp.astimezone(timezone.utc)


def create_key_pair(private_path: str | Path, trust_path: str | Path,
                    key_id: str, *, source_repo: str) -> None:
    if not KEYID.fullmatch(key_id):
        raise ValueError("Invalid key ID")
    if not source_repo.startswith("https://github.com/"):
        raise ValueError("Trusted source must be an https GitHub repository URL")
    if Path(private_path).absolute() == Path(trust_path).absolute():
        raise ValueError("Private key and trust store must use different files")
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    public = key.public_key().public_bytes(serialization.Encoding.Raw,
                                           serialization.PublicFormat.Raw)
    # Files are created with O_EXCL/no-clobber permissions 0600.
    safe_write_bytes(private_path, raw)
    write_json(trust_path, {"schema_version": 1, "signers": [{
        "key_id": key_id, "algorithm": "Ed25519", "active": True,
        "roles": ["patch-approver", "audit-approver"], "source_repos": [source_repo],
        "public_key_base64": base64.b64encode(public).decode("ascii")
    }]})


def provenance(*, actor: str, source_repo: str, source_commit: str,
               base_sha256: str, dataset_sha256: str, run_id: str) -> dict:
    if not all(type(v) is str and v and len(v) <= 256 for v in
               (actor, source_repo, source_commit, base_sha256, dataset_sha256, run_id)):
        raise PatchRejected("Invalid provenance field")
    if not COMMIT.fullmatch(source_commit):
        raise PatchRejected("Source commit must be a 40-character lowercase SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", base_sha256) or not re.fullmatch(r"[0-9a-f]{64}", dataset_sha256):
        raise PatchRejected("Dataset and checkpoint hashes must be SHA-256")
    return {"generated_by": actor, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_repo": source_repo, "source_commit": source_commit,
            "base_sha256": base_sha256, "dataset_sha256": dataset_sha256,
            "run_id": run_id}


def validate_provenance(meta: object) -> dict:
    required = {"generated_by", "created_at_utc", "source_repo", "source_commit",
                "base_sha256", "dataset_sha256", "run_id"}
    if type(meta) is not dict or set(meta) != required:
        raise PatchRejected("Provenance requires precisely the documented fields")
    for k in required - {"created_at_utc"}:
        if type(meta[k]) is not str or not meta[k] or len(meta[k]) > 256:
            raise PatchRejected(f"Invalid provenance {k}")
    _timestamp(meta["created_at_utc"])
    if not COMMIT.fullmatch(meta["source_commit"]):
        raise PatchRejected("Invalid provenance source commit")
    for k in ("base_sha256", "dataset_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", meta[k]):
            raise PatchRejected(f"Invalid provenance {k}")
    return meta


def sign_patch(patch: dict, meta: dict, private_key_pem: bytes, key_id: str) -> dict:
    validate_patch_schema(patch, require_dtype=True)
    validate_provenance(meta)
    if patch["base_sha256"] != meta["base_sha256"]:
        raise PatchRejected("Patch and signed provenance disagree on base hash")
    if not KEYID.fullmatch(key_id):
        raise PatchRejected("Invalid signer key ID")
    try:
        key = serialization.load_pem_private_key(private_key_pem, password=None)
    except (ValueError, TypeError) as exc:
        raise PatchRejected("Invalid private key PEM") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise PatchRejected("Only Ed25519 signing keys are supported")
    body = {"envelope_version": 1, "patch": patch, "provenance": meta}
    signature = key.sign(canonical(body))
    return {**body, "signature": {"algorithm": "Ed25519", "key_id": key_id,
                                "value_base64": base64.b64encode(signature).decode("ascii")}}


def verify_signed_patch(envelope: object, trust_store: dict,
                        *, expected_dataset_sha256: str | None = None,
                        now: datetime | None = None) -> dict:
    if type(envelope) is not dict or set(envelope) != {"envelope_version", "patch", "provenance", "signature"}:
        raise PatchRejected("Unsigned or malformed signed patch envelope")
    if type(envelope["envelope_version"]) is not int or envelope["envelope_version"] != 1:
        raise PatchRejected("Unsupported envelope version")
    signature = envelope["signature"]
    if type(signature) is not dict or set(signature) != {"algorithm", "key_id", "value_base64"}:
        raise PatchRejected("Invalid signature object")
    if signature["algorithm"] != "Ed25519" or type(signature["key_id"]) is not str:
        raise PatchRejected("Unexpected signature algorithm or signer")
    if type(trust_store) is not dict or set(trust_store) != {"schema_version", "signers"} or type(trust_store["schema_version"]) is not int or trust_store["schema_version"] != 1 or type(trust_store["signers"]) is not list:
        raise PatchRejected("Invalid trusted-key policy")
    signers = [s for s in trust_store["signers"] if type(s) is dict and s.get("key_id") == signature["key_id"]]
    if len(signers) != 1:
        raise PatchRejected("Signer not found or ambiguous in trust store")
    signer = signers[0]
    if set(signer) != {"key_id", "algorithm", "active", "roles", "source_repos", "public_key_base64"} or signer["algorithm"] != "Ed25519" or signer["active"] is not True or type(signer["roles"]) is not list or "patch-approver" not in signer["roles"] or type(signer["source_repos"]) is not list:
        raise PatchRejected("Signer lacks active trusted patch-approver role")
    public = Ed25519PublicKey.from_public_bytes(_b64decode(signer["public_key_base64"], 32))
    sigbytes = _b64decode(signature["value_base64"], 64)
    body = {k: envelope[k] for k in ("envelope_version", "patch", "provenance")}
    try:
        public.verify(sigbytes, canonical(body))
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise PatchRejected("Signature verification failed") from exc
    # Only after authenticity and trusted authorization do we parse candidates.
    patch = validate_patch_schema(envelope["patch"], require_dtype=True)
    meta = validate_provenance(envelope["provenance"])
    if meta["source_repo"] not in signer["source_repos"]:
        raise PatchRejected("Provenance source repository not allowlisted for signer")
    if meta["base_sha256"] != patch["base_sha256"]:
        raise PatchRejected("Signed base hash does not match patch")
    if expected_dataset_sha256 is not None and meta["dataset_sha256"] != expected_dataset_sha256:
        raise PatchRejected("Evaluation dataset manifest fingerprint mismatch")
    utcnow = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    created = _timestamp(meta["created_at_utc"])
    if created > utcnow + timedelta(minutes=5) or utcnow - created > timedelta(days=30):
        raise PatchRejected("Signed patch timestamp outside the 30-day validity window")
    return patch


def load_private_key(path: str | Path) -> bytes:
    return safe_read_bytes(path, max_bytes=8192)


def sign_audit(audit: dict, private_key_pem: bytes, key_id: str,
               source_repo: str, source_commit: str) -> dict:
    from .evidence import sha
    if type(audit) is not dict or audit.get("schema_version") != 1:
        raise PatchRejected("Invalid audit manifest")
    if not KEYID.fullmatch(key_id) or not COMMIT.fullmatch(source_commit):
        raise PatchRejected("Invalid audit signer or source commit")
    try:
        key = serialization.load_pem_private_key(private_key_pem, password=None)
    except (ValueError, TypeError) as exc:
        raise PatchRejected("Invalid private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise PatchRejected("Only Ed25519 keys supported")
    body = {"attestation_version": 1, "audit_sha256": sha(canonical(audit)),
            "run_id": audit["run_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_repo": source_repo, "source_commit": source_commit}
    return {**body, "signature": {"algorithm": "Ed25519", "key_id": key_id,
                                "value_base64": base64.b64encode(key.sign(canonical(body))).decode("ascii")}}


def verify_signed_audit(attestation: dict, audit: dict, trust_store: dict) -> bool:
    from .evidence import sha
    expected = {"attestation_version", "audit_sha256", "run_id", "created_at_utc",
                "source_repo", "source_commit", "signature"}
    if type(attestation) is not dict or set(attestation) != expected or attestation["attestation_version"] != 1:
        raise PatchRejected("Malformed audit attestation")
    if attestation["audit_sha256"] != sha(canonical(audit)) or attestation["run_id"] != audit["run_id"]:
        raise PatchRejected("Audit attestation does not bind supplied manifest")
    sig = attestation["signature"]
    if type(sig) is not dict or set(sig) != {"algorithm", "key_id", "value_base64"} or sig["algorithm"] != "Ed25519":
        raise PatchRejected("Invalid audit signature")
    if type(trust_store) is not dict or trust_store.get("schema_version") != 1 or type(trust_store.get("signers")) is not list:
        raise PatchRejected("Invalid trust store")
    signers = [k for k in trust_store["signers"] if type(k) is dict and k.get("key_id") == sig["key_id"]]
    if len(signers) != 1:
        raise PatchRejected("Untrusted audit signer")
    signer = signers[0]
    if (signer.get("active") is not True or "audit-approver" not in signer.get("roles", []) or
            attestation["source_repo"] not in signer.get("source_repos", []) or
            not COMMIT.fullmatch(attestation["source_commit"])):
        raise PatchRejected("Audit signer or repository is not authorized")
    timestamp = _timestamp(attestation["created_at_utc"])
    if timestamp > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise PatchRejected("Audit attestation dated in the future")
    body = {k: attestation[k] for k in expected - {"signature"}}
    key = Ed25519PublicKey.from_public_bytes(_b64decode(signer["public_key_base64"], 32))
    try:
        key.verify(_b64decode(sig["value_base64"], 64), canonical(body))
    except (InvalidSignature, ValueError) as exc:
        raise PatchRejected("Audit attestation signature invalid") from exc
    return True
