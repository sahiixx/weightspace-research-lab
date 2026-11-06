"""Locally inspectable evidence-chain and artifact digest tree.

Neither file hashes nor mutable local JSON offer immutability or authenticity.
For an authenticity check, create a detached Ed25519 audit attestation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import sys
import uuid
import torch

from .checkpoints import read_json, write_json
from .io_safety import safe_read_bytes
from .trust import canonical


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_audit_manifest(directory: str | Path, events: list[dict],
                         filenames: list[str], run_id: str | None = None) -> dict:
    """Record exact artifact bytes, event-chain hashes and runtime metadata."""
    directory = Path(directory)
    records = []
    for filename in sorted(filenames):
        if Path(filename).name != filename or filename.startswith("."):
            raise ValueError("Only flat, nonhidden generated artifact filenames are supported")
        b = safe_read_bytes(directory / filename, root=directory)
        records.append({"path": filename, "size": len(b), "sha256": sha(b)})
    leaves = [bytes.fromhex(x["sha256"]) for x in records]
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [hashlib.sha256(leaves[i] + leaves[i+1]).digest()
                  for i in range(0, len(leaves), 2)]
    chain = []
    previous = "0" * 64
    for index, event in enumerate(events):
        item = {"index": index, "name": event["name"], "data": event["data"],
                "previous_sha256": previous}
        item["entry_sha256"] = sha(canonical(item))
        chain.append(item)
        previous = item["entry_sha256"]
    return {"schema_version": 1, "run_id": run_id or uuid.uuid4().hex,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cli_version": "0.4.0", "environment": {
                "python": sys.version.split()[0], "torch": torch.__version__,
                "platform": platform.platform()},
            "artifacts": records,
            "merkle_root_sha256": leaves[0].hex() if leaves else sha(b""),
            "events": chain, "last_event_sha256": previous,
            "note": "Integrity-only local evidence; use signed attestation for signer authenticity"}


def verify_audit_manifest(directory: str | Path, audit: dict) -> bool:
    if type(audit) is not dict or audit.get("schema_version") != 1:
        raise ValueError("Unsupported audit manifest")
    names = []
    for record in audit["artifacts"]:
        filename = record["path"]
        if type(filename) is not str or Path(filename).name != filename or filename.startswith("."):
            raise ValueError("Unsafe artifact path")
        if filename in names:
            raise ValueError("Duplicate artifact path")
        names.append(filename)
        data = safe_read_bytes(Path(directory) / filename, root=directory)
        if len(data) != record["size"] or sha(data) != record["sha256"]:
            raise ValueError(f"Artifact failed SHA-256 verification: {filename}")
    prior = "0" * 64
    for index, event in enumerate(audit["events"]):
        if event["index"] != index or event["previous_sha256"] != prior:
            raise ValueError("Event chain sequence or parent mismatch")
        expected = {k: event[k] for k in ("index", "name", "data", "previous_sha256")}
        if set(event) != {*expected, "entry_sha256"} or sha(canonical(expected)) != event["entry_sha256"]:
            raise ValueError("Event chain hash mismatch")
        prior = event["entry_sha256"]
    if audit["last_event_sha256"] != prior:
        raise ValueError("Event chain head mismatch")
    leaves = [bytes.fromhex(record["sha256"]) for record in audit["artifacts"]]
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [hashlib.sha256(leaves[i] + leaves[i+1]).digest() for i in range(0, len(leaves), 2)]
    if (leaves[0].hex() if leaves else sha(b"")) != audit["merkle_root_sha256"]:
        raise ValueError("Artifact Merkle root mismatch")
    return True
