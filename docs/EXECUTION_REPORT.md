# End-to-end execution evidence: v0.4 security hardening

**Date:** 2026-11-06  
**Scope:** small CPU-only, deterministic toy classifier with cryptographically signed **local** patch approval and signed audit attestation. This is not a reproduction of the published Qwen experiment.

## Tests actually executed

- Existing compatibility tests: **31/31 passed**, including previous local audit and CLI coverage.
- Added security/trust regression tests: **21/21 passed** (signature tampering, authorization, provenance binding, dataset mismatch, strict values, symlinks, path traversal, audit integrity).
- Direct signed CLI workflow: generated Ed25519 test key and trusted public-key policy outside repository; signed patch; verified trust/validation hash before application; saved exactly one changed weight; signed and verified evidence manifest. **Succeeded**. Test key material was deleted.
- Deterministic pipeline (256 samples per validation/test split): baseline test accuracy **60.55%**, after local edit **100.00%**. Dataset manifest, chained events, artifact digests and audit manifest verified.
- Ten deliberate malformed or regressive patches: **10/10 rejected** by the existing local policy.
- Python compilation: passed.

## Security guarantees that are *not* claimed

Ed25519 verifies the patch and audit manifest **only under a separately protected trusted key policy**. The signed `source_commit`, `generated_by`, and UTC timestamp are signer statements, not independent GitHub or human identity proof. The locally generated event chain is not immutable, and benchmark improvement on this toy rule is not adversarial model security certification.

## Not run

- Docker container execution: Docker daemon unavailable in this environment.
- Full published model download and Qwen GPU benchmark: external downloads/GPU unavailable.
- Remote GitHub CI for this hardened branch: requires branch publication and a GitHub Actions run.

For security assumptions and remaining threats, read [`SECURITY_AND_TRUST.md`](SECURITY_AND_TRUST.md) and [`THREAT_MODEL.md`](THREAT_MODEL.md). This file records the measured local results only.
