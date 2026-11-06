# Threat Model and Security Boundaries

Read [Security and Trust Model](SECURITY_AND_TRUST.md) for the concrete authentication protocol and limitations.

## In scope
- Malformed patches, unexpected fields, unsafe indices, non-finite numbers, booleans disguised as numeric fields, float32 rounding errors, wrong dtypes/devices and duplicate coordinates.
- Stale SHA-256 checkpoint fingerprints, wrong original values and changes outside the allowed tiny tensor subset.
- Untrusted or revoked signers, unauthorized source repositories, tampered signed content, validation-dataset mismatches and expired patch approvals.
- Accidental file overwrites, common symlink/path-traversal attacks, duplicate-key JSON and artifact edits after an audit manifest is created.
- Performance regressions detected by local deterministic validation and predefined group metrics.

## Out of scope
- Fully malicious model checkpoints, unknown layers, giant pretrained models, arbitrary file formats, executable pickle payloads or remote model serving.
- Same-user filesystem adversaries who can alter the trusted root, private signing keys, public-key policy, benchmark source, Python interpreter or test results.
- Data poisoning resistant evaluation, inference-time backdoors, model fairness and alignment guarantees, side-channel defenses or formal proofs of improved mathematical reasoning.
- Immutable evidence, external timestamp authorities, identity providers, multi-tenant process isolation, deployment security and production-grade key management.

## Reproducible mitigation tests

`tests/test_trust_boundaries.py` covers signed round trips, mutation after signature, unknown/revoked signers, source repository allowlist, validation dataset hash, unsigned CLI rejection, exact dtype coercion rejection, float32 representability, output path confinement, symlink and overwrite rejection, JSON duplicates, and evidence changes after signing.

These tests demonstrate fail-closed behavior under specified assumptions, **not** production security certification.
