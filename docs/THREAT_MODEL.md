# Threat model and safety limitations

## Scope of assets

Assets are the base tensor checkpoint, JSON patch manifest, modified checkpoint, validation protocol and holdout benchmark. The default tool is intended for **single-user, locally trusted** research. It executes no foreign source code, no model-proposed Python, and no network operations.

## Implemented checks

- `torch.load(..., weights_only=True)` for loading a tensor-only, expected-shape state dictionary (still apply platform dependency updates).
- Canonical SHA-256 state fingerprint covers sorted tensor name, shape, dtype and raw bytes. A mismatch rejects a stale/incorrect patch.
- Tensor allowlist, index range and type checks, explicit old-value match, no-op and duplicate-coordinate rejection.
- Maximum edit count, total changed fraction, per-element delta and final magnitude; NaN/Infinity are rejected.
- Candidate tensors are cloned; base checkpoint is never modified in place.
- Validation accuracy improvement and predeclared subgroup regression gate.
- `verify` checks that every output tensor matches the exact state obtained by applying the approved patch to the base.
- Test split is never used by patch search or approval. Per-split metrics and candidate count are recorded in reports.

## Explicit non-goals

- A SHA-256 hash alone **does not authenticate** whoever created a patch and is not a digital signature. Use authenticated signing and a trust root for hostile environments.
- The input patch is a trusted local JSON file. The lab does not guarantee denial-of-service resistance against huge files or sandbox untrusted plugins/processes.
- No inference-facing network server, accounts, tenants, OAuth, or authorization system is supplied.
- A single synthetic holdout and two groups do **not establish fairness**, adversarial robustness, transferability or safety for real applications.
- Tiny edits can still produce backdoors, and edit-size limits alone are not a backdoor detector.
- No model license, model card, dataset license, or external code trust checks are automated for optional downloads.
- Docker's no-network runtime setting is defense-in-depth, **not a secure execution sandbox** for arbitrary malicious weight files or executable code.

## Before extending to real models

Pin exact source checkpoint revisions and licenses, validate tensor hashes against independently trusted provenance, use a physically or logically isolated sandbox, record a bounded compute budget, pre-register multiple benchmark families, test false-positive and backdoor regressions, use a fresh held-out test suite, and enforce rollback before any production deployment.
