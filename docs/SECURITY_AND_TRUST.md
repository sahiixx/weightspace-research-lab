# Security and Trust Model (v0.4)

This repository is a **toy research laboratory**, not an authorization service, model sandbox, malicious-model scanner, or production approval platform. Version 0.4 intentionally distinguishes **local validation** from **cryptographic approval**.

## Trust boundaries

| Layer | What it establishes | What it does **not** establish |
|---|---|---|
| Canonical SHA-256 checkpoint fingerprint | Exactly which local tensor bytes an edit was written against | Who supplied the checkpoint or whether it contains a backdoor |
| Strict patch schema / `apply_checked` | Typed structure, index bounds, CPU float32 round-trip, actual changed coordinate count | Permission to perform the edit |
| Ed25519 signed envelope | A holder of an approved private key signed the complete patch + provenance | Human identity, key-holder intent, sound model behavior or a genuine source commit |
| Trusted approver policy | The key ID, role and declared source repository are allowlisted by the **local trust-store file** | Protection if an attacker can modify that policy file or run code as its owner |
| Validation dataset SHA-256 | Signed approval binds the exact synthetic input/label/group bytes and generation parameters | External data reliability, causal improvement, adversarial robustness or data poisoning resistance |
| Atomic no-clobber local writes | Prevents accidental overwrites and common symlink/traversal mistakes on a trusted POSIX filesystem | A hostile same-user process or attacker who can replace the trusted root directory |
| Evidence hash list and event chain | Locally inspectable, reproducible artifact bytes and event ordering | Append-only storage or independent time/identity proof |
| Signed audit attestation | An approver signed the audit manifest digest and run ID | Durable transparency log, hardware-rooted identity, or immutable timestamps |

**Threat actors:** untrusted patch contributors, accidental user edits, stale checkpoints, malformed JSON, symlinked paths, altered artifacts. The workflow is designed to *reject or detect* these when the signing key, trust-store policy, source code and trusted output directory are protected. An attacker who controls the private key, trust store, Python interpreter, host filesystem, GitHub workflow or evaluation dataset generator is **outside scope**.

## Patch approval workflow

The CLI command `wslab apply` **rejects unsigned legacy `patch.json` by default**. An unsigned patch may be proposed and checked with `apply_checked` or the offline `pipeline` educational workflow, but that only demonstrates mathematical constraints, not authorization.

A trusted operator generates an Ed25519 signing key **outside the repository**, protects it with operating-system permissions, and installs an allowlisted `trusted.json` public-key policy from a trusted channel. The policy has a key ID, active state, permitted roles and exact allowed source repository URLs. Keep the trust store in a path untrusted contributors cannot edit; never take it from an untrusted patch bundle.

The `sign` command binds the signature to:
- exact patch payload, including each `tensor`, `index`, `from`, `to`, `dtype`, `device`
- generated-by identifier (signed assertion, **not** authenticated human identity)
- UTC creation time (checked against a 30-day validity window for patches)
- exact original checkpoint SHA-256, run ID, validation dataset SHA-256
- asserted source repository URL and 40-character commit SHA

The `apply` command verifies signature and trusted key role, source allowlist and dataset hash **before reading the checkpoint or running validation**. It then enforces the toy patch policy and evaluation gate and writes the approved output under an explicit `--root` with no-clobber atomic file operations.

The source commit value is signed metadata **not a verified GitHub attestation**. To validate commit provenance, use pinned Git history and independently controlled CI/build attestations. A key holder may authorize a malicious patch. Crypto verification is necessary but not sufficient.

## Reproducible data and audit trail

`pipeline` produces `dataset_manifest.json` with the seed, exact split, generator offset, sample count, tensor shapes/dtypes and SHA-256 for `x`, `y`, `groups`. It also writes `audit_manifest.json` containing run ID, timestamp, CLI version, environment, per-file byte hashes, a Merkle-style digest root, and chained event hashes. These are not immutable: a local user who can rewrite all files can recompute the hashes.

To add **signer authenticity**, produce `audit_signed.json` with `wslab audit-sign` and validate it with `wslab audit-verify` using a protected `trusted.json`. To obtain an externally tamper-evident trail, export the signed manifest to immutable storage or a separate transparency-log service with protected write permissions and independently verifiable timestamps.

## I/O and external models

Input reads reject symlinks and duplicate JSON keys and are size-bounded; checkpoint reads always use `torch.load(..., weights_only=True)` and require an exact known toy architecture. Writes use directory-descriptor traversal, `O_NOFOLLOW`, restrictive `0600` file permissions, exclusive temporary creation, fsync and atomic hard-link/replace semantics. Overwriting existing outputs is denied by default. The trusted root and its ancestors must remain controlled; this is not a general-purpose sandbox.

Optional external model download scripts remain **opt-in**. Reproducing an externally sourced Qwen artifact requires independently pinned hashes, verified publishers, safe parsers, appropriate model licenses and isolated compute. The toy Ed25519 policy does **not** authenticate downloaded models or protect model-serving systems.

## Key handling

- Do **not** generate or commit private keys in this repository. `.gitignore` excludes common PEM/key suffixes as a secondary mistake guard.
- Use separate keys for human approval versus CI if expanded beyond the toy demonstration, and establish issuance/revocation procedures outside this code.
- Do not send private keys in pull requests, bug reports, build logs, or model inputs.
- Key compromise or a writable trust store defeats signature-based approval; rotate keys and independently investigate affected artifacts.
- Because this is not a controlled environment, do not treat its tests as a production penetration test or formal security proof.
