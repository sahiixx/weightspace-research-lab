# Weight-Space Research Lab v0.4: hardening validation

**Date:** 2026-11-06  
**Environment:** Python 3.13, PyTorch 2.10 CPU, cryptography 46, Linux  
**Claim scope:** educational toy weight-editing and local signer-policy workflow, **not** production model security certification.

## Completed local checks

- Existing compatibility tests: **31/31 passed**, run with `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q` over the six original test files.
- Newly added trust-boundary tests: **21/21 passed** (`pytest -q tests/test_trust_boundaries.py`).
- Total: **52 tests passed** in separately executed batches.
- Offline full pipeline: generated `runs/hardened-demo/`, checkpoint, unsigned toy patch, dataset manifest, evidence hash-chain and report (256 samples/split). Local test accuracy baseline **60.55%**, patched **100.00%**; these are purely toy data.
- Negative controls: **10/10 rejected**; base checkpoint unmodified.
- Authentication tests: signed roundtrip, post-signature tampering rejection, signer revocation, source allowlist, dataset mismatch, unsigned CLI rejection, float32 representability, path traversal, symlinks, no-clobber, duplicate JSON fields, and signed audit evidence tampering.

## Not verified / limitations

- Live GitHub Actions v0.4 CI is unverified until the branch runs on GitHub.
- Docker runtime was not executed locally; config is provided for external validation.
- Live Qwen checkpoint reconstruction, model downloads and GPU benchmarks were **not** executed.
- Signed assertions are only as trustworthy as the protected signer key, source code, and trust-store policy. Signed provenance is not independent source-commit verification or human-identity proof. Evidence chaining is not immutable without independent protected storage.
