# v0.5: what was actually built and what is not yet verified

## Qwen reproduction

- Upstream published experiment: `https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math`
- Baseline: `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`
- Published edited release: `RikiyaT/Qwen3.5-0.8B-weight-space-programming`
- Existing tools: `scripts/get_official_resources.sh`, `scripts/rebuild_official.sh`, and `scripts/verify_official_qwen.py`.
- **New** `scripts/qwen_reproduction.py preflight` is CPU-compatible, downloads nothing, and makes an environment report. `published-logs` requires a user-pinned upstream Git commit and checks existing published score summaries against the documented values, recording their SHA-256 digests.
- **Not done by the new scripts:** no actual original-model download, no weight-edit reconstruction, no full model/binary hash comparison, no 32k-token inference, and no independent MATH-500 reproduction. Preflight and published-log matching cannot establish successful Qwen reproduction.
- To reproduce inference, provision suitable CUDA GPU hardware, inspect the upstream model code and licenses, download exact pinned model and logs revisions, reconstruct the patched checkpoint, verify exact tensor digests, and benchmark with the upstream think35 protocol (including thinking mode, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, max 32768 tokens, four samples per MATH-500 item). Publish baseline/edited runs, prompts, generations, metrics and runtime metadata.

## Trust Core

`packages/sahiixx_trust_core` is a separately installable Python distribution, independent of torch. The signer policy is an independently provisioned JSON object, *not* a GitHub identity lookup. `artifact_sha256` covers canonical JSON payloads. For actual model/checkpoint files, the payload must include the binary's externally computed digest, and deployment must re-check the file before applying. Audit events are hash-chained but not immutable. SDK's envelopes are **not** binary compatible with the older wslab v0.4 approval format.

The SDK is an **experimental foundation**, not a production security boundary or UAE fintech/real-estate regulatory compliance certification. Key custody, PKI or KMS/HSM integration, tenant isolation, policy versioning, key rotation/revocation operations, transparency-log anchoring, offline deployments and external adversarial review are outstanding.

## Codex integration

`AGENTS.md`, `.agents/skills/weightspace-proposal/SKILL.md`, `codex/tools/propose.py` and `codex/tools/inspect_proposal.py` provide a reviewable, proposal-only interface. There is **no autonomous apply tool**, and no key or policy file is included. A Codex workspace can still execute arbitrary code if given too much filesystem access; isolate the agent process, keep the signing key outside its workspace, and require independent human approval.

## Release quality gates

The new tests verify tampering, unauthorized source, signer revocation, expiry, wrong approval purpose, hash chains, pinned upstream summaries, and agent tool fail-closed behavior. CI runs the CPU suite and preflight without external weights. Full GPU/model reproduction remains open.

Pinned upstream log commit retrieved from the GitHub repository's current `main` on 2026-11-06: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. This is a research snapshot, not a commitment that the repository will never change. Original `merged.json` scores are stored as fractions (for example 0.481), displayed as percentages in the authors' README.

```bash
OFFICIAL_LOGS_COMMIT=4d09ba6d6a7e891dd8063974dfe81b3d78e3494f bash scripts/get_official_resources.sh
python scripts/qwen_reproduction.py published-logs \
  --expected-logs-commit 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f \
  --out runs/published-logs.json
```
