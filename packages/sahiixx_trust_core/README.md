# SAHIIXX Trust Core (experimental SDK)

A standalone package requiring **only `cryptography`**, not PyTorch. The API provides canonical JSON hashing, domain-separated Ed25519 approval envelopes, allowlisted policy verification, and local hash-chained events. Import `sahiixx_trust_core` after `pip install -e packages/sahiixx_trust_core`.

Use cases include signed **configuration changes**, **model patch proposals**, and **document/workflow approval records**. An operator must control signing keys and a trusted policy **outside** any agent-accessible workspace. `sign_envelope` is deliberately not a Codex agent tool.

The `artifact_sha256` binds canonical JSON **payload**, not a raw model/checkpoint binary. For binary artifacts, include an independently verified file digest and identifying metadata inside the signed payload; re-check the binary at deployment. Do not confuse a signed metadata claim with a verified Git commit or human identity.

The event chain is editable/recomputable by a local attacker and is not an immutable transparency log. No regulated-industry compliance is claimed. Separate key custody, deployment-time verification, independent build attestations, immutable audit storage, revocation process and threat assessment are prerequisites for production.
