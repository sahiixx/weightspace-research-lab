# Codex operating instructions: Weight-Space Research Lab

Agent tools in `codex/tools/` are **proposal-and-inspection only**. They may not sign or approve patches, load private keys, modify protected trust stores, deploy changes, or call `wslab apply`. The **human operator must approve, sign and apply in a separate trusted terminal** after reviewing the exact candidate digest and validation evidence. A general Codex execution approval does not constitute a cryptographic authorization.

Workflow:
1. On a disposable checkpoint, run `python codex/tools/propose.py --checkpoint runs/demo/base.pt --workspace . --proposal-name candidate.json`.
2. Run `python codex/tools/inspect_proposal.py --checkpoint runs/demo/base.pt --patch runs/agent-proposals/candidate.json`.
3. Present the unsigned patch, evidence, dataset seed/count and precise `base_sha256` to the operator. **Stop**. Do not generate signing keys or invoke `wslab sign`/`wslab apply` autonomously.
4. Only the independent approver executes the commands in `docs/CODEX_APPROVAL.md` using an out-of-workspace key/trust store. Approver must re-check the candidate and commit, and independently validate the expected dataset hash.

Do not fetch model checkpoints or execute upstream research scripts without explicit scope and review. Never interpret a successful toy score as evidence of Qwen or general-model safety. Commit no private keys, credentials, checkpoint binary files or generated evaluation dumps.
