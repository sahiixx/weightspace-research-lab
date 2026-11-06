# Codex agent tool boundary and human approval procedure

## Agent side (unprivileged workspace)

```bash
python -m wslab pipeline --out runs/demo
python codex/tools/propose.py --checkpoint runs/demo/base.pt --workspace . --proposal-name candidate.json
python codex/tools/inspect_proposal.py --checkpoint runs/demo/base.pt --patch runs/agent-proposals/candidate.json
```

The agent produces unsigned data and displays an evaluation; it **never** receives the operator's signing key or trusted policy. The tools have no approval or deploy subcommand. A passed validation gate is **not an authorization**.

## Human side (separate trusted environment)

The operator audits the exact proposal, source repository and revision, validation split, and output path. For the existing toy experiment the human may execute the existing `wslab sign` and `wslab apply` commands from `README.md` with a **protected private Ed25519 key and independently provisioned trust policy**. Do not mount these into the Codex workspace or grant the agent broader filesystem access. `wslab apply` checks signer permission before evaluating or writing the candidate.

Never copy the approver's keys into Codex prompts, repository files, GitHub Actions secrets accessible to PRs, or agent tools. Keep approvals and deployment as distinct identities and environments. Codex sandbox/approval mechanisms do not prove operator identity, key custody, or audit immutability. An authorized signature establishes who controls the trusted signing key, **not** that the proposed change is safe.
