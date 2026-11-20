# PR #1 reconciliation and GPU readiness

Reviewed on 2026-11-20 against these exact commits:

- Main: `f2005696da9d4c084153b99694d64500e55c9dac`
- PR #1: `1642fb706f3f0304c8a07cd2d6cb94bdc7ada4b5`
- Common ancestor: `4cf2c835a136541646cad642b56601fc8f3f9eff`

## Standards

No hard documented-standard violations were found in the targeted comparison.
An optional maintainability observation is that `src/wslab/evidence.py`
duplicates its Merkle-root reduction in manifest creation and verification.
Both copies already exist unchanged on main; this is not missing PR work.

## Spec

No PR #1 implementation needs recovery. Of the 21 files changed by PR #1,
17 are byte-identical on main, including every changed `src/wslab` module,
the trust-boundary tests, the pinned-resource script, and the security and
threat-model documents. The other four files are `README.md`, `VALIDATION.md`,
`SHA256SUMS.txt`, and `pyproject.toml`; main adds subsequent release content,
checksum entries, the newer version, and a test dependency.

The authenticated patch, strict schema, bounded IO, dataset-digest, and audit
mechanisms are therefore already present through merged [PR #2](https://github.com/sahiixx/weightspace-research-lab/pull/2).
[PR #1](https://github.com/sahiixx/weightspace-research-lab/pull/1) is superseded
and can be closed without merging its older branch or deleting that branch.
Protected key custody and immutable auditing remain external assumptions;
this comparison does not establish production security.

Reproduce the source comparison:

```bash
git diff --name-only 4cf2c835a136541646cad642b56601fc8f3f9eff 1642fb706f3f0304c8a07cd2d6cb94bdc7ada4b5
git diff 1642fb706f3f0304c8a07cd2d6cb94bdc7ada4b5 f2005696da9d4c084153b99694d64500e55c9dac -- src tests/test_trust_boundaries.py scripts/get_official_resources.sh .gitignore docs/SECURITY_AND_TRUST.md docs/THREAT_MODEL.md
```

The second command produces no differences.

## GPU readiness

The existing manual GPU workflow imported PyTorch immediately after
`actions/setup-python`, before installing vLLM and its PyTorch dependency.
A fresh Python environment can therefore fail with `ModuleNotFoundError`
before dependency installation. The fix keeps the NVIDIA driver check first
and moves the PyTorch CUDA check after dependency installation, before any
model downloads or inference.

On 2026-11-20, the repository runner API returned `total_count: 0`.
The local machine reports AMD Radeon graphics and has no `nvidia-smi`
command. No GPU workflow execution was attempted during this review.

Execution requires an independently provisioned Linux x64 CUDA runner with
the `qwen-gpu` label. After the workflow fix is reviewed and merged, dispatch
`qwen-gpu-evaluation.yml` with `mode=smoke` and `protocol=think35` first.
Inspect both reports, their matching source pins and sample configurations,
and the paired comparison before scheduling `mode=full` with the same
protocol and an agreed compute budget. Preserve the generation artifacts;
the workflow retains them for seven days. See
[the execution protocol](QWEN_REAL_VERIFICATION.md) for the full benchmark
configuration and independent weight-verification procedure.

The prior [CPU inference run](https://github.com/sahiixx/weightspace-research-lab/actions/runs/42208277814)
and [main CI run](https://github.com/sahiixx/weightspace-research-lab/actions/runs/42209012271)
passed. They do not establish MATH-500 accuracy or GPU readiness.
Signing remains a separate operator-controlled step under `AGENTS.md`.

## Validation of this maintenance change

- 22 focused CPU tests passed across real-Qwen tooling, CPU smoke reports,
  reproduction tools, and the Codex workflow boundary.
- YAML checks confirmed manual-only dispatch, read-only repository
  permissions, the required runner labels, and dependency installation
  before the PyTorch CUDA assertion and model download.
- Bash syntax checks passed for every GPU workflow command block.
- The source checksum manifest includes this report and the workflow fix.

These checks do not execute vLLM or test CUDA runtime compatibility.
