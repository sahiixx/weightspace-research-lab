# Local validation record, v0.5

**Scope:** local CPU development and independent-signer primitives. All results are local and do not establish production security or a model-scale Qwen benchmark.

Commands executed:

```bash
PYTHONPATH=src:packages/sahiixx_trust_core/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
# 61 passed
python -m pip install --no-deps --no-build-isolation -e packages/sahiixx_trust_core
# import sahiixx_trust_core succeeded from /tmp
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m wslab pipeline --out runs/codex-demo --count 256
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python codex/tools/propose.py --checkpoint runs/codex-demo/base.pt --workspace . --proposal-name v0.5-candidate.json --count 256
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python codex/tools/inspect_proposal.py --checkpoint runs/codex-demo/base.pt --patch runs/agent-proposals/v0.5-candidate.json --count 256
python scripts/qwen_reproduction.py preflight --out runs/qwen-preflight.json
```

Agent proposal was `unsigned_proposal_only`, `authorized=false`; the inspector returned `validation_gate=passed`, `authorization=NOT_APPROVED`, validation accuracy 0.640625 -> 1.0, toy data only.

Qwen preflight: Python 3.13, PyTorch 2.10 CPU, CUDA `false`, `vllm` not installed, no official log checkout, no original-model weights. No GPU evaluation was done. Public upstream merged score files inspected separately at git revision `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`: raw `acc` values are fractions (unedited MATH-500 0.079 and edited p50c 0.481), corresponding to 7.9% and 48.1%. The new `published-logs` command explicitly converts raw fractions to percentages. **This is not an independent inference reproduction.**

Unknown / not done: actual GPU checkpoint reconstruction, standalone Docker execution, binary model provenance checks, Codex live session tool call, externally anchored audit retention, real operator key custody, external adversarial audit, enterprise compliance.

Release is intended for a GitHub review PR, not automatic merge.
