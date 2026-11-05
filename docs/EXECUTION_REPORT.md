# Executed end-to-end verification report

**Date:** 2026-11-05  
**Package edition:** 0.2.0  
**Scope:** CPU-only, synthetic *toy-linear-repair* research lab; no real Qwen inference.

## Completed commands

```bash
python -m pytest -q
python -m compileall -q src tests scripts
python -m wslab pipeline --out runs/full-audit/pipeline --seed 42 --count 2048
python -m wslab verify --base runs/full-audit/pipeline/base.pt \
  --patched runs/full-audit/pipeline/patched.pt \
  --patch runs/full-audit/pipeline/patch.json
python -m wslab sweep --out runs/full-audit --seed 42 --seeds 12 --count 1024
python -m wslab negative-controls --out runs/full-audit
```

All completed successfully in the build environment. The shell script
`scripts/run_all.sh` executes this same sequence, with CPU thread limits.

## Actual local metrics

| Measure | Outcome |
|---|---:|
| Automated tests | 27 passed |
| Full workflow exact tensor diff | 1 changed scalar |
| Full workflow holdout accuracy before | 65.48% |
| Full workflow holdout accuracy after | 100.00% |
| Independent seed runs | 12 |
| Mean cross-seed validation before/after | 65.34% / 100.00% |
| Mean cross-seed test before/after | 64.71% / 100.00% |
| Mean cross-seed audit before/after | 64.48% / 100.00% |
| Unsafe/regressive patch cases rejected | 10 of 10 |
| Base checkpoint modified during safety checks | No |

The rule and the target weight are known in advance; no gradients, dataset
training, LLM-agent actions, or causal circuit discovery are involved.
These benchmark numbers are not transferable to Qwen or other LLMs.

## Verified runtime

- Python `3.13.5`
- PyTorch `2.10.0+cpu`
- Linux x86_64, no CUDA device
- Source compilation and local CLI checks passed

## Not executed and why

| Capability | Status | Why |
|---|---|---|
| Docker image build and container execution | Not run | Docker CLI/daemon not available |
| Downloading official logs or Qwen weights | Not run | This runtime cannot resolve external download hosts |
| Published edited-tensor SHA comparison | Not run | Edited model's ~1.8 GB checkpoint not available locally |
| Reconstructing official Qwen model | Not run | Exact base safetensors and upstream logs absent |
| GPU MATH-500/GSM8K/IFEval evaluation | Not run | No CUDA device, original evaluation is very compute-intensive |
| Replicating the researchers' Astra editor | Not run | Requires their frontier-model/harness environment and time budget |

There are commands, source references and integrity helpers in
`docs/REPRODUCTION.md` for running the optional model-scale work on a separately
provisioned machine. No claims of completion are made for those tasks.

## Integrity of generated evidence

- Machine-readable complete pipeline report: `runs/full-audit/pipeline/report.json`
- Human-readable pipeline report: `runs/full-audit/pipeline/report.md`
- Multi-seed raw metrics: `runs/full-audit/sweep.json`
- Per-seed summary: `runs/full-audit/sweep.md`
- Rejected proposal details: `runs/full-audit/negative_controls.json`
- Patch verification: run `python -m wslab verify ...` with the generated checkpoints
- File digests: `SHA256SUMS.txt` (file hashes, **not cryptographic signatures**)

## Grounding in external public research

- Original research: https://rikiyat.github.io/blog/weight-space-programming/
- Original experiment logs and harness: https://github.com/RikiyaT/lm_understand-logs
- Experiment 8 details: https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md
- Release and published BF16 tensor SHA256s: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming

The public Qwen metrics are researchers' reports and independent external
re-evaluations, not this lab's results. See `docs/RESEARCH.md` for the caveats.
