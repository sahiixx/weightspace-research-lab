# Weight-Space Research Lab

A **complete, runnable, CPU-friendly, offline educational experiment** in localized neural-weight editing with PyTorch, bounded coordinate search, validation gates, immutable base checkpoints, independent holdout scoring, exact-diff verification, CLI, tests, Docker, and JSON/Markdown reports.

**Scope in one sentence:** the bundled experiment repairs one intentionally corrupted scalar in a tiny classifier *without gradients*, but **does not claim to reproduce the 2026 GPT-6 Astra/Qwen weight-editing breakthrough**. The published Qwen experiment is accessible through optional resource/reproduction tooling described below.

## Quick start

Requirements: Python >=3.10, pip, a CPU. All example data are generated locally.

```bash
python3 -m venv .venv
source .venv/bin/activate                # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
python -m wslab pipeline --out runs/demo
python -m pytest -q
python -m wslab verify --base runs/demo/base.pt --patched runs/demo/patched.pt --patch runs/demo/patch.json
```

After running, inspect:

- `runs/demo/base.pt`: intentionally corrupted, immutable initial weights (tensor-only PyTorch state_dict)
- `runs/demo/patch.json`: bounded patch proposal with original fingerprint and old/new values
- `runs/demo/patched.pt`: approved modified checkpoint
- `runs/demo/report.json`: per-split scores and subgroup metrics, complete candidate-selection evidence
- `runs/demo/report.md`: readable experiment report

## Run everything locally, including robustness and negative controls

```bash
# One command, after the installation in Quick start
bash scripts/run_all.sh runs/full-audit
```

This runs all tests, compiles the source, produces a separate baseline/edited report,
verifies the exact patch, repeats the experiment across 12 independent deterministic
seeds (validation/test/**audit** splits), and deliberately submits 10 bad or
regressive patch proposals. See `runs/full-audit/sweep.md` and
`runs/full-audit/negative_controls.md`.

For a customized sweep:

```bash
python -m wslab sweep --out runs/audit --seed 123 --seeds 25 --count 1024
python -m wslab negative-controls --out runs/audit
```

**Interpretation:** each seed has its own validation samples and independently
selected scalar value; test/audit splits are scored only afterward. Some smaller
validation sets can select near-optimal, rather than mathematically exact, weights.
Cross-seed averages are a repeatability check on *this toy rule*, not confidence
intervals for general weight-space editing. These test cases do not prove a secure
model-serving system.

## One step at a time

```bash
python -m wslab init --out runs/manual/base.pt
python -m wslab inspect --checkpoint runs/manual/base.pt
python -m wslab propose --checkpoint runs/manual/base.pt --out runs/manual/patch.json
python -m wslab apply --checkpoint runs/manual/base.pt --patch runs/manual/patch.json --out runs/manual/patched.pt
python -m wslab benchmark --checkpoint runs/manual/base.pt --split test
python -m wslab benchmark --checkpoint runs/manual/patched.pt --split test
python -m wslab verify --base runs/manual/base.pt --patched runs/manual/patched.pt --patch runs/manual/patch.json
```

Keep the default `--seed` and `--count` the same between `propose` and `apply`, since the acceptance gate must use the same declared validation protocol. Never select a candidate on the test split. The tool does not make API calls, train anything, or require weights downloads.

## Publish to GitHub

The project includes a clean Git history export, a safe `gh`-based publication script, and a GitHub Actions audit workflow. See [docs/GITHUB_PUBLISH.md](docs/GITHUB_PUBLISH.md). No publication is implied until the repo exists and the push succeeds.

## Docker

```bash
docker compose build
docker compose run --rm lab
docker compose run --rm --entrypoint python lab -m pytest -q
```

The container uses a PyTorch **CPU wheel**, runs without networking at runtime, writes results to a bind-mounted `./runs/`, uses a read-only root filesystem and a temporary `/tmp`. Network connectivity is required to *build* the image and install dependencies. `docker compose` assumes a recent Docker installation. This is an experiment runner, not a production service.

## CLI commands

| Command | Purpose |
|---|---|
| `pipeline` | Generate base checkpoint, search patch, safety-check, score, and write reports |
| `init` | Save deliberately corrupted starting checkpoint |
| `propose` | Evaluate a fixed bounded set of candidate edits using *validation only* |
| `apply` | Check patch schema, bounds, fingerprint and validation improvement, then save new checkpoint |
| `benchmark` | Score an existing checkpoint on validation, test or audit split |
| `inspect` | Show tensor shapes and a canonical SHA-256 fingerprint |
| `verify` | Rebuild expected edit from base and compare *every* tensor against the supplied patch checkpoint |
| `sweep` | Repeat validation-only patch selection across multiple seeds, then score independent test/audit splits |
| `negative-controls` | Deliberately attempt unsafe edits and write rejection evidence |

Use `python -m wslab COMMAND --help` for flags.

## Experiment: what actually happens

The classifier encodes the synthetic rule `y = (2*x0 + x1 > 0)` with two linear output logits. We deliberately change `linear.weight[1,0]` from `+2.0` to `-2.0`. Our deliberately transparent search evaluates 32 alternative scalar values in `[-4,4]` on deterministic synthetic validation examples; it picks the best legal candidate with no optimizer, gradients or backward pass. It checks the improvement and subgroup accuracy before saving. A separate seeded test split is **not used for patch selection**. This is a minimal demonstration of editable weight space and evaluator/safety plumbing; it is *not* an experiment in weight comprehension by a frontier LLM.

### What the safety system rejects

Stale base fingerprint; unknown tensor; invalid or repeated index; wrong old value; no-op patch; nonfinite values; overly large per-element changes; too many edits; validation accuracy not improving sufficiently; predefined subgroup regression. All proposed tensor changes occur in a deep copy until the proposal clears validation. `verify` checks the complete output state against the exact approved edit; it rejects unlisted modifications.

These controls are **not** cryptographic signing, sandboxed model execution, model security certification, fairness guarantees, or a production multi-tenant authorization system. They demonstrate useful safeguards in a limited, trusted local workflow.

## Optional: verify the real published Qwen checkpoint

The official `RikiyaT/Qwen3.5-0.8B-weight-space-programming` model is ~1.8 GB and running the official full benchmark requires significant accelerator compute. Neither model nor benchmark datasets are included in this archive.

```bash
# Inspect the full primary logs and official code without downloading model weights
bash scripts/get_official_resources.sh

# Optional: install extra dependencies
python -m pip install -e '.[official]'

# Obtain the published official edited model (downloads ~1.8GB)
hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming \
  --local-dir external/qwen-edited

# Compare the two edited tensor byte-hashes with those on the official model card
python scripts/verify_official_qwen.py --patched-dir external/qwen-edited
```

To also compare values to the unedited model, download `Qwen/Qwen3.5-0.8B` at the exact base revision documented in `docs/REPRODUCTION.md`, then pass `--base-dir external/qwen-base`. `verify_official_qwen.py` checks the two named tensors; it does **not** prove that no other tensor differs. For an authoritative exact reconstruction, use the official `exp8_math/code/apply_edit.py`, which verifies tensor hashes when applying the upstream patch.

To reconstruct the authors' 50-pass edit from the exact original BF16 base checkpoint,
use `bash scripts/rebuild_official.sh /path/to/original/model.safetensors /path/to/edited.safetensors`
after downloading the upstream logs. This wraps the upstream `apply_edit.py` entrypoint
and does not implement its edit logic. It cannot operate without the base weights.

The official `exp8_math/README.md` and `exp8_math/code/evaluate.py` define the paper's full scoring protocol. Do **not** confuse that GPU-heavy benchmark with the offline `wslab benchmark` command: the latter measures only the toy classifier.

## Scientific grounding

Read [docs/RESEARCH.md](docs/RESEARCH.md), [docs/REPRODUCTION.md](docs/REPRODUCTION.md) and [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md). The 2026 researchers released eight experiments and documented a Qwen3.5-0.8B edit that affected 64 columns of the MLP down-projection weights in each of layers 15 and 19 (131,072 touched scalar positions). Independent verification of the saved checkpoint and alternative inference-setting baselines are important context. Our small lab uses the **idea** of bounded direct edits, not the original experiment's architecture or its agentic exploration protocol.

## Structure

```text
weightspace-research-lab/
├── src/wslab/                   # CLI, model, deterministic data, patching, gating, scoring
├── tests/                       # Data, safety, regression, checkpoint, CLI integration tests
├── scripts/
│   ├── run_all.sh and rebuild_official.sh
│   ├── get_official_resources.sh
│   └── verify_official_qwen.py
├── docs/                        # Evidence, GitHub publication, reproduction, limitations and threat model
├── runs/                        # Generated checkpoints/reports (gitignored)
├── Dockerfile
├── compose.yaml
├── pyproject.toml
├── Makefile
└── .github/workflows/ci.yml
```

## Reproduction status of this edition

The complete local suite was executed on CPU. Docker build/run and downloading the
public ~1.8GB Qwen weights are provided as optional commands, but require a Docker
daemon or outbound network/accelerator resources that were not available in the
construction environment. See `VALIDATION.md`. No real Qwen benchmark result is
attributed to an experiment conducted by this lab.

## Provenance and licensing

This lab's original source code and documentation are MIT-licensed (`LICENSE`). It **does not bundle** the original authors' datasets, model weights or scripts. The official research logs/data/networks are CC BY 4.0, their original scripts are MIT, and MinAtar snippets in the original logs have GPL-3.0 terms. Respect each third-party model/dataset license when downloading external resources.

Research: https://rikiyat.github.io/blog/weight-space-programming/
Primary logs: https://github.com/RikiyaT/lm_understand-logs
Official edited checkpoint: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
