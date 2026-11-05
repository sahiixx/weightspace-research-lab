# Local validation summary

Date: **2026-11-05**. See [docs/EXECUTION_REPORT.md](docs/EXECUTION_REPORT.md)
for commands, environment and explicitly unexecuted reproduction work.

- **31/31 automated tests passed** (including offline GitHub delivery checks).
- Source compiled (`compileall`), and all offline CLI workflows completed.
- Synthetic full pipeline: 65.48% holdout baseline -> 100.00% repaired.
- Exact checkpoint verification: **one edited scalar**, unchanged initial tensor state.
- 12 independent deterministic seed runs: mean test accuracy 64.71% -> 100.00%.
- **10/10** deliberately unsafe or regressive patch proposals rejected.
- Docker execution: **not tested**, CLI unavailable.
- Official Qwen checkpoint download/hash or GPU benchmarking: **not tested**, no outbound DNS or CUDA.

These results are **toy-model results** and do not reproduce the published Qwen benchmark.

## GitHub-ready packaging (v0.3.0)

The project contains `scripts/publish_to_github.sh`, a GitHub Actions workflow with read-only permissions, and `docs/GITHUB_PUBLISH.md`. Source packaging and local Git history do not imply that the repository was created, pushed, or that GitHub Actions completed. Check the actual destination repository and its workflow runs after publishing.
