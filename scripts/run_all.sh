#!/usr/bin/env bash
# End-to-end offline verification. Invoke from repository root after pip install -e '.[test]'.
set -euo pipefail
# Avoid high thread-launch overhead on the deliberately tiny neural net.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OUT="${1:-runs/full-audit}"
python -m pytest -q
python -m compileall -q src tests scripts
python -m wslab pipeline --out "$OUT/pipeline" --seed 42 --count 2048
python -m wslab verify --base "$OUT/pipeline/base.pt" --patched "$OUT/pipeline/patched.pt" --patch "$OUT/pipeline/patch.json"
python -m wslab sweep --out "$OUT" --seed 42 --seeds 12 --count 1024
python -m wslab negative-controls --out "$OUT"
printf '\nAll local tests and demos completed. View %s/\n' "$OUT"
