#!/usr/bin/env bash
# Explicitly requested optional download of the public official log repository.
# The local toy demo never contacts the internet.
set -euo pipefail
mkdir -p external
if [[ -e external/lm_understand-logs ]]; then
  echo 'external/lm_understand-logs already exists; inspect it before updating.' >&2
  exit 1
fi
git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs.git external/lm_understand-logs
printf '\nNext: read external/lm_understand-logs/exp8_math/README.md\n'
printf 'For rebuilding the official Qwen edit, follow exp8_math/code/apply_edit.py\n'
printf 'The official 32k-token multi-sample benchmark requires substantial GPU resources.\n'