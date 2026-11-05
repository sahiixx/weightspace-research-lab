#!/usr/bin/env bash
# Optional upstream Qwen edit reconstruction. Requires downloaded base model and official logs.
# Usage: bash scripts/rebuild_official.sh /path/to/model.safetensors /path/to/output-edited.safetensors
set -euo pipefail
if [[ "$#" != 2 ]]; then
  echo "Usage: $0 ORIGINAL_BASE_SAFETENSORS OUTPUT_PATCHED_SAFETENSORS" >&2
  exit 2
fi
BASE="$1"
OUTPUT="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$ROOT/external/lm_understand-logs"
if [[ ! -f "$BASE" ]]; then
  echo "Original base safetensors not found: $BASE" >&2
  exit 2
fi
if [[ ! -f "$UPSTREAM/exp8_math/code/apply_edit.py" ]]; then
  echo "Official logs missing: run 'bash scripts/get_official_resources.sh' first" >&2
  exit 2
fi
if [[ "$OUTPUT" != *.safetensors ]]; then
  echo "Output must end in .safetensors for full checkpoint reconstruction" >&2
  exit 2
fi
if [[ "$(realpath -m "$BASE")" == "$(realpath -m "$OUTPUT")" ]]; then
  echo "Refusing to replace the original base checkpoint" >&2
  exit 2
fi
python "$UPSTREAM/exp8_math/code/apply_edit.py" \
  "$BASE" "$UPSTREAM/exp8_math/runs/p50c/edit" "$OUTPUT"
