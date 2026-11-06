#!/usr/bin/env bash
# Optional external research logs. Do not fetch arbitrary moving refs.
# User must obtain the 40-char commit SHA through an independently trusted channel.
set -euo pipefail
PIN="${OFFICIAL_LOGS_COMMIT:-}"
if [[ ! "$PIN" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Set OFFICIAL_LOGS_COMMIT to an independently verified 40-character Git commit SHA.' >&2
  exit 2
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEST="$ROOT/external/lm_understand-logs"
if [[ -e "$DEST" || -L "$DEST" ]]; then
  echo "Refusing to overwrite existing external resource: $DEST" >&2
  exit 2
fi
mkdir -p "$ROOT/external"
git clone --no-checkout --filter=blob:none \
  https://github.com/RikiyaT/lm_understand-logs.git "$DEST"
git -C "$DEST" fetch --depth 1 origin "$PIN"
git -C "$DEST" checkout --detach "$PIN"
if [[ "$(git -C "$DEST" rev-parse HEAD)" != "$PIN" ]]; then
  echo 'Downloaded research log commit mismatch' >&2
  exit 1
fi
printf 'Research logs pinned to %s. Review them before running any code.\n' "$PIN"
