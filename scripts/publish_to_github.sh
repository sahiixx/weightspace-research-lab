#!/usr/bin/env bash
# Create an EMPTY GitHub repository (private by default) and push the current local Git history.
# Usage: bash scripts/publish_to_github.sh sahiixx/weightspace-research-lab [--public|--private]
set -euo pipefail
REPO="${1:-}"
VISIBILITY="${2:---private}"
if [[ ! "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "Usage: bash scripts/publish_to_github.sh OWNER/REPOSITORY [--private|--public]" >&2
  exit 2
fi
if [[ "$VISIBILITY" != "--private" && "$VISIBILITY" != "--public" ]]; then
  echo "Visibility must be --private or --public" >&2
  exit 2
fi
for command in gh git; do
  command -v "$command" >/dev/null 2>&1 || { echo "Missing command: $command" >&2; exit 2; }
done
gh auth status >/dev/null
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -d .git ]]; then
  echo "Initialize and commit the project before publishing: git init -b main && git add . && git commit -m 'Initial research lab'" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree has uncommitted edits; commit or stash first" >&2
  exit 2
fi
if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "Switch to the main branch before publication" >&2
  exit 2
fi
# Never overwrite a non-empty target repository or guess which repository to use.
if gh repo view "$REPO" --json nameWithOwner >/dev/null 2>&1; then
  # A failed branch read is not evidence that the remote is empty.
  if ! EXISTING="$(gh api "repos/$REPO/branches" --jq '.[].name')"; then
    echo "Unable to confirm that the target repository is empty; refusing to push" >&2
    exit 3
  fi
  if [[ -n "$EXISTING" ]]; then
    echo "Target repository already has commits. Refusing to overwrite or merge unrelated work." >&2
    exit 3
  fi
else
  gh repo create "$REPO" "$VISIBILITY" --description "Weight-space programming reproducibility and safety research laboratory"
fi
if git remote get-url origin >/dev/null 2>&1; then
  OLD_REMOTE="$(git remote get-url origin)"
  if [[ "$OLD_REMOTE" != "https://github.com/$REPO.git" && "$OLD_REMOTE" != "git@github.com:$REPO.git" ]]; then
    echo "origin points to another repository ($OLD_REMOTE); refusing to change it" >&2
    exit 3
  fi
else
  git remote add origin "https://github.com/$REPO.git"
fi
git push -u origin main
echo "Published to https://github.com/$REPO"
