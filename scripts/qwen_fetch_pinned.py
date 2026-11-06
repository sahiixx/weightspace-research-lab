#!/usr/bin/env python3
"""Opt-in download of pinned public Qwen model snapshots and MATH-500 test.jsonl.

No model code is executed. No private credentials are required for public sources.
A changed edited-model main revision is rejected until it is reviewed/re-pinned.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

BASE_ID = "Qwen/Qwen3.5-0.8B"
BASE_SHA = "2fc06364715b967f1860aea9cf38778875588b17"
EDITED_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
EDITED_SHA_PREFIX = "d1671a17"  # Observed and independently reported; resolve complete SHA before downloading.
DATA_ID = "HuggingFaceH4/MATH-500"
DATA_SHA = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def resolve_pins(api) -> dict:
    edited_sha = api.model_info(EDITED_ID, revision="main").sha
    if not isinstance(edited_sha, str) or len(edited_sha) != 40 or not edited_sha.startswith(EDITED_SHA_PREFIX):
        raise ValueError("Edited model revision changed; review new publisher commit and explicitly repin")
    return {"base": {"repo": BASE_ID, "revision": BASE_SHA},
            "edited": {"repo": EDITED_ID, "revision": edited_sha},
            "dataset": {"repo": DATA_ID, "revision": DATA_SHA}}


def download(root: Path, *, no_models=False) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    pins = resolve_pins(HfApi())
    if root.exists() and any(root.iterdir()):
        raise ValueError("Destination must be empty; never overwrite existing downloads")
    root.mkdir(parents=True, exist_ok=True)
    if not no_models:
        for kind in ("base", "edited"):
            snapshot_download(repo_id=pins[kind]["repo"], revision=pins[kind]["revision"],
                              local_dir=root / kind,
                              allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json", "tokenizer*", "merges.txt", "vocab.json", "chat_template.jinja", "generation_config.json"])
    snapshot_download(repo_id=DATA_ID, repo_type="dataset", revision=DATA_SHA,
                      local_dir=root / "dataset", allow_patterns=["test.jsonl"])
    path = root / "dataset" / "test.jsonl"
    if not path.is_file():
        raise ValueError("Pinned dataset snapshot lacks test.jsonl")
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    if len(rows) != 500 or not all("problem" in row and "answer" in row for row in rows):
        raise ValueError("Unexpected MATH-500 split contents")
    pins["dataset"]["test_jsonl_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "source-pins.json").write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--no-models", action="store_true", help="Only fetch pinned MATH-500")
    args = parser.parse_args()
    try:
        print(json.dumps(download(args.root, no_models=args.no_models), sort_keys=True))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print("Pinned download failed:", exc, file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
