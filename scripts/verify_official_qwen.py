#!/usr/bin/env python3
"""Optional verification of the two *published* Qwen edited-tensor fingerprints.

Requires: pip install '.[official]' and a locally downloaded official model.
It does NOT reproduce the edit-generation search or the full benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

# SHA-256 values published in the official Hugging Face model card (Nov 2026).
EXPECTED = {
    "model.language_model.layers.15.mlp.down_proj.weight":
        "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
    "model.language_model.layers.19.mlp.down_proj.weight":
        "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
}


def tensor_hash(tensor) -> str:
    # Reinterpret raw bytes without converting BF16 to FP32 (which alters hashes).
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8)
                          .cpu().numpy().tobytes()).hexdigest()


def read_named_tensors(directory: Path, names: set[str]):
    from safetensors import safe_open
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    found = {}
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors shards in {directory}")
    for shard in files:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            matches = names.intersection(handle.keys())
            for name in matches:
                if name in found:
                    raise ValueError(f"Duplicate tensor {name} in shards")
                found[name] = handle.get_tensor(name)
    if names - found.keys():
        raise ValueError(f"Missing tensors: {sorted(names - found.keys())}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patched-dir", type=Path, required=True,
                        help="Directory containing official edited model safetensors")
    parser.add_argument("--base-dir", type=Path,
                        help="Optional original Qwen model directory for change counts")
    args = parser.parse_args()
    patched = read_named_tensors(args.patched_dir, set(EXPECTED))
    report = {"source": "Official published Hugging Face tensor hashes", "tensors": {}}
    all_match = True
    base = read_named_tensors(args.base_dir, set(EXPECTED)) if args.base_dir else None
    for name, expected_sha in EXPECTED.items():
        p = patched[name]
        sha = tensor_hash(p)
        match = (sha == expected_sha)
        all_match &= match
        entry = {"shape": list(p.shape), "dtype": str(p.dtype),
                 "sha256": sha, "expected_sha256": expected_sha, "matches": match}
        if base:
            b = base[name]
            if b.shape != p.shape or b.dtype != p.dtype:
                raise ValueError(f"Shape/dtype mismatch for {name}")
            entry["changed_elements_from_base"] = int((b != p).sum().item())
        report["tensors"][name] = entry
    report["all_published_hashes_match"] = all_match
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all_match else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
