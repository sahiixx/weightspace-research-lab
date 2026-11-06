#!/usr/bin/env python3
"""Independent, CPU-only, bitwise reconstruction/verification of Qwen p50c sparse edit.

Does not execute upstream code, download weights, evaluate inference or sign results.
Requires locally obtained, pinned base/edited safetensors and upstream edit NPZ+manifest.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

EXPECTED_NAMES = {
    "model.language_model.layers.15.mlp.down_proj.weight": "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
    "model.language_model.layers.19.mlp.down_proj.weight": "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
}
EXPECTED_CHANGED = 130188
BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"
LOG_REV = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
DATA_REV = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json(path: Path, maximum_bytes: int = 1_000_000) -> dict:
    if not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ValueError(f"Invalid JSON source: {path}")
    def pairs(p):
        d = {}
        for k, v in p:
            if k in d:
                raise ValueError("Duplicate JSON object key")
            d[k] = v
        return d
    result = json.loads(path.read_text("utf-8"), object_pairs_hook=pairs,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if type(result) is not dict:
        raise ValueError("Top-level JSON must be an object")
    return result


def _opens(directory: Path, stack: ExitStack):
    from safetensors import safe_open
    if not directory.is_dir():
        raise ValueError(f"Checkpoint directory missing: {directory}")
    shards = sorted(directory.glob("*.safetensors"))
    if not shards or len(shards) > 50:
        raise ValueError("Expected a safetensors checkpoint with <=50 shards")
    names = {}
    for shard in shards:
        h = stack.enter_context(safe_open(str(shard), framework="pt", device="cpu"))
        for name in h.keys():
            if name in names:
                raise ValueError(f"Duplicate tensor name across shards: {name}")
            names[name] = h
    return names


def raw_bf16(t):
    import torch
    if t.dtype is not torch.bfloat16 or t.device.type != "cpu":
        raise ValueError("Expected original raw CPU bfloat16 tensor")
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def reconstruct(base, patch, idx: int):
    import numpy as np
    import torch
    if base.dtype != torch.bfloat16:
        raise ValueError("Base tensor must be bfloat16")
    if patch["form"] != "sparse":
        raise ValueError("Only the independently audited sparse p50c format is supported")
    if list(base.shape) != patch["shape"] or patch["dtype"] != "bfloat16":
        raise ValueError("Patch tensor shape/dtype mismatch")
    if sha256(raw_bf16(base)) != patch["base_sha256"]:
        raise ValueError("Base tensor SHA-256 differs from released patch manifest")
    # This reconstruction uses BF16 bit patterns rather than FP32 arithmetic.
    # The patch records the exact replacement bit patterns at sparse flat indices.
    indices = np.asarray(patch["values"][f"t{idx}_idx"])
    values = np.asarray(patch["values"][f"t{idx}_val"])
    if indices.ndim != 1 or values.ndim != 1 or len(indices) != len(values):
        raise ValueError("Sparse indices and values must be equally sized 1-D arrays")
    if indices.dtype.kind not in "ui" or values.dtype != np.uint16:
        raise ValueError("Indices must be integers; bit-pattern values must be uint16")
    if len(indices) > 1_000_000 or np.any(indices >= base.numel()):
        raise ValueError("Sparse edit out of bounds")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Sparse edit contains duplicate flat indices")
    bits = base.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1).copy()
    before = bits.copy()
    bits[indices] = values
    count = int(np.count_nonzero(before != bits))
    rebuilt = bits.view(np.int16).reshape(tuple(base.shape))
    rebuilt_raw = rebuilt.tobytes(order="C")
    if sha256(rebuilt_raw) != patch["edited_sha256"]:
        raise ValueError("Independently reconstructed bytes do not match published edited hash")
    if count != patch["changed_entries"]:
        raise ValueError("Changed entries differ from patch manifest")
    return rebuilt_raw, count


def verify(base_dir: Path, edited_dir: Path, manifest_path: Path, npz_path: Path, *,
           expected: dict[str, str] | None = None, expected_count: int | None = EXPECTED_CHANGED) -> dict:
    import numpy as np
    expected = EXPECTED_NAMES if expected is None else expected
    manifest = _strict_json(manifest_path)
    records = manifest.get("tensors")
    if type(records) is not list or len(records) != len(expected):
        raise ValueError("Unexpected patch manifest tensor count")
    if npz_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Patch NPZ exceeds size limit")
    actual = {}
    count_total = 0
    with np.load(npz_path, allow_pickle=False) as values, ExitStack() as stack:
        base = _opens(base_dir, stack)
        edited = _opens(edited_dir, stack)
        if set(base) != set(edited):
            raise ValueError("Base and edited checkpoints have different tensor inventories")
        if len(base) < len(expected):
            raise ValueError("Model contains fewer tensors than patch manifest")
        changed = []
        for name in sorted(base):
            b = base[name].get_tensor(name)
            e = edited[name].get_tensor(name)
            if b.dtype != e.dtype or tuple(b.shape) != tuple(e.shape):
                raise ValueError(f"Shape/dtype drift in {name}")
            if name in expected:
                if b.dtype.__str__() != "torch.bfloat16":
                    raise ValueError("Edited tensor must be bfloat16")
                actual[name] = {"base_sha256": sha256(raw_bf16(b)),
                                "edited_sha256": sha256(raw_bf16(e)),
                                "shape": list(b.shape)}
            if not bool((b == e).all()):
                changed.append(name)
        if set(changed) != set(expected):
            raise ValueError(f"Unexpected changed-tensor set: {changed}")
        for i, record in enumerate(records):
            if type(record) is not dict or record.get("name") not in expected:
                raise ValueError("Unexpected tensor in upstream patch manifest")
            name = record["name"]
            if record["edited_sha256"] != expected[name]:
                raise ValueError("Expected edited hash differs from published reference")
            if actual[name]["base_sha256"] != record["base_sha256"]:
                raise ValueError("Base weights do not match the released patch inputs")
            if actual[name]["edited_sha256"] != record["edited_sha256"]:
                raise ValueError("Edited checkpoint bytes do not match released values")
            patch_info = dict(record, values=values)
            rebuilt, n = reconstruct(base[name].get_tensor(name), patch_info, i)
            if rebuilt != raw_bf16(edited[name].get_tensor(name)):
                raise ValueError("Independent reconstruction differs from edited release")
            actual[name]["changed_entries"] = n
            count_total += n
    if expected_count is not None and count_total != expected_count:
        raise ValueError("Total changed BF16 entries differ from released manifest")
    return {
        "kind": "real_checkpoint_bitwise_reconstruction_no_inference",
        "schema_version": 1,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "base_repo": "Qwen/Qwen3.5-0.8B", "base_revision": BASE_REV,
        "edited_repo": "RikiyaT/Qwen3.5-0.8B-weight-space-programming",
        "logs_commit": LOG_REV, "dataset_revision": DATA_REV,
        "tensor_count": len(base), "changed_tensor_names": sorted(changed),
        "changed_bf16_entries": count_total, "tensors": actual,
        "upstream_patch_manifest_sha256": sha256(manifest_path.read_bytes()),
        "upstream_patch_npz_sha256": sha256(npz_path.read_bytes()),
        "not_verified": ["GPU inference", "MATH-500 performance", "author's editing agent trajectory", "checkpoint publisher identity beyond pinned source"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("base-dir", "edited-dir", "patch-manifest", "patch-npz", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    try:
        output = verify(args.base_dir, args.edited_dir, args.patch_manifest, args.patch_npz)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(output, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        print(json.dumps({"status": "verified", "changed_elements": output["changed_bf16_entries"],
                          "tensor_count": output["tensor_count"], "report": str(args.out)}))
        return 0
    except (OSError, ValueError, KeyError, ImportError) as e:
        print("Verification failed:", e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
