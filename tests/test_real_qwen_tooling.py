"""Adversarial CPU fixtures for independent real-model reproduction machinery."""
from __future__ import annotations
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

BASE = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    path = BASE / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

V = load("qwen_weight_verify.py")
E = load("qwen_gpu_eval.py")
R = load("qwen_evidence.py")
F = load("qwen_fetch_pinned.py")


def fixture(tmp_path):
    base = tmp_path / "base"
    edited = tmp_path / "edited"
    base.mkdir(); edited.mkdir()
    names = ("layer.15", "layer.19")
    tensors = {
        names[0]: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        names[1]: torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16),
        "unchanged": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    revised = {k: v.clone() for k, v in tensors.items()}
    revised[names[0]][0, 0] = -1.0
    revised[names[1]][1, 1] = 12.0
    save_file(tensors, str(base / "model.safetensors"))
    save_file(revised, str(edited / "model.safetensors"))
    records = []
    changes = {}
    for i, name in enumerate(names):
        b, e = tensors[name], revised[name]
        oldbits = b.view(torch.int16).numpy().view(np.uint16).ravel()
        newbits = e.view(torch.int16).numpy().view(np.uint16).ravel()
        idx = np.where(oldbits != newbits)[0].astype(np.int64)
        changes[f"t{i}_idx"] = idx
        changes[f"t{i}_val"] = newbits[idx].astype(np.uint16)
        records.append({"name": name, "form": "sparse", "dtype": "bfloat16",
                        "shape": list(b.shape), "base_sha256": V.sha256(V.raw_bf16(b)),
                        "edited_sha256": V.sha256(V.raw_bf16(e)), "changed_entries": 1})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tensors": records}), encoding="utf-8")
    npz = tmp_path / "edit.npz"
    np.savez(npz, **changes)
    expected = {r["name"]: r["edited_sha256"] for r in records}
    return base, edited, manifest, npz, expected


def test_independent_reconstruction_three_tensors(tmp_path):
    args = fixture(tmp_path)
    report = V.verify(*args[:4], expected=args[4], expected_count=2)
    assert report["tensor_count"] == 3
    assert report["changed_bf16_entries"] == 2
    assert set(report["changed_tensor_names"]) == {"layer.15", "layer.19"}
    assert report["kind"] == "real_checkpoint_bitwise_reconstruction_no_inference"


def test_reject_unknown_tensor_edit(tmp_path):
    base, edited, manifest, npz, expected = fixture(tmp_path)
    from safetensors.torch import load_file
    content = load_file(str(edited / "model.safetensors"))
    content["unchanged"][0, 0] = torch.tensor(0, dtype=torch.bfloat16)
    save_file(content, str(edited / "model.safetensors"))
    with pytest.raises(ValueError, match="Unexpected changed-tensor set"):
        V.verify(base, edited, manifest, npz, expected=expected, expected_count=2)


def test_reject_manifest_base_forgery(tmp_path):
    base, edited, manifest, npz, expected = fixture(tmp_path)
    obj = json.loads(manifest.read_text())
    obj["tensors"][0]["base_sha256"] = "0" * 64
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="Base weights"):
        V.verify(base, edited, manifest, npz, expected=expected, expected_count=2)


def test_reject_duplicate_sparse_indices(tmp_path):
    base, edited, manifest, npz, expected = fixture(tmp_path)
    with np.load(npz) as f:
        items = {x: f[x] for x in f.files}
    items["t0_idx"] = np.array([0, 0], dtype=np.int64)
    items["t0_val"] = np.array([0, 0], dtype=np.uint16)
    np.savez(npz, **items)
    with pytest.raises(ValueError, match="duplicate flat indices"):
        V.verify(base, edited, manifest, npz, expected=expected, expected_count=2)


def test_reject_forged_expected_sha(tmp_path):
    base, edited, manifest, npz, expected = fixture(tmp_path)
    expected["layer.15"] = "f" * 64
    with pytest.raises(ValueError, match="published reference"):
        V.verify(base, edited, manifest, npz, expected=expected, expected_count=2)


def test_nested_final_box_extractor():
    assert E.last_boxed("first \\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert E.last_boxed("no answer") is None
    assert E.last_boxed("bad \\boxed{1") is None


def test_benchmark_modes_cannot_mix():
    f = E.evaluation_config("full")
    s = E.evaluation_config("smoke")
    assert f["samples_per_problem"] == 4 and f["max_new_tokens"] == 32768
    assert s["samples_per_problem"] == 1 and s["max_new_tokens"] == 512
    assert f["thinking_enabled"] is True
    with pytest.raises(ValueError): E.evaluation_config("unsupported")


def toy_run(tag="base"):
    samples = [{"id": "x", "sample_index": 0, "correct": tag == "edited", "limit_hit": False}]
    return {"kind": "actual_gpu_inference_evaluation", "model_tag": tag,
            "dataset_sha256": "a" * 64, "protocol": E.evaluation_config("smoke"),
            "source_pins": {"a": 1}, "samples": samples}


def test_comparison_requires_same_dataset_and_protocol():
    b, e = toy_run(), toy_run("edited")
    assert R.compare(b, e)["delta_accuracy_points"] == 100
    e["dataset_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="dataset"): R.compare(b, e)


def test_comparison_rejects_published_log_disguised_as_evaluation():
    b, e = toy_run(), toy_run("edited")
    b["kind"] = "published_log_consistency_check_only"
    with pytest.raises(ValueError, match="actual GPU"): R.compare(b, e)


def test_comparison_rejects_misaligned_sample_id():
    b, e = toy_run(), toy_run("edited")
    e["samples"][0]["id"] = "unpaired"
    with pytest.raises(ValueError, match="not exactly aligned"): R.compare(b, e)


def test_reject_unpinned_edited_revision():
    class API:
        def model_info(self, *args, **kwargs):
            return type("Response", (), {"sha": "f" * 40})()
    with pytest.raises(ValueError, match="revision changed"):
        F.resolve_pins(API())


def test_accept_full_pinned_revision():
    class API:
        def model_info(self, *args, **kwargs):
            return type("Response", (), {"sha": F.EDITED_SHA_PREFIX + "0" * 32})()
    assert F.resolve_pins(API())["base"]["revision"] == V.BASE_REV
