import json
import subprocess
import sys
import torch
import pytest
from wslab.checkpoints import fingerprint, load_checkpoint
from wslab.pipeline import run_pipeline, verify_exact
from wslab.checkpoints import read_json


def test_pipeline_repair_holdout_and_evidence(tmp_path):
    report = run_pipeline(tmp_path, seed=42, count=512)
    assert report["base"]["test"]["accuracy"] < report["patched"]["test"]["accuracy"]
    assert report["patched"]["test"]["accuracy"] >= 0.96
    assert report["verification"]["changed_elements"] == 1
    assert report["search"]["selection_split"] == "validation"
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "patch.json").exists()
    assert verify_exact(load_checkpoint(tmp_path / "base.pt"),
                        load_checkpoint(tmp_path / "patched.pt"),
                        read_json(tmp_path / "patch.json"))["matched_expected_patch"]


def test_checkpoint_hash_stable_roundtrip(tmp_path):
    run_pipeline(tmp_path, count=256)
    base = load_checkpoint(tmp_path / "base.pt")
    assert fingerprint(base) == read_json(tmp_path / "patch.json")["base_sha256"]


def test_verify_fails_if_checkpoint_has_extra_edit(tmp_path):
    run_pipeline(tmp_path, count=256)
    base = load_checkpoint(tmp_path / "base.pt")
    edited = load_checkpoint(tmp_path / "patched.pt")
    edited["linear.bias"][0] += 0.1
    with pytest.raises(ValueError, match="differs"):
        verify_exact(base, edited, read_json(tmp_path / "patch.json"))


def test_python_module_cli(tmp_path):
    run = subprocess.run([sys.executable, "-m", "wslab", "pipeline", "--out", str(tmp_path),
                          "--count", "256"], capture_output=True, text=True, check=True)
    result = json.loads(run.stdout)
    assert result["status"] == "success"
    result = subprocess.run([sys.executable, "-m", "wslab", "verify", "--base",
                             str(tmp_path / "base.pt"), "--patched", str(tmp_path / "patched.pt"),
                             "--patch", str(tmp_path / "patch.json")],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["changed_elements"] == 1
