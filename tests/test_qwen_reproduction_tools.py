"""Evidence tool tests are independent of GPUs/downloads."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qwen_reproduction.py"
spec = importlib.util.spec_from_file_location("qwen_reproduction", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cpu_preflight_is_explicitly_not_reproduction(tmp_path):
    info = module.preflight(tmp_path / "missing")
    assert info["status"] == "preflight_does_not_prove_reproduction"
    assert info["required_base"]["revision"] == module.BASE_REVISION
    assert info["logs_git_commit"] is None


def test_log_integrity_check_requires_verified_commit(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.org"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    for run, benchmarks in module.REPORTED_ACCURACIES.items():
        path = tmp_path / "exp8_math" / "eval" / run / "merged.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({key: {"acc": val / 100} for key,val in benchmarks.items()}))
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "test evidence"], check=True)
    head = module._git_head(tmp_path)
    report = module.published_logs_report(tmp_path, head)
    assert report["kind"] == "published_log_consistency_check_only"
    assert "GPU inference" in report["not_verified"]
    assert report["data"]["p50c"]["scores_percent"]["math500"] == 48.1
    with pytest.raises(ValueError, match="does not match"):
        module.published_logs_report(tmp_path, "a" * 40)
    path = tmp_path / "exp8_math" / "eval" / "p50c" / "merged.json"
    data = json.loads(path.read_text())
    data["math500"]["acc"] = 90
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differs"):
        module.published_logs_report(tmp_path, head)
