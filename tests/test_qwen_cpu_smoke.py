"""No-network structure / integrity tests for real CPU smoke harness."""
import importlib.util
from pathlib import Path
import json
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qwen_cpu_smoke.py"
spec = importlib.util.spec_from_file_location("qwen_cpu_smoke", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make(tag):
    return module.create_report(model_tag=tag, model_revision="a" * 40,
        prompt="Test", response="12", generated_tokens=2,
        max_new_tokens=32, runtime={"torch":"test"})


def test_real_smoke_is_explicitly_not_benchmark():
    r = make("base")
    assert "not_benchmark" in r["kind"]
    assert r["prompt_sha256"] == module.digest(b"Test")
    assert "accuracy_percent" not in r


def test_bad_counts_and_tags_rejected():
    with pytest.raises(ValueError):
        module.create_report(model_tag="invalid", model_revision="a"*40,
            prompt="x", response="y", generated_tokens=1,
            max_new_tokens=32, runtime={})
    with pytest.raises(ValueError):
        module.create_report(model_tag="base", model_revision="a"*40,
            prompt="x", response="y", generated_tokens=33,
            max_new_tokens=32, runtime={})


def test_compare_rejects_mismatched_prompts(tmp_path):
    b = make("base")
    e = make("edited")
    e["prompt"] = "different"
    for name, obj in (("b",b),("e",e)):
        (tmp_path / (name+".json")).write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        module.compare(tmp_path/"b.json", tmp_path/"e.json", tmp_path/"out.json")


def test_compare_rejects_overwrite(tmp_path):
    b, e = make("base"), make("edited")
    for name,obj in (("b",b),("e",e)):
        (tmp_path / (name+".json")).write_text(json.dumps(obj))
    out = tmp_path / "out.json"
    r = module.compare(tmp_path/"b.json", tmp_path/"e.json", out)
    assert r["math500_accuracy_measured"] is False
    with pytest.raises(ValueError):
        module.compare(tmp_path/"b.json", tmp_path/"e.json", out)


def test_pinned_downloader_copies_required_processor_configs():
    downloader = (SCRIPT.parent / "qwen_fetch_pinned.py").read_text()
    assert '"preprocessor_config.json"' in downloader
    assert '"processor_config.json"' in downloader
