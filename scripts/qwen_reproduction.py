#!/usr/bin/env python3
"""Qwen Experiment-8 reproduction *evidence* tooling.

No model downloads, arbitrary upstream execution, or GPU evaluations happen here.
There are separate steps for preflight and reading the original published logs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

BASE_REPO = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LOGS_URL = "https://github.com/RikiyaT/lm_understand-logs"
PATCHED_REPO = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
REPORTED_ACCURACIES = {
    "unedited": {"math500": 7.9, "gsm8k": 13.8, "ifeval": 21.0},
    "p50c": {"math500": 48.1, "gsm8k": 67.1, "ifeval": 62.3},
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _git_head(root: Path):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def preflight(logs_dir: Path) -> dict:
    try:
        import torch
        gpu = {"available": bool(torch.cuda.is_available()), "torch_version": torch.__version__,
               "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}
    except ImportError:
        gpu = {"available": False, "torch_version": None, "device_count": 0}
    d = shutil.disk_usage(Path.cwd())
    return {
        "schema_version": 1, "kind": "preflight_only_no_evaluation", "time_utc": _now(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "free_disk_gib": round(d.free / (1024 ** 3), 2),
        "cuda": gpu,
        "dependencies": {name: bool(importlib.util.find_spec(name)) for name in
                         ("huggingface_hub", "safetensors", "vllm", "math_verify")},
        "logs_directory_exists": logs_dir.is_dir(),
        "logs_git_commit": _git_head(logs_dir) if logs_dir.is_dir() else None,
        "required_base": {"repo": BASE_REPO, "revision": BASE_REVISION},
        "patched_release": PATCHED_REPO,
        "status": "preflight_does_not_prove_reproduction",
    }


def published_logs_report(logs_dir: Path, expected_commit: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("--expected-logs-commit must be a full 40-digit lowercase SHA")
    observed = _git_head(logs_dir)
    if observed != expected_commit:
        raise ValueError("Upstream logs checkout is missing or does not match its independently pinned commit")
    datasets = {}
    for run, expected in REPORTED_ACCURACIES.items():
        path = logs_dir / "exp8_math" / "eval" / run / "merged.json"
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
            raise ValueError(f"Published summary missing/invalid: {path}")
        data = json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite {x}")))
        if type(data) is not dict:
            raise ValueError("Unexpected merged.json object")
        scores = {}
        for benchmark, reported in expected.items():
            section = data.get(benchmark)
            if type(section) is not dict or type(section.get("acc")) not in (float, int):
                raise ValueError(f"Missing {run}/{benchmark}.acc")
            accuracy = float(section["acc"])
            # Upstream merged.json stores fractions (e.g. 0.481 = 48.1%).
            # The published README prints percentages; never mix scales.
            if not 0 <= accuracy <= 1 or abs(accuracy * 100 - reported) > 0.06:
                raise ValueError(f"Published {run}/{benchmark} value differs from expected {reported}: {accuracy}")
            scores[benchmark] = round(accuracy * 100, 5)
        datasets[run] = {"scores_percent": scores, "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"schema_version": 1, "kind": "published_log_consistency_check_only",
            "time_utc": _now(), "source_url": LOGS_URL, "logs_git_commit": observed,
            "data": datasets, "not_verified": ["original Qwen checkpoint bytes", "BF16 edited checkpoint reconstruction",
                                      "GPU inference", "MATH-500 re-evaluation", "agent edit-generation trajectory"],
            "interpretation": "This checks published summary files; it does NOT reproduce model inference."}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "published-logs"))
    p.add_argument("--logs-dir", type=Path, default=Path("external/lm_understand-logs"))
    p.add_argument("--expected-logs-commit", default=None)
    p.add_argument("--out", type=Path, help="New evidence report path; existing files are never overwritten")
    args = p.parse_args(argv)
    try:
        if args.command == "preflight": result = preflight(args.logs_dir)
        else:
            if not args.expected_logs_commit: raise ValueError("The logs commit must be independently pinned")
            result = published_logs_report(args.logs_dir, args.expected_logs_commit)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out:
            if args.out.is_symlink() or any(p.is_symlink() for p in args.out.parents):
                raise ValueError("Symlinked output path is not permitted")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as f: f.write(encoded)
        print(encoded, end="")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("REPRODUCTION CHECK FAILED:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
