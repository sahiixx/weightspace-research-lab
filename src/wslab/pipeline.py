"""Reproducible local end-to-end demonstration and generated evidence."""
from __future__ import annotations

from pathlib import Path
import platform
import sys
import torch
from .benchmarks import evaluate
from .checkpoints import fingerprint, load_checkpoint, save_checkpoint, write_json
from .data import samples
from .models import corrupted_model, model_from_state, reference_model
from .safety import apply_checked, validation_gate
from .search import propose


def verify_exact(base: dict[str, torch.Tensor], patched: dict[str, torch.Tensor],
                 patch: dict) -> dict:
    expected = apply_checked(base, patch)
    if set(expected) != set(patched) or any(not torch.equal(expected[k], patched[k]) for k in expected):
        raise ValueError("Patched checkpoint differs from the approved exact patch")
    changed = sum(int((base[k] != patched[k]).sum().item()) for k in base)
    return {"matched_expected_patch": True, "changed_elements": changed,
            "base_sha256": fingerprint(base), "patched_sha256": fingerprint(patched)}


def format_report(report: dict) -> str:
    def row(label: str, record: dict) -> str:
        val = record["validation"]
        test = record["test"]
        return f"| {label} | {val['accuracy']:.2%} | {test['accuracy']:.2%} | {test['cross_entropy']:.4f} |"
    return "\n".join([
        "# Weight-Space Research Lab: generated run",
        "",
        "**Experiment:** corrupted deterministic linear classifier, one edited scalar, no gradients.",
        "**Important:** this is an educational analogue, NOT a reproduction of the Qwen benchmark.",
        "",
        "| Checkpoint | Validation accuracy | Holdout test accuracy | Test cross-entropy |",
        "|---|---:|---:|---:|",
        row("Reference (known synthetic rule)", report["reference"]),
        row("Corrupted baseline", report["base"]),
        row("Approved edit", report["patched"]),
        "",
        f"- Validation candidate probes: {report['search']['candidates_evaluated']}.",
        f"- Runtime: Python {report['runtime']['python']}, PyTorch {report['runtime']['torch']}.",
        f"- Weight elements changed: {report['verification']['changed_elements']}.",
        f"- Original checkpoint SHA-256: `{report['verification']['base_sha256']}`.",
        f"- Modified checkpoint SHA-256: `{report['verification']['patched_sha256']}`.",
        "- The candidate was selected using the validation split; holdout test was scored afterward.",
        "- SHA-256 is a reproducibility fingerprint, NOT a signed authenticity claim.",
        "- Training, finetuning, LLM tool access and full-scale Qwen experiments were not run.",
        "",
        "Refer to `docs/RESEARCH.md` and `docs/REPRODUCTION.md` for primary-source evidence and limitations.",
        "",
    ])


def run_pipeline(out: str | Path, seed: int = 42, count: int = 2048) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    validation = samples("validation", seed=seed, count=count)
    test = samples("test", seed=seed, count=count)
    reference = reference_model().state_dict()
    original = corrupted_model().state_dict()
    save_checkpoint(out / "base.pt", original)
    patch, search_report = propose(original, validation)
    write_json(out / "patch.json", patch)
    edited = apply_checked(original, patch)
    validation_gate(evaluate(model_from_state(original), validation),
                    evaluate(model_from_state(edited), validation))
    save_checkpoint(out / "patched.pt", edited)
    verification = verify_exact(load_checkpoint(out / "base.pt"),
                                load_checkpoint(out / "patched.pt"), patch)
    report = {
        "project": "weightspace-research-lab",
        "experiment": "toy-linear-repair",
        "seed": seed,
        "samples_per_split": count,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform()},
        "reference": {s.split: evaluate(model_from_state(reference), s).to_dict() for s in (validation, test)},
        "base": {s.split: evaluate(model_from_state(original), s).to_dict() for s in (validation, test)},
        "patched": {s.split: evaluate(model_from_state(edited), s).to_dict() for s in (validation, test)},
        "search": search_report,
        "verification": verification,
        "notes": ["Synthetic deterministic task, not an LLM edit", "Test split never used for candidate selection"],
    }
    write_json(out / "report.json", report)
    (out / "report.md").write_text(format_report(report), encoding="utf-8")
    return report
