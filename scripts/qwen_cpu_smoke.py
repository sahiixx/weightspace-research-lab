#!/usr/bin/env python3
"""Real base/edited Qwen3.5 text-generation CPU smoke test, never MATH-500 reproduction.

Loads publicly pinned multimodal checkpoints with Transformers, performs a single
small deterministic generation from each, and writes unsigned, inspectable reports.
No result is interpreted as a measure of mathematical accuracy or model safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROMPT = "What is 7 plus 5? Answer with the number only."
MODEL_IDS = {
    "base": "Qwen/Qwen3.5-0.8B",
    "edited": "RikiyaT/Qwen3.5-0.8B-weight-space-programming",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pins(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("base"), dict) or not isinstance(obj.get("edited"), dict):
        raise ValueError("Expected source-pins.json with base and edited revision records")
    return obj


def create_report(*, model_tag: str, model_revision: str, prompt: str,
                  response: str, generated_tokens: int, max_new_tokens: int,
                  runtime: dict[str, str]) -> dict[str, Any]:
    if model_tag not in MODEL_IDS:
        raise ValueError("model_tag must be base or edited")
    if not (0 <= generated_tokens <= max_new_tokens <= 128):
        raise ValueError("CPU smoke generation cap out of bounds")
    if not isinstance(response, str) or not response:
        raise ValueError("Empty model output")
    return {
        "kind": "real_qwen_cpu_inference_smoke_not_benchmark",
        "version": 1,
        "model_tag": model_tag,
        "model_id": MODEL_IDS[model_tag],
        "model_revision": model_revision,
        "utc": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "prompt_sha256": digest(prompt.encode()),
        "response": response,
        "response_sha256": digest(response.encode()),
        "generated_tokens": generated_tokens,
        "max_new_tokens": max_new_tokens,
        "decode": "greedy_non_thinking",
        "runtime": runtime,
        "limitations": [
            "Single prompt, CPU, greedy, non-thinking, short token cap",
            "Not the published thinking-mode MATH-500 protocol",
            "Not evidence of improvement, generalization, or model safety",
            "Unsigned execution report, not operator-approved attestation",
        ],
    }


def run(model_dir: Path, pins_path: Path, model_tag: str, output: Path,
        max_new_tokens: int) -> dict:
    if model_tag not in MODEL_IDS or not 1 <= max_new_tokens <= 128:
        raise ValueError("Invalid model tag or max_new_tokens")
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to overwrite output")
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError("Expected downloaded local model directory")
    pins = load_pins(pins_path)
    # Identifying fields are always reported; no trusted model signature is claimed.
    model_info = pins[model_tag]
    revision = str(model_info.get("resolved_revision") or model_info.get("revision") or "unknown")
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        raise ValueError("Pinned model revision must be a resolved 40-character Git commit")
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    from importlib.metadata import version

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=False)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, trust_remote_code=False,
        low_cpu_mem_usage=True, device_map="cpu").eval()
    prompt = PROMPT
    encoded = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", enable_thinking=False)
    with torch.inference_mode():
        tokens = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                                use_cache=True)
    input_len = int(encoded["input_ids"].shape[-1])
    new_tokens = tokens[0, input_len:]
    response = processor.decode(new_tokens, skip_special_tokens=False)
    report = create_report(model_tag=model_tag, model_revision=revision,
                           prompt=prompt, response=response,
                           generated_tokens=int(new_tokens.numel()),
                           max_new_tokens=max_new_tokens,
                           runtime={"torch": torch.__version__,
                                    "transformers": version("transformers"),
                                    "cpu_threads": str(torch.get_num_threads())})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return report


def compare(base_path: Path, edited_path: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to overwrite comparison")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    edited = json.loads(edited_path.read_text(encoding="utf-8"))
    if any(x.get("kind") != "real_qwen_cpu_inference_smoke_not_benchmark" for x in (base, edited)):
        raise ValueError("Expected paired CPU inference smoke reports")
    if (base.get("model_tag"), edited.get("model_tag")) != ("base", "edited"):
        raise ValueError("Expected base and edited tags")
    for key in ("prompt_sha256", "prompt", "max_new_tokens", "decode"):
        if base[key] != edited[key]:
            raise ValueError(f"Mismatched run configuration: {key}")
    result = {
        "kind": "paired_real_qwen_cpu_smoke_not_benchmark",
        "version": 1,
        "base_model_revision": base["model_revision"],
        "edited_model_revision": edited["model_revision"],
        "prompt_sha256": base["prompt_sha256"],
        "max_new_tokens": base["max_new_tokens"],
        "base_response_sha256": base["response_sha256"],
        "edited_response_sha256": edited["response_sha256"],
        "outputs_differ": base["response"] != edited["response"],
        "baseline_generation_confirmed": True,
        "edited_generation_confirmed": True,
        "math500_accuracy_measured": False,
        "scope": "One short greedy non-thinking generation per model; not a comparative benchmark",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-dir", type=Path, required=True)
    r.add_argument("--source-pins", type=Path, required=True)
    r.add_argument("--model-tag", choices=("base", "edited"), required=True)
    r.add_argument("--max-new-tokens", type=int, default=32)
    r.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--base", type=Path, required=True)
    c.add_argument("--edited", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.command == "run":
            report = run(a.model_dir, a.source_pins, a.model_tag, a.output, a.max_new_tokens)
        else:
            report = compare(a.base, a.edited, a.output)
        print(json.dumps({k: v for k, v in report.items() if k in ("kind", "model_tag", "response_sha256", "outputs_differ", "math500_accuracy_measured")}))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as e:
        print(f"CPU smoke test failed: {e}", file=__import__('sys').stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
