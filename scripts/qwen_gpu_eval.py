#!/usr/bin/env python3
"""Independent paired Qwen MATH-500 evaluation for an explicitly provisioned GPU.

Modes are deliberately distinct: smoke is not comparable with the paper;
full uses the published Qwen3.5 thinking-mode sampling and math-verify 0.9.0.
Never invoked by the standard CPU test workflow or Codex proposal tools.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

PROTOCOL = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
    "seed": 0,
    "thinking_enabled": True,
    "final_answer_instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
}


def last_boxed(text: str) -> str | None:
    """Last balanced boxed/fbox expression, retaining braces within fractions."""
    if type(text) is not str:
        raise ValueError("Generated response must be text")
    choices = [text.rfind("\\boxed{"), text.rfind("\\fbox{")]
    ix = max(choices)
    if ix < 0:
        return None
    brace = text.find("{", ix)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:i]
    return None


def grade(gold: str, predicted: str | None) -> bool:
    """Match upstream grader's math-verify 0.9.0 boxed-string contract."""
    if predicted is None:
        return False
    from math_verify import parse, verify
    return bool(verify(parse("\\boxed{" + str(gold) + "}"),
                       parse("\\boxed{" + predicted + "}")))


def load_cases(dataset: Path, mode: str) -> tuple[list[dict], str]:
    if mode not in ("smoke", "full"):
        raise ValueError("Unsupported benchmark mode")
    if not dataset.is_file() or dataset.stat().st_size > 2_000_000:
        raise ValueError("Invalid local MATH-500 JSONL file")
    content = dataset.read_bytes()
    cases = [json.loads(line) for line in content.splitlines() if line.strip()]
    if len(cases) != 500 or any(type(r) is not dict or type(r.get("problem")) is not str
                                or type(r.get("answer")) is not str for r in cases):
        raise ValueError("MATH-500 must contain 500 labeled math problems")
    selected = cases if mode == "full" else cases[:5]
    return selected, hashlib.sha256(content).hexdigest()


def evaluation_config(mode: str, protocol: str = "think35") -> dict:
    if protocol not in ("think35", "nothink35"):
        raise ValueError("Unsupported benchmark protocol")
    base = dict(PROTOCOL, protocol=protocol)
    if protocol == "nothink35":
        base.update(temperature=0.7, top_p=0.8, thinking_enabled=False)
    if mode == "full":
        return {**base, "mode": mode, "problems": 500, "samples_per_problem": 4,
                "max_new_tokens": 32768 if protocol == "think35" else 8192,
                "context_reserve": 2048, "grader": "math-verify==0.9.0"}
    if mode == "smoke":
        return {**base, "mode": mode, "problems": 5, "samples_per_problem": 1,
                "max_new_tokens": 512, "context_reserve": 2048, "grader": "math-verify==0.9.0"}
    raise ValueError("Unsupported evaluation mode")


def evaluate(model: Path, dataset: Path, output: Path, mode: str, *,
             model_tag: str, source_pins: Path | None = None,
             protocol: str = "think35") -> dict:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    if model_tag not in ("base", "edited"):
        raise ValueError("--model-tag must be base or edited")
    if not model.is_dir():
        raise ValueError("Model directory missing")
    if output.exists():
        raise ValueError("Evidence output directory exists; no overwriting allowed")
    config = evaluation_config(mode, protocol)
    cases, dsha = load_cases(dataset, mode)
    pins = json.loads(source_pins.read_text("utf-8")) if source_pins else None
    if pins and pins.get("dataset", {}).get("test_jsonl_sha256") != dsha:
        raise ValueError("Dataset hash differs from pinned download manifest")
    tok = AutoTokenizer.from_pretrained(str(model), trust_remote_code=False)
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" +
                 PROTOCOL["final_answer_instruction"]}], tokenize=False,
                 add_generation_prompt=True, enable_thinking=config["thinking_enabled"]) for r in cases]
    engine = LLM(model=str(model), dtype="bfloat16", trust_remote_code=False,
                 max_model_len=config["max_new_tokens"] + config["context_reserve"],
                 seed=0, gpu_memory_utilization=0.85)
    sampling = SamplingParams(temperature=config["temperature"], top_p=config["top_p"],
                              top_k=config["top_k"], presence_penalty=config["presence_penalty"], max_tokens=config["max_new_tokens"],
                              n=config["samples_per_problem"], seed=0)
    output.mkdir(parents=True, exist_ok=False)
    scores = []
    truncated = 0
    total = 0
    with gzip.open(output / "generations.jsonl.gz", "wt", encoding="utf-8") as sink:
        for start in range(0, len(prompts), 10):
            results = engine.generate(prompts[start:start + 10], sampling, use_tqdm=False)
            for case_offset, (case, response) in enumerate(zip(cases[start:start + 10], results)):
                for sample_index, response_item in enumerate(response.outputs):
                    text = response_item.text
                    pred = last_boxed(text)
                    correct = grade(case["answer"], pred)
                    limit_hit = response_item.finish_reason == "length"
                    truncated += int(limit_hit)
                    total += 1
                    scores.append({"id": case.get("unique_id", str(start + case_offset)), "sample_index": sample_index,
                                   "correct": correct, "limit_hit": limit_hit,
                                   "tokens": len(response_item.token_ids),
                                   "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
                    sink.write(json.dumps({"id": case.get("unique_id", str(start + case_offset)),
                                           "sample_index": sample_index, "answer": case["answer"],
                                           "prediction": pred, "correct": correct, "finish_reason": response_item.finish_reason,
                                           "text": text}, ensure_ascii=False) + "\n")
    from importlib.metadata import version as distribution_version
    import torch
    report = {"kind": "actual_gpu_inference_evaluation", "schema_version": 1,
              "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(0),
                          "vllm": distribution_version("vllm"),
                          "transformers": distribution_version("transformers"),
                          "math_verify": distribution_version("math-verify")},
              "time_utc": datetime.now(timezone.utc).isoformat(), "model_tag": model_tag,
              "model_path": str(model.resolve()), "dataset_sha256": dsha,
              "source_pins": pins, "protocol": config,
              "count": total, "correct": sum(s["correct"] for s in scores),
              "accuracy_percent": round(100 * sum(s["correct"] for s in scores) / total, 4),
              "limit_hit_percent": round(100 * truncated / total, 4),
              "generations_sha256": hashlib.sha256((output / "generations.jsonl.gz").read_bytes()).hexdigest(),
              "samples": scores,
              "limitations": ["New independent harness; check exact parity with upstream prompt/grader versions",
                              "Small smoke mode is not comparable with published MATH-500 scores"]}
    with (output / "report.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--model-tag", choices=("base", "edited"), required=True)
    p.add_argument("--protocol", choices=("think35", "nothink35"), default="think35")
    p.add_argument("--source-pins", type=Path)
    a = p.parse_args()
    try:
        r = evaluate(a.model, a.dataset, a.output, a.mode, model_tag=a.model_tag,
                     source_pins=a.source_pins, protocol=a.protocol)
        print(json.dumps({k: r[k] for k in ("model_tag", "accuracy_percent", "count", "limit_hit_percent")}))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print("GPU benchmark failed:", exc, file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
