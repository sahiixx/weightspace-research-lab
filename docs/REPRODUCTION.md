# Official research reproduction: separate from the bundled demo

This guide is a pointer to the **original authors' published code**, not a second implementation of their generation/evaluation harness.

## A. Rebuild the published weight diff (research-scale, optional)

1. Run `bash scripts/get_official_resources.sh` to get the real `RikiyaT/lm_understand-logs` repository into `external/`.
2. Read `external/lm_understand-logs/exp8_math/README.md`, particularly "The weight edits".
3. For the 50-pass Qwen experiment, the patch manifest and diff are at `exp8_math/runs/p50c/edit/` and the official reconstruction script is `exp8_math/code/apply_edit.py`.
4. Obtain the original `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17` as the experiment specifies. The editing logs and diff do not make the underlying model weight download optional for actual reconstruction.
5. The verified upstream command takes three positional arguments (base safetensors,
edit directory, output). With `external/lm_understand-logs/` downloaded and a
single original base `model.safetensors` file available, run:

```bash
bash scripts/rebuild_official.sh /path/to/base/model.safetensors external/qwen-rebuilt.safetensors
```

This wrapper invokes the upstream `exp8_math/code/apply_edit.py` and its SHA-256
checks. Ensure you use the *exact original BF16 base revision* above, not an arbitrary
quantized or converted model. Upstream dependencies are required; inspect the source.
The full Hugging Face checkpoint may be sharded in another release and may need
conversion or consolidation before using this one-file upstream command.
No upstream scripts or weights are bundled here.

## B. Verify the *published release* without generating the edit

```bash
python -m pip install -e '.[official]'
hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming --local-dir external/qwen-edited
python scripts/verify_official_qwen.py --patched-dir external/qwen-edited
```

Expected raw BF16-byte SHA-256 digests, copied from the released model card:

- `model.language_model.layers.15.mlp.down_proj.weight`:
  `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53`
- `model.language_model.layers.19.mlp.down_proj.weight`:
  `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2`

This utility checks **only those two named tensors**. It does not assert untouched tensors elsewhere match the original checkpoint.

## C. Run the original benchmark (requires a suitable GPU)

Follow `external/lm_understand-logs/exp8_math/code/evaluate.py` and `exp8_math/README.md` under the upstream repository. Its protocol for Experiment 8 includes:

- vLLM with the Qwen3.5-0.8B text model, thinking enabled;
- temperature `0.6`, `top_p=0.95`, `top_k=20`, presence penalty `1.5`;
- up to **32,768** generated tokens per answer;
- grade the last `\\boxed{...}` with `math-verify`;
- **four** samples per MATH-500 item, **two** per GSM8K item; other datasets use different settings.

The authors' 50-pass editor itself is expensive: they allowed CPU Python analysis of weights, 50 forward-pass tool calls and extended agent reasoning. The authors publish harness files and complete experiment turn logs, but reproducing the editor's *decision-making trajectory* needs the same external model/service, inference environment and compute budget. This package does not implement or run that harness.

The edited checkpoint on Hugging Face includes an independent reported MATH-500 run under the paper's protocol (51.0%). These numbers are historical published references, not freshly reproduced results here. Always report exact sampling settings, output-token cap and thinking mode with any model-accuracy claim.

## D. Local repeatable demonstration (CPU)

```bash
python -m pip install -e '.[test]'
python -m wslab pipeline --out runs/demo --count 2048 --seed 42
python -m pytest -q
```

This is the only benchmark computed automatically by the bundled project. Its score reflects a synthetic linear decision rule and has no comparability to MATH-500, GSM8K or IFEval.

## E. Verification boundaries

The official patched release and the authors' independently run 51.0% MATH-500
result are published evidence, not results calculated in this local educational
repo. The paper's headline 48.1% vs 7.9% uses thinking-enabled, high-token-limit
sampling; it largely mitigates looping and should not be presented as proof of a
new model capability under every inference configuration. Always compare baseline
settings and review untouched metrics. See the sources in `docs/RESEARCH.md`.
