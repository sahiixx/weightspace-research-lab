# Qwen checkpoint reconstruction and GPU benchmark: v0.6 execution protocol

This repository now contains **two independent experimental claims**, and neither must be inferred from the other:

1. **Exact edit reconstruction:** given two externally downloaded public model checkpoints and the authors' sparse `p50c` patch, independently verify all safetensors tensors are identical except the released two, bitwise reconstruct those edits with NumPy, match published BF16 SHA-256 hashes, and count exactly **130,188 changed stored values**. This is a CPU operation and does not evaluate reasoning.
2. **Benchmark reproduction:** load both verified model checkpoints and measure real model generations under identical sampling settings, using pinned MATH-500 data and `math-verify==0.9.0`. This requires a GPU and compatible modern vLLM/Transformers. The full mode is 500 problems × 4 generations per model, max 32,768 new tokens. A five-problem smoke check is not a paper replication.

Both demand careful provenance handling. The Ed25519 approval system signs *bytes of the comparison report*, not the truth of a third-party checkpoint or the safety of outputs.

## Public source pins

| Evidence | Pin |
| --- | --- |
| Qwen base `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
| Edited release `RikiyaT/Qwen3.5-0.8B-weight-space-programming` | Main is accepted **only if** its resolved 40-hex SHA begins `d1671a17`; the **complete** resolved SHA is stored in `source-pins.json` |
| Original logs `RikiyaT/lm_understand-logs` | `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` |
| `HuggingFaceH4/MATH-500` dataset | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| p50c edited tensor layer 15 | `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53` |
| p50c edited tensor layer 19 | `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2` |

The exact upstream patch manifest says layer 15 has **65,260** changed BF16 values and layer 19 **64,928**, total **130,188**. The commonly repeated 131,072 is the number of targeted positions, including entries that round back to their original BF16 values. This distinction is relevant to the changed-parameter fraction.

**Security caution:** Hub download provenance is based on the repository/revisions and content-addressed artifacts, not on our lab's patch-approval keys. A repository SHA-1 is a content identifier, not a human identity proof. No remote `trust_remote_code`, pickle checkpoint, or automatically executed upstream Python script is required for the independent CPU test.

## A. Independent CPU checkpoint check on GitHub

The new `.github/workflows/qwen-real-artifacts.yml` runs automatically once on the v0.6 research branch (or manually on `main` after merge). It uses a standard hosted Ubuntu runner, downloads approximately 4 GB of public artifacts, and uploads only small JSON reports, **never model weights**.

From a local machine with adequate disk/network:

```bash
python -m pip install 'torch>=2.3,<3' 'numpy>=1.26,<3' 'safetensors>=0.4,<1' 'huggingface_hub>=1,<2'
OFFICIAL_LOGS_COMMIT=4d09ba6d6a7e891dd8063974dfe81b3d78e3494f \
  bash scripts/get_official_resources.sh
python scripts/qwen_fetch_pinned.py --root external/qwen-real
python scripts/qwen_weight_verify.py \
  --base-dir external/qwen-real/base \
  --edited-dir external/qwen-real/edited \
  --patch-manifest external/lm_understand-logs/exp8_math/runs/p50c/edit/manifest.json \
  --patch-npz external/lm_understand-logs/exp8_math/runs/p50c/edit/edit.npz \
  --out runs/qwen-real-verified.json
```

A passing JSON document with `kind: real_checkpoint_bitwise_reconstruction_no_inference`, `tensor_count: 488`, `changed_bf16_entries: 130188`, and precisely two changed tensor names is independent reconstruction evidence. It **does not** prove better math results or re-enact Astra's weight-edit generation.

## B. Explicit GPU evaluation, manual only

The `.github/workflows/qwen-gpu-evaluation.yml` only supports manually dispatched jobs using a self-hosted Linux x86 runner labeled `qwen-gpu`, equipped with CUDA and sufficient VRAM/disk. **It does not start on pushes or pull requests.** You must provision/configure this runner independently; none is provisioned automatically here and no paid compute has been authorized.

The workflow installs GPU dependencies, downloads the same pinned checkpoints, then runs one process for the baseline and a separate process for the edited model. It stores sample scores and gzipped text generations separately.

Run locally on a compatible GPU environment after installing vLLM, recent Transformers and `math-verify==0.9.0`:

```bash
python scripts/qwen_gpu_eval.py --mode smoke --model-tag base \
  --model external/qwen-real/base --dataset external/qwen-real/dataset/test.jsonl \
  --source-pins external/qwen-real/source-pins.json --output runs/gpu-base
python scripts/qwen_gpu_eval.py --mode smoke --model-tag edited \
  --model external/qwen-real/edited --dataset external/qwen-real/dataset/test.jsonl \
  --source-pins external/qwen-real/source-pins.json --output runs/gpu-edited
python scripts/qwen_evidence.py compare --base runs/gpu-base/report.json \
  --edited runs/gpu-edited/report.json --weight-report runs/qwen-real-verified.json \
  --out runs/gpu-comparison.json
```

For comparable full benchmarks, substitute `--mode full` for both models: this uses 500 problems × 4 samples, 32,768 max new tokens, thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 and seed 0. Verify actual environment/memory compatibility before launching; the full experiment can be expensive.

A separate decoding control is available as `--protocol nothink35` for *both* arms. That switches to non-thinking, temperature 0.7, top-p 0.8 and an 8,192-token full-mode cap. Keep these runs as a distinct comparison; never mix their scores with `think35`.

**Limits of this independent evaluator:** It uses the original boxed answer prompt and `math-verify` equivalence contract, but is a new harness, not a byte-identical replay of upstream vLLM internals. Record vLLM, tokenizer, CUDA, driver, grading versions, and numerical precision when comparing results. Repeating full scores is not re-creating the author's search trajectory. The published research scores are never inserted as new measurement results.

## C. Operator-controlled Trust Core signing (separate environment)

After independently inspecting the paired report and source artifacts, an authorized person can sign the JSON comparison **outside the untrusted agent and CI runner**:

```bash
python scripts/qwen_evidence.py sign --report runs/gpu-comparison.json \
  --private-key /protected/approver-ed25519.pem --signer-id qwen-reviewer \
  --source-commit YOUR_EXACT_40_HEX_GIT_COMMIT --run-id YOUR_RUN_ID \
  --out /protected/qwen-approval.json
python scripts/qwen_evidence.py verify --report runs/gpu-comparison.json \
  --envelope /protected/qwen-approval.json --policy /protected/trusted.json \
  --source-commit YOUR_EXACT_40_HEX_GIT_COMMIT
```

`trusted.json` must grant the key the `qwen_evaluation` purpose and allowlist `https://github.com/sahiixx/weightspace-research-lab`. No signing keys or approval secrets belong in GitHub Actions/PR workspaces. GitHub's checkmarked workflow result is **not** an operator signature or externally immutable audit record.

## D. What is verified in this project

- Local automated tests: tiny BF16 checkpoints, exact sparse reconstruction, unexpected tensor detection, forged hashes/indices, dataset/sampling alignment, and source revision guards.
- Real checkpoint byte verification: **only when the CPU workflow has downloaded both model snapshots and a successful real-checkpoint report exists.**
- GPU math-score reproduction: **only if the optional GPU workflow has run successfully and there are paired original/edited generation artifacts.**
- Production security/certification: **not established**.

Official logs: https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math

Checkpoint: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
