# Real Qwen checkpoint verification evidence

**Date:** 2026-11-06 (UTC)  
**Repository workflow:** [GitHub Actions run #42205689279](https://github.com/sahiixx/weightspace-research-lab/actions/runs/42205689279)  
**Result:** Completed successfully on a GitHub-hosted CPU runner.

This folder is an archival copy of the two small JSON reports downloaded from that workflow. It contains no model weights, no secrets, and **no GPU benchmark outputs**. It preserves evidence after the Actions artifact expires.

## What was verified
- Pinned Qwen3.5-0.8B base and edited checkpoint downloads.
- Inventories, shapes and dtypes for **488 tensors**.
- Exactly two edited MLP down-projection tensors, with **130,188 changed BF16 entries** in total.
- Independent exact-bit reconstruction from the pinned upstream sparse edit and comparison against released edited checkpoint tensors.
- Reported per-tensor SHA-256 hashes agree with the pinned patch manifest.

## Limitations
These reports do **not** verify inference, any MATH-500 score, the original model-editing agent's trajectory, or the publisher identity beyond the pinned public sources. The files here are **unsigned**; a GitHub commit records bytes and history but is not a production cryptographic approval. A separately controlled approver must verify and sign any formal attestation.

## File SHA-256
| File | SHA-256 |
|---|---|
| `verified.json` | `98f889a48787e724d1aa2a73c08967213798f59db2b515830a3f774aa3944acd` |
| `source-pins.json` | `8bce32d0ba48975e107ded04771f6596062828efcd77ba1e376357cfcaf3ab7a` |


## Remaining reproduction step
The workflow `.github/workflows/qwen-gpu-evaluation.yml` requires a GPU-equipped self-hosted GitHub Actions runner labelled `qwen-gpu`. GitHub's standard `ubuntu-latest` runners do not have NVIDIA GPUs. A full independent MATH-500 performance result is **not available** until the `full` GPU workflow completes with a verified run artifact and separately approved signing process.
