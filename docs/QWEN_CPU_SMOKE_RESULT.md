# Qwen3.5-0.8B real CPU smoke: measured evidence

- **GitHub Actions run:** https://github.com/sahiixx/weightspace-research-lab/actions/runs/42208277814
- **Revision tested:** `16d20b46818fc550d074c430a31eaa0754581c49`
- **Execution:** GitHub-hosted Ubuntu CPU; PyTorch 2.6.0+cpu; Transformers 5.22.1; 4 CPU threads
- **Original checkpoint:** `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`
- **Patched checkpoint:** `RikiyaT/Qwen3.5-0.8B-weight-space-programming` at `d1671a178705bcab7644806d23c535517f35723b`
- **Prompt (same for both):** `What is 7 plus 5? Answer with the number only.`
- **Decode:** greedy, non-thinking, maximum 32 generated tokens per model
- **Actual tokens generated:** base 5, edited 5
- **Base textual output:** `12<|im_end|>\n<|endoftext|>`
- **Edited textual output:** `12<|im_end|>\n<|endoftext|>`
- **Outputs differ:** `False`
- **Response SHA-256:** `f27c438a1e6e8ae6568e70abcb634cc2ce545f788eaa9f4dad7d8be65bb4bfa9` (both models)
- **Evidence ZIP SHA-256:** `f531acfb57ea76e1285a6c20602880505842a43f040b1e46ddfeb2d220dea03a`

The Actions artifact stores `base.json`, `edited.json`, and `paired.json`.
Those are unsigned machine-generated observations, *not human-attested evidence*.
The original task's research benchmark was **not** run: no 500-question
MATH-500 evaluation, no four-sample protocol, no thinking-mode 32k generation,
no vLLM GPU evaluation, and no MATH-500 accuracy estimate. A single
identical answer cannot establish that the patch improves reasoning or has
any safety property.

The independent real-checkpoint reconstruction was separately successful:
https://github.com/sahiixx/weightspace-research-lab/actions/runs/42205689279

To run the full test, provision a GPU runner labeled `qwen-gpu` and use the
`qwen-gpu-evaluation.yml` manual workflow with mode `full`, protocol `think35`,
then validate its paired raw outputs. No GPU runner has been provisioned
by this lab, and no paid compute was charged for this CPU smoke.
