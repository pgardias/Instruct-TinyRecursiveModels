# TinyRecursiveModels – Instruction Tuning

This repository layers SlimOrca-style instruction tuning and inference workflows on top of the original [TinyRecursiveModels](docs/TINYRECURSIVEMODELS_README.md) project. Everything in the upstream codebase remains available; we add lightweight dataset plumbing, training scripts, and CLIs geared toward conversational supervision.

> **Upstream credit:** the base project is Alexia Jolicoeur-Martineau’s [TinyRecursiveModels](https://github.com/alexiajolicoeurmartineau/TinyRecursiveModels) implementation from *Less is More: Recursive Reasoning with Tiny Networks*. This extension keeps that lineage explicit and focuses only on the instruction-tuning additions.

## Documentation Guide

- [`docs/TINYRECURSIVEMODELS_README.md`](docs/TINYRECURSIVEMODELS_README.md) – original project README preserved verbatim for reference.
- [`docs/slimorca.md`](docs/slimorca.md) – decisions and best practices for using the SlimOrca pipeline (dataset loading, training recommendations, inference tips).
- [`docs/instruct_training.md`](docs/instruct_training.md) – step-by-step instruction training playbook and an example SlimOrca run.
- [`AGENTS.md`](AGENTS.md) – running notes from prior automation passes (contains troubleshooting breadcrumbs and historical context).

## Quick Commands

> **Prerequisites:**  
> - Install dependencies from `requirements.txt`.  
> - Place the LLaMA 32k SentencePiece model at `tokenizers/llama-32k/tokenizer.model` (or export `LLAMA_TOKENIZER` to an alternate path).  
> - Set `DISABLE_COMPILE=1` for the instruction workflows; checkpoints are saved without `torch.compile`.
> - Datasets and checkpoints are git-ignored. Follow the download guidance in [`docs/slimorca.md`](docs/slimorca.md#dataset-download--layout) to fetch SlimOrca locally before running the commands below.

### Smoke-Test Training (small slice)

```bash
export LLAMA_TOKENIZER="$PWD/tokenizers/llama-32k/tokenizer.model"
DISABLE_COMPILE=1 \
scripts/pretrain_instruct_smoke.sh \
  dataset.seq_len=512 \
  "$@"
```

The smoke script defaults to a 64-example subset and writes checkpoints under `checkpoints/SlimOrca-ACT/`.

### Full SlimOrca Epoch (single GPU)

```bash
export LLAMA_TOKENIZER="$PWD/tokenizers/llama-32k/tokenizer.model"
DISABLE_COMPILE=1 \
python pretrain_instruct.py \
  dataset.data_dir=data/slimorca_full \
  dataset.subset_size=null \
  dataloader_workers=0 \
  global_batch_size=48 \
  epochs=1
```

This configuration consumes the entire SlimOrca corpus once, assumes a 12 GB-class GPU, and only saves a checkpoint at the end of the epoch.

### Instruction Inference

```bash
DISABLE_COMPILE=1 \
python inference_instruct.py \
  --checkpoint checkpoints/SlimOrca-ACT/<run_name>/step_<N> \
  --split test \
  --save-outputs preds \
  --tokenizer-path tokenizers/llama-32k/tokenizer.model \
  --subset-size 5000
```

Adjust the subset, split, or `dataset.test_ratio` overrides if the held-out partition is empty. Saved tensors land next to the specified checkpoint directory.

### Single Message Generation

```bash
DISABLE_COMPILE=1 \
python generate_instruct.py \
  --checkpoint checkpoints/SlimOrca-ACT/<run_name>/step_<N> \
  --message "Explain how recursion works in simple terms." \
  --tokenizer-path tokenizers/llama-32k/tokenizer.model
```

The script mirrors SlimOrca’s conversation template (system prompt + user + assistant) and prints the generated assistant reply. Use `--overrides` for custom Hydra overrides (e.g., sequence length), `--max-output-tokens` to adjust the length cap (defaults to half of `seq_len`), and `-V/--verbose` to inspect greedy decoding steps with top token probabilities.

> Checkpoints trained prior to the autoregressive shift (assistant tokens masked in the inputs) will output nonsensical generations. Retrain with the current code path before using this helper.

---

For ARC or other original-task workflows, use the commands outlined in the preserved upstream README.

## Dataset Scale & Runtime Estimates

Timings below assume a single Ampere-class GPU with `global_batch_size=48`, `seq_len=512`, `dataloader_workers=0`, and `DISABLE_COMPILE=1`.

| Dataset / Schedule | Approx. Examples | Relationship | 1-Epoch Runtime (h) | GPU-hours |
| --- | --- | --- | --- | --- |
| SlimOrca | ~5.2×10<sup>5</sup> | Curated subset of OpenOrca | ≈4.5 (measured) | ≈4.5 |
| OpenOrca (full) | ~9.7×10<sup>6</sup> | Superset containing SlimOrca | ≈84<sup>†</sup> | ≈84 |
| SlimOrca → OpenOrca (sequential) | ~1.0×10<sup>7</sup> combined | SlimOrca warm-start followed by one OpenOrca epoch | ≈88.5<sup>†</sup> | ≈88.5 |

<sup>†</sup>Estimated by scaling the measured SlimOrca throughput linearly with example count; actual times change with hardware, I/O, or hyperparameter tweaks.

SlimOrca is a high-quality subset of the OpenOrca corpus and is practical for single-GPU experimentation. Full OpenOrca requires multi-day runs unless you distribute training or adopt staged curricula (e.g., SlimOrca pretraining followed by sampled OpenOrca shards).
