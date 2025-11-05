# SlimOrca Instruction Tuning Notes

This repository layer extends TinyRecursiveModels with a SlimOrca-based instruction-tuning workflow. The highlights below capture the decisions we finalized while wiring up end-to-end training and inference.

## Key Entry Points

- Training: `pretrain_instruct.py` (aligned with Hydra config `config/cfg_pretrain_instruct.yaml`).
- Smoke test: `scripts/pretrain_instruct_smoke.sh` (accepts `SMOKE_*` overrides for local debugging).
- Inference: `inference_instruct.py`, which mirrors the training config surface and reuses the shared `evaluate` loop.

## Dataset Download & Layout

- The SlimOrca JSONL (~1 GB) is Apache-2.0 licensed and hosted on Hugging Face: `https://huggingface.co/datasets/Open-Orca/SlimOrca`.
- By default the loader downloads the file on demand into `data/slimorca/SlimOrca.jsonl`. Because `data/` is git-ignored, collaborators must fetch it locally before training.
- For the full corpus you can either:
  1. Let `pretrain_instruct.py` or `inference_instruct.py` run once; the code will download the JSONL automatically if it is missing, or
  2. Manually fetch via `huggingface-cli` / `curl` (optionally the `.zst` variant), then place the decompressed file at `data/slimorca_full/SlimOrca.jsonl`.
- The LLaMA 32k tokenizer (not distributed here) should live at `tokenizers/llama-32k/tokenizer.model` or another path pointed to by `LLAMA_TOKENIZER`.

## Data Loading

- `dataset/slimorca.py` downloads (or reuses) `SlimOrca.jsonl`, tokenizes conversations in-memory with the LLaMA-32k SentencePiece model, and now surfaces a `tqdm` progress bar while tokenizing. The bar always shows a true denominator; we removed previous caching so it recomputes counts per run.
- Set `dataset.data_dir=data/slimorca_full` and `dataset.subset_size=null` to consume the full 518k example corpus. For small-scale tests, override `dataset.subset_size`.
- Use `dataloader_workers=0` when training on a single GPU under WSL/Windows to avoid duplicated dataset copies and SSD thrash.

## Training Defaults and Recommendations

- Always export `LLAMA_TOKENIZER=/path/to/tokenizers/llama-32k/tokenizer.model`.
- Set `DISABLE_COMPILE=1` for now; it keeps PyTorch from wrapping the model in `torch.compile`, matching how checkpoints were produced.
- The default config keeps ACT recursion active. For faster iterations you can reduce `arch.halt_max_steps` or cycles (`arch.L_cycles`, `arch.H_cycles`) before scaling back up.
- With a 12 GB GPU, `global_batch_size=48` (single process) comfortably fits in memory once `dataloader_workers=0` is applied; the full-epoch runtime estimate is roughly 4.5 hours.

## Inference Workflow

- `inference_instruct.py` accepts the same dataset overrides as the trainer. The script defaults `DISABLE_COMPILE=1` to load checkpoints saved without compilation and will emit saved tensors (e.g., `preds`) under the chosen `checkpoint_path`.
- If the test split comes up empty (common with tiny subsets), tweak `--subset-size`, `dataset.test_ratio`, or target the train split.

## Troubleshooting

- When `global_batch_size` exceeds the number of available examples (e.g., a tiny subset), the loader will error with “No train batches available.” Reduce the batch size or increase the subset.
- Watching throughput: the SlimOrca tokenizer bar should progress immediately; once it completes you will see the standard training `tqdm` bar and `W&B` metrics. Lack of output usually means tokenization is still running.
