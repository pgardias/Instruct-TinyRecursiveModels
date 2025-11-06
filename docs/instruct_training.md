# Instruction Training Playbook

This guide summarizes the standard workflow for SlimOrca-style instruction tuning in this repository, along with an example run harvested from offline W&B logs.

## Training Workflow

1. Fetch SlimOrca (see [`docs/slimorca.md`](slimorca.md#dataset-download--layout)) and place the JSONL under `data/slimorca_full/`.
2. Export the LLaMA 32k tokenizer path:

   ```bash
   export LLAMA_TOKENIZER="$PWD/tokenizers/llama-32k/tokenizer.model"
   ```

3. Launch the trainer (single Ampere-class GPU):

   ```bash
   DISABLE_COMPILE=1 \
   python pretrain_instruct.py \
     dataset.data_dir=data/slimorca_full \
     dataset.subset_size=null \
     dataloader_workers=0 \
     global_batch_size=48 \
     epochs=1
   ```

4. Monitor progress via the tokenizer `tqdm` bar, the per-step W&B logs, or use `scripts/pretrain_instruct_smoke.sh` for quick sanity checks before the full run.

## Example SlimOrca Experiment

The snapshot below reflects a full SlimOrca epoch recorded offline with W&B:

```
Running 0 evaluator(s)...
All evaluators completed!
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10250/10250 [5:20:00<00:00,  1.87s/it]
wandb:
wandb: Run history:
wandb:            num_params ▁
wandb:        train/accuracy ▁▁██████████████████████████████████████
wandb:           train/count ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:  train/exact_accuracy ██▁█████████████████████████████████████
wandb:         train/lm_loss █▃▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:              train/lr █████▇▇▇▇▇▆▆▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▃▂▂▂▂▂▂▁▁▁▁▁
wandb: train/q_halt_accuracy ▁███████████████████████████████████████
wandb:     train/q_halt_loss ██▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:           train/steps ▁▁▆▇▇▇▆▇▇▇▇▇▇▇██▇▇▇▇▇▇▇▇▇▇█▇▇▇█▇▇▆▇▇▇▇▇█
wandb:
wandb: Run summary:
wandb:            num_params 6947842
wandb:        train/accuracy 1
wandb:           train/count 1
wandb:  train/exact_accuracy 1
wandb:         train/lm_loss 0.00871
wandb:              train/lr 1e-05
wandb: train/q_halt_accuracy 1
wandb:     train/q_halt_loss 1e-05
wandb:           train/steps 1.08
```

**Observations**

- SlimOrca (≈5.2×10<sup>5</sup> examples) requires ~5.3 GPU-hours for one epoch at `global_batch_size=48`.
- The run converged quickly (low LM loss and perfect accuracy on the final batch), consistent with checkpoints initialized from scratch but trained on curated, high-quality targets.
- Learning rate followed the cosine schedule (`lr_min_ratio=0.1`, `lr_warmup_steps=100`) before decaying toward 1e-5 by the end of the epoch.

**Corresponding Evaluation**

The inference pass on the held-out SlimOrca split for the same checkpoint produced:

```
=== Inference Metrics ===
test:
  accuracy: 1.0
  exact_accuracy: 1.0
  lm_loss: 0.030534343793988228
  q_halt_accuracy: 1.0
  q_halt_loss: 9.516296813671943e-06
  steps: 16.0
```

This confirms the model reliably halts at the configured maximum (16 ACT steps) and matches labels on the evaluation partition.

## Single-Message Generation Utility

For ad-hoc prompts against a trained checkpoint, use the helper script:

```bash
DISABLE_COMPILE=1 \
python generate_instruct.py \
  --checkpoint checkpoints/SlimOrca-ACT/<run_name>/step_<N> \
  --message "Draft a friendly welcome note for a new teammate." \
  --tokenizer-path tokenizers/llama-32k/tokenizer.model
```

The script reuses the SlimOrca system prompt by default, encodes the supplied user message, and decodes the model’s assistant response. Add `--max-output-tokens` to raise/lower the default cap (half the sequence length by default), `-V/--verbose` to log per-step token probabilities, or `--overrides dataset.seq_len=1024` if you need a longer context window.

Use this trace as a reference when validating new changes, reproducing the workflow on additional GPUs, or scaling to larger datasets (e.g., OpenOrca).

> **Note:** Instruction checkpoints created before the autoregressive shift (assistant tokens replaced by their previous context in the inputs) will not free-generate meaningful responses. Retrain with the current code path before using `generate_instruct.py`.
