# Agent Summary

- Installed CUDA 13.0 compatible nightly wheels for `torch`, `torchvision`, and `torchaudio`, then installed the project requirements and ensured `adam-atan2` was present without cache.
- Created `inference.py`, a CLI that composes Hydra configs, loads checkpoints, builds dataloaders/models, and reuses `evaluate` for single-node inference with optional tensor export and batch limiting.
- Added an "Inference" section to `README.md` documenting the new script, key flags, and usage example.
- Built a small ARC demo dataset (`data/arc-demo`) and saved a random TRM checkpoint (`checkpoints/random_trm.pt`) to smoke-test the inference path; the run is wired but blocked locally by `cudaGetDeviceCount` failing due to lack of GPU access in this environment.
- Ran/troubleshot minimal TRM training on ARC demo data, including resolving puzzle-embedding shape mismatches and ensuring CUDA toolkit (13.0) and drivers are aligned.

# Planning
- Training currently fails on minimal ARC runs because puzzle embeddings were disabled without also zeroing `arch.puzzle_emb_len`. Future configs should either keep puzzle embeddings enabled or set both dimensions to zero to maintain consistent sequence shapes.
- For autoregressive NLP experiments, plan to emit tokenized text in the ARC dataset format but disable puzzle embeddings unless needed for document-level conditioning. Additional work is required to make attention causal (`causal=True` paths) before TRM can serve as a left-to-right language model.

# Instruction-Tuning Expansion
- Objective: extend TinyRecursiveModels beyond ARC puzzles to run supervised instruction tuning (InstructLM) on SlimOrca and similar datasets, starting with small-scale prototypes on a 12 GB RTX 3060.
- Key decisions:
  - Use LLaMA 32k SentencePiece tokenizer for compatibility with SlimOrca and future datasets.
  - Default sequence length 512 for prototyping (configurable to scale up); rely on fixed padding/truncation for stability.
  - Keep TRM’s halting/recursive mechanism active, but adapt attention to causal mode for autoregressive training.
  - Build a new dataset pipeline that loads JSONL data into CPU memory with on-demand tokenization; no multi-stage preprocessing.
  - Maintain existing ARC code paths; add a parallel `pretrain_instruct.py` entry point and supporting configs/modules.
- Scope:
  - Implement SlimOrcaDataset with automated download, conversation templating, masking, and train/test splits.
  - Update TRM model to respect causal masks while retaining recursion.
  - Provide Hydra/config plumbing and documentation for the new instruction-tuning workflow.
  - Maintain `scripts/pretrain_instruct_smoke.sh` as the canonical SlimOrca smoke-test command (defaults can be tweaked via `SMOKE_*` env vars, including `SMOKE_DATA_DIR` for dataset location, plus any extra Hydra overrides).
