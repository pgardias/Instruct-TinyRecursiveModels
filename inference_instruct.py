import argparse
import itertools
import os
from typing import Iterable, Optional

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from dataset.slimorca import SlimOrcaDataset
from pretrain import TrainState, create_model, evaluate
from pretrain_instruct import InstructionConfig, _build_dataloader, _wrap_loader

# Ensure compatibility with checkpoints saved without torch.compile.
os.environ.setdefault("DISABLE_COMPILE", "1")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tiny Recursive Model inference on SlimOrca-style instruction data."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint produced by pretrain_instruct.py.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="cfg_pretrain_instruct",
        help="Hydra config name to compose (default: cfg_pretrain_instruct).",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        default=None,
        help="Additional Hydra style overrides (e.g. dataset.seq_len=2048 arch.hidden_size=768).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to run inference on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override global batch size. Defaults to the value in the config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation artifacts (predictions/metrics).",
    )
    parser.add_argument(
        "--save-outputs",
        type=str,
        nargs="*",
        default=None,
        help="List of tensor keys to persist (e.g. preds inputs q_halt_logits).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on number of batches processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override location for SlimOrca data (defaults to data/slimorca).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Explicit path to tokenizer.model if not using LLAMA_TOKENIZER env var.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Optional cap on loaded SlimOrca examples (per split).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override maximum sequence length fed to the model.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Optional project name override saved alongside outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override saved alongside outputs.",
    )
    return parser.parse_args()


def compose_config(args: argparse.Namespace) -> DictConfig:
    overrides = args.overrides or []
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    OmegaConf.set_struct(cfg, False)

    cfg.load_checkpoint = args.checkpoint

    if args.batch_size is not None:
        cfg.global_batch_size = args.batch_size
    if args.output_dir is not None:
        cfg.checkpoint_path = args.output_dir
    if args.save_outputs is not None:
        cfg.eval_save_outputs = list(args.save_outputs)
    if args.project_name is not None:
        cfg.project_name = args.project_name
    if args.run_name is not None:
        cfg.run_name = args.run_name

    if args.data_dir is not None:
        cfg.dataset.data_dir = args.data_dir
    if args.tokenizer_path is not None:
        cfg.dataset.tokenizer_path = args.tokenizer_path
    if args.subset_size is not None:
        cfg.dataset.subset_size = args.subset_size
    if args.seq_len is not None:
        cfg.dataset.seq_len = args.seq_len

    return cfg


def maybe_limit_batches(loader: Iterable, max_batches: Optional[int]) -> Iterable:
    if max_batches is None:
        return loader
    return itertools.islice(loader, max_batches)


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    hydra_cfg = compose_config(args)
    config = InstructionConfig(**hydra_cfg)  # type: ignore[arg-type]

    # Inference-only defaults
    config.eval_interval = 1
    config.min_eval_interval = 0
    config.checkpoint_every_eval = False
    config.epochs = max(config.epochs, 1)

    if args.save_outputs is not None:
        config.eval_save_outputs = list(args.save_outputs)
    if args.output_dir is not None:
        config.checkpoint_path = args.output_dir

    if config.checkpoint_path is not None:
        os.makedirs(config.checkpoint_path, exist_ok=True)

    split = args.split
    rank = 0
    world_size = 1

    dataset = SlimOrcaDataset(config.dataset, split=split)
    sampler, loader = _build_dataloader(
        dataset,
        config,
        world_size=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    if sampler is not None:
        sampler.set_epoch(0)

    # Build model + load checkpoint
    model, optimizers, optimizer_lrs = create_model(config, dataset.metadata, rank=rank, world_size=world_size)
    model.eval()

    train_state = TrainState(
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        step=0,
        total_steps=0,
    )

    limited_loader = maybe_limit_batches(_wrap_loader(loader, world_size, set_name=split), args.max_batches)

    metrics = evaluate(
        config,
        train_state,
        limited_loader,
        dataset.metadata,
        evaluators=[],
        rank=rank,
        world_size=world_size,
        cpu_group=None,
    )

    if metrics:
        print("=== Inference Metrics ===")
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")

    if config.checkpoint_path is not None and config.eval_save_outputs:
        print(f"Saved tensor outputs: {config.checkpoint_path}")


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "disabled")
    torch.set_float32_matmul_precision("high")
    main()
