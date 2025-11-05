import argparse
import itertools
import os
from typing import Iterable, Optional

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from pretrain import (
    TrainState,
    create_dataloader,
    create_evaluators,
    create_model,
    evaluate,
    load_synced_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tiny Recursive Model inference from a saved checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint produced by pretrain.py (torch.save).",
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs="+",
        default=None,
        help="One or more processed dataset directories (e.g. data/arc1concept-aug-1000).",
    )
    parser.add_argument(
        "--data-paths-test",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of dataset directories to use only for evaluation.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="cfg_pretrain",
        help="Hydra config name to compose (default: cfg_pretrain).",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        default=None,
        help="Additional Hydra style overrides (e.g. arch=trm arch.hidden_size=384).",
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
        help="List of tensor keys to persist (e.g. preds inputs puzzle_identifiers).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on number of batches processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-evaluators",
        action="store_true",
        help="Disable evaluator hooks (e.g. ARC voting metrics).",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Optional project name override (avoids auto-generated names).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override.",
    )
    return parser.parse_args()


def compose_config(args: argparse.Namespace) -> DictConfig:
    overrides = args.overrides or []
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    OmegaConf.set_struct(cfg, False)

    cfg.load_checkpoint = args.checkpoint
    if args.data_paths is not None:
        cfg.data_paths = list(args.data_paths)
    if args.data_paths_test is not None:
        cfg.data_paths_test = list(args.data_paths_test)
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
    config = load_synced_config(hydra_cfg, rank=0, world_size=1)

    # Inference-specific defaults
    config.eval_interval = 1
    config.min_eval_interval = 0
    config.checkpoint_every_eval = False
    config.epochs = max(config.epochs, 1)
    if args.save_outputs is not None:
        config.eval_save_outputs = list(args.save_outputs)
    if args.output_dir is not None:
        config.checkpoint_path = args.output_dir

    if not config.data_paths:
        raise ValueError("At least one --data-paths entry is required for inference.")

    os.makedirs(config.checkpoint_path, exist_ok=True) if config.checkpoint_path else None

    if args.batch_size is not None:
        config.global_batch_size = args.batch_size

    split = args.split
    test_set_mode = split != "train"

    eval_loader, eval_metadata = create_dataloader(
        config,
        split=split,
        rank=0,
        world_size=1,
        test_set_mode=test_set_mode,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )

    # Build model + load checkpoint
    model, optimizers, optimizer_lrs = create_model(config, eval_metadata, rank=0, world_size=1)
    model.eval()

    train_state = TrainState(
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        step=0,
        total_steps=0,
    )

    evaluators = []
    if not args.skip_evaluators:
        try:
            evaluators = create_evaluators(config, eval_metadata)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed to create evaluators ({exc}). Continuing without evaluators.")

    limited_loader = maybe_limit_batches(eval_loader, args.max_batches)

    metrics = evaluate(
        config,
        train_state,
        limited_loader,
        eval_metadata,
        evaluators,
        rank=0,
        world_size=1,
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
    torch.set_float32_matmul_precision("high")
    main()
