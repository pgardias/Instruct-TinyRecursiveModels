from __future__ import annotations

import copy
import os
from typing import Iterable, List, Optional, Tuple

import coolname
import hydra
import pydantic
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

import tqdm
import wandb

from dataset.slimorca import SlimOrcaDataset, SlimOrcaDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from models.ema import EMAHelper
from pretrain import ArchConfig, TrainState, create_model, evaluate, save_train_state, train_batch


def _default_wandb_mode():
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"


class InstructionConfig(pydantic.BaseModel):
    arch: ArchConfig
    dataset: SlimOrcaDatasetConfig = SlimOrcaDatasetConfig()

    global_batch_size: int = 32
    epochs: int = 1
    eval_interval: Optional[int] = 1

    lr: float = 1e-4
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 100

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    puzzle_emb_lr: float = 0.0
    puzzle_emb_weight_decay: float = 0.0

    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    checkpoint_every_eval: bool = False
    eval_save_outputs: List[str] = []

    seed: int = 0
    min_eval_interval: int = 0

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False

    dataloader_workers: int = 2


def _ensure_divisible(batch_size: int, world_size: int) -> int:
    if batch_size % world_size != 0:
        raise ValueError(
            f"global_batch_size={batch_size} must be divisible by world_size={world_size}"
        )
    return batch_size // world_size


def _build_dataloader(
    dataset: SlimOrcaDataset,
    config: InstructionConfig,
    world_size: int,
    rank: int,
    *,
    shuffle: bool,
    drop_last: bool,
) -> Tuple[DistributedSampler, DataLoader]:
    local_batch_size = _ensure_divisible(config.global_batch_size, world_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=config.dataloader_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    return sampler, loader


def _wrap_loader(
    loader: DataLoader, world_size: int, set_name: str
) -> Iterable[Tuple[str, dict, int]]:
    for batch in loader:
        global_batch_size = batch["inputs"].shape[0] * world_size
        yield set_name, batch, global_batch_size


def _init_train_state(
    config: InstructionConfig,
    train_metadata: PuzzleDatasetMetadata,
    total_steps: int,
    rank: int,
    world_size: int,
) -> TrainState:
    model, optimizers, optimizer_lrs = create_model(
        config, train_metadata, rank=rank, world_size=world_size
    )
    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )


def _save_code_and_config(config: InstructionConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    config_path = os.path.join(config.checkpoint_path, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(config.model_dump_json(indent=2))
    wandb.run.log_code(config.checkpoint_path)


def _sync_config(
    hydra_config: DictConfig, rank: int, world_size: int
) -> InstructionConfig:
    objects = [None]
    if rank == 0:
        config = InstructionConfig(**hydra_config)  # type: ignore
        if config.project_name is None:
            config.project_name = "SlimOrca-ACT"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(
                "checkpoints", config.project_name, config.run_name
            )
        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain_instruct", version_base=None)
def launch(hydra_config: DictConfig):
    _default_wandb_mode()

    rank = 0
    world_size = 1
    cpu_group = None

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        cpu_group = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(cpu_group) == rank
            and dist.get_world_size(cpu_group) == world_size
        )

    config = _sync_config(hydra_config, rank=rank, world_size=world_size)

    torch.random.manual_seed(config.seed + rank)

    train_dataset = SlimOrcaDataset(config.dataset, split="train")
    train_sampler, train_loader = _build_dataloader(
        train_dataset,
        config,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise RuntimeError(
            "No train batches available. Increase dataset subset_size or reduce global_batch_size."
        )

    try:
        eval_dataset = SlimOrcaDataset(config.dataset, split="test")
        eval_sampler, eval_loader = _build_dataloader(
            eval_dataset,
            config,
            world_size=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    except Exception:
        eval_dataset = None
        eval_sampler = None
        eval_loader = None

    total_steps = config.epochs * steps_per_epoch
    train_state = _init_train_state(
        config, train_dataset.metadata, total_steps, rank=rank, world_size=world_size
    )

    progress_bar = None
    ema_helper = None
    if rank == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )  # type: ignore
        wandb.log(
            {"num_params": sum(x.numel() for x in train_state.model.parameters())},
            step=0,
        )
        _save_code_and_config(config)
    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    eval_interval = config.eval_interval or config.epochs
    if config.epochs % eval_interval != 0:
        raise ValueError("eval_interval must divide total epochs")

    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        if eval_sampler is not None:
            eval_sampler.set_epoch(epoch)

        train_state.model.train()
        if rank == 0:
            print(f"[Rank {rank}, World Size {world_size}]: Epoch {epoch}")

        for _, batch, global_batch_size in _wrap_loader(
            train_loader, world_size, set_name="train"
        ):
            metrics = train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                if progress_bar is not None:
                    progress_bar.update(train_state.step - progress_bar.n)
            if config.ema and ema_helper is not None:
                ema_helper.update(train_state.model)

        should_eval = (
            eval_loader is not None
            and (epoch + 1) >= config.min_eval_interval
            and (epoch + 1) % eval_interval == 0
        )
        if should_eval:
            if rank == 0:
                print("EVALUATE")
            if config.ema and ema_helper is not None:
                eval_state = copy.deepcopy(train_state)
                eval_state.model = ema_helper.ema_copy(eval_state.model)
            else:
                eval_state = train_state
            eval_state.model.eval()
            metrics = evaluate(
                config,
                eval_state,
                _wrap_loader(eval_loader, world_size, set_name="test"),  # type: ignore
                eval_dataset.metadata,  # type: ignore
                evaluators=[],
                rank=rank,
                world_size=world_size,
                cpu_group=cpu_group,
            )
            if rank == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
            if config.checkpoint_every_eval and rank == 0:
                save_train_state(config, eval_state)
            if config.ema and ema_helper is not None:
                del eval_state

    if rank == 0 and not config.checkpoint_every_eval:
        save_train_state(config, train_state)

    if progress_bar is not None:
        progress_bar.close()

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
