import argparse
import logging
import os
from typing import List, Sequence

import torch
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from dataset.common import PuzzleDatasetMetadata
from dataset.slimorca import SYSTEM_PROMPT
from pretrain import TrainState, create_model
from pretrain_instruct import InstructionConfig
from utils.tokenization import (
    ConversationMessage,
    IGNORE_LABEL_ID,
    load_llama_tokenizer,
    tokenize_conversation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a single SlimOrca-style response with a TinyRecursiveModel checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (torch.save) produced by pretrain_instruct.py.",
    )
    parser.add_argument("--message", type=str, required=True, help="User message to respond to.")
    parser.add_argument(
        "--system-message",
        type=str,
        default=SYSTEM_PROMPT,
        help="Optional system prompt. Defaults to SlimOrca's system prompt.",
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
        help="Additional Hydra-style overrides (e.g. arch.hidden_size=768 dataset.seq_len=1024).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Explicit tokenizer.model path (falls back to config/LLAMA_TOKENIZER).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional upper bound on decoded tokens (after assistant prefix).",
    )
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Enable debug logging with per-step token probabilities.",
    )
    return parser.parse_args()


def compose_config(args: argparse.Namespace) -> DictConfig:
    overrides = args.overrides or []
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)

    cfg.load_checkpoint = args.checkpoint
    if args.tokenizer_path is not None:
        cfg.dataset.tokenizer_path = args.tokenizer_path

    return cfg


def _build_metadata(tokenizer, seq_len: int) -> PuzzleDatasetMetadata:
    pad_id = tokenizer.pad_id()
    if pad_id < 0:
        pad_id = tokenizer.eos_id()
    return PuzzleDatasetMetadata(
        pad_id=pad_id,
        ignore_label_id=IGNORE_LABEL_ID,
        blank_identifier_id=0,
        vocab_size=tokenizer.get_piece_size(),
        seq_len=seq_len,
        num_puzzle_identifiers=1,
        total_groups=1,
        mean_puzzle_examples=1.0,
        total_puzzles=1,
        sets=["inference"],
    )


def _decode_tokens(tokenizer, tokens: Sequence[int], start_index: int, max_tokens: int | None) -> str:
    pad_id = tokenizer.pad_id()
    eos_id = tokenizer.eos_id()
    collected: List[int] = []
    for token in tokens[start_index:]:
        if token == pad_id:
            break
        if eos_id >= 0 and token == eos_id:
            break
        collected.append(token)
        if max_tokens is not None and len(collected) >= max_tokens:
            break
    if not collected:
        return ""
    text = tokenizer.decode(collected)
    prefix = "[ASSISTANT]\n"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return text.strip()


def main() -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("DISABLE_COMPILE", "1")

    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    log = logging.getLogger("generate_instruct")

    hydra_cfg = compose_config(args)
    config = InstructionConfig(**hydra_cfg)  # type: ignore[arg-type]
    config.global_batch_size = 1

    tokenizer = load_llama_tokenizer(config.dataset.tokenizer_path)
    metadata = _build_metadata(tokenizer, config.dataset.seq_len)

    messages = [
        ConversationMessage(role="system", content=args.system_message),
        ConversationMessage(role="user", content=args.message),
        ConversationMessage(role="assistant", content=""),
    ]

    input_ids, _labels, attention_mask = tokenize_conversation(
        tokenizer=tokenizer,
        messages=messages,
        seq_len=config.dataset.seq_len,
        pad_token_id=metadata.pad_id,
    )

    input_ids_list = list(input_ids)
    attention_mask_list = list(attention_mask)
    prompt_len = int(sum(attention_mask_list))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels_tensor = torch.full((1, config.dataset.seq_len), IGNORE_LABEL_ID, dtype=torch.long, device=device)

    model, optimizers, optimizer_lrs = create_model(config, metadata, rank=0, world_size=1)
    model.eval()

    train_state = TrainState(
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        step=0,
        total_steps=0,
    )

    generated_tokens = 0
    default_max_tokens = max(1, config.dataset.seq_len // 2)
    requested_max = args.max_output_tokens if args.max_output_tokens is not None else default_max_tokens
    available_space = max(0, config.dataset.seq_len - prompt_len)
    max_steps = min(requested_max, available_space)
    pad_id = metadata.pad_id
    eos_id = tokenizer.eos_id()

    log.debug(
        "Prompt length=%d, available=%d, max_steps=%d (requested=%s)",
        prompt_len,
        available_space,
        max_steps,
        "None" if args.max_output_tokens is None else args.max_output_tokens,
    )

    while generated_tokens < max_steps and prompt_len + generated_tokens < config.dataset.seq_len:
        batch = {
            "inputs": torch.tensor([input_ids_list], dtype=torch.long, device=device),
            "labels": labels_tensor,
            "attention_mask": torch.tensor([attention_mask_list], dtype=torch.long, device=device),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=device),
        }

        final_preds = None
        final_logits = None
        with torch.inference_mode():
            with torch.device(device):
                carry = train_state.model.initial_carry(batch)  # type: ignore[attr-defined]
            while True:
                carry, _, _, outputs, all_finish = train_state.model(  # type: ignore[call-arg]
                    carry=carry, batch=batch, return_keys={"preds", "logits"}
                )
                final_preds = outputs.get("preds")
                final_logits = outputs.get("logits")
                if all_finish:
                    break

        if final_preds is None:
            raise RuntimeError("Model did not return predictions.")

        next_index = prompt_len + generated_tokens
        next_token = int(final_preds[0, next_index].item())

        if final_logits is not None:
            step_logits = final_logits[0, next_index]
            probs = F.softmax(step_logits.to(torch.float32), dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5)
            if args.verbose:
                tokens_readable = [
                    (tokenizer.id_to_piece(int(idx)), float(prob))
                    for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
                ]
                log.debug(
                    "Step %d | next_index=%d | top tokens: %s",
                    generated_tokens + 1,
                    next_index,
                    ", ".join(f"{piece!r}:{prob:.4f}" for piece, prob in tokens_readable),
                )

        if next_token == pad_id or (eos_id >= 0 and next_token == eos_id):
            if args.verbose:
                log.debug("Stopping due to pad/eos token (id=%d).", next_token)
            break

        if args.verbose:
            log.debug("Appended token id=%d piece=%r", next_token, tokenizer.id_to_piece(next_token))

        input_ids_list[next_index] = next_token
        attention_mask_list[next_index] = 1
        generated_tokens += 1

    decoded = _decode_tokens(
        tokenizer,
        input_ids_list,
        start_index=prompt_len,
        max_tokens=generated_tokens if args.max_output_tokens is None else min(generated_tokens, args.max_output_tokens),
    )

    print("=== Generated Response ===")
    print(decoded or "[No tokens generated]")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
