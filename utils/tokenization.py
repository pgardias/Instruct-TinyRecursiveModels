from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import sentencepiece as spm

IGNORE_LABEL_ID = -100

DEFAULT_LLAMA_TOKENIZER_PATH = Path("tokenizers/llama-32k/tokenizer.model")


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str


class TokenizerLoaderError(RuntimeError):
    pass


def load_llama_tokenizer(tokenizer_path: Optional[str] = None) -> spm.SentencePieceProcessor:
    """
    Loads a SentencePiece tokenizer trained on the 32k LLaMA vocabulary.

    Parameters
    ----------
    tokenizer_path:
        Optional path to the SentencePiece model. Falls back to
        ``tokenizers/llama-32k/tokenizer.model`` relative to the project root.
    """
    resolved_path = Path(tokenizer_path) if tokenizer_path is not None else DEFAULT_LLAMA_TOKENIZER_PATH
    resolved_path = resolved_path.expanduser().resolve()

    if not resolved_path.exists():
        raise TokenizerLoaderError(
            f"LLaMA tokenizer model not found at {resolved_path}. "
            "Provide a valid --tokenizer_path pointing to a SentencePiece .model file."
        )

    processor = spm.SentencePieceProcessor()
    if not processor.load(str(resolved_path)):
        raise TokenizerLoaderError(f"Failed to load SentencePiece model from {resolved_path}.")
    return processor


def _message_prefix(role: str) -> str:
    role = role.lower()
    if role == "system":
        return "[SYSTEM]\n"
    if role == "user":
        return "[USER]\n"
    if role == "assistant":
        return "[ASSISTANT]\n"
    return f"[{role.upper()}]\n"


def tokenize_conversation(
    tokenizer: spm.SentencePieceProcessor,
    messages: Sequence[ConversationMessage],
    seq_len: int,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
    pad_token_id: Optional[int] = None,
    assistant_roles: Iterable[str] = ("assistant",),
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Tokenize conversation messages into fixed-length inputs and labels.

    Returns
    -------
    Tuple of (input_ids, labels, attention_mask).
    """
    assistant_roles = {role.lower() for role in assistant_roles}
    pad_token_id = tokenizer.pad_id() if pad_token_id is None else pad_token_id
    if pad_token_id < 0:
        pad_token_id = tokenizer.eos_id()

    input_ids: List[int] = []
    labels: List[int] = []
    attention_mask: List[int] = []

    for idx, message in enumerate(messages):
        prefix = _message_prefix(message.role)
        content = message.content.strip()
        suffix = "" if message.role.lower() == "assistant" else "\n"
        text = f"{prefix}{content}{suffix}"

        add_bos_flag = add_bos and not input_ids
        add_eos_flag = add_eos and (idx == len(messages) - 1)

        tokens = tokenizer.encode(text, out_type=int, add_bos=add_bos_flag, add_eos=add_eos_flag)

        input_ids.extend(tokens)

        role_is_assistant = message.role.lower() in assistant_roles
        if role_is_assistant:
            segment_labels = list(tokens)
            if add_bos_flag and segment_labels:
                segment_labels[0] = ignore_label_id
        else:
            segment_labels = [ignore_label_id] * len(tokens)

        labels.extend(segment_labels)

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        labels = labels[:seq_len]

    attention_mask = [1] * len(input_ids)

    if len(input_ids) < seq_len:
        pad_length = seq_len - len(input_ids)
        input_ids.extend([pad_token_id] * pad_length)
        labels.extend([ignore_label_id] * pad_length)
        attention_mask.extend([0] * pad_length)

    return input_ids, labels, attention_mask


__all__ = [
    "ConversationMessage",
    "DEFAULT_LLAMA_TOKENIZER_PATH",
    "load_llama_tokenizer",
    "tokenize_conversation",
    "IGNORE_LABEL_ID",
]
