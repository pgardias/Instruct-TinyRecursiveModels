from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Optional
import urllib.error
import urllib.request

import pydantic
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from huggingface_hub import hf_hub_download, list_repo_files
try:
    from huggingface_hub import HfHubHTTPError  # type: ignore
except ImportError:  # huggingface_hub<0.21
    try:
        from huggingface_hub.utils import HfHubHTTPError  # type: ignore
    except ImportError:  # very old versions
        class HfHubHTTPError(Exception):
            pass

from dataset.common import PuzzleDatasetMetadata
from utils.tokenization import (
    ConversationMessage,
    IGNORE_LABEL_ID,
    load_llama_tokenizer,
    tokenize_conversation,
)


SLIMORCA_URL = (
    "https://huggingface.co/datasets/Open-Orca/SlimOrca/resolve/main/SlimOrca.jsonl?download=1"
)
SLIMORCA_FILENAME = "SlimOrca.jsonl"
SYSTEM_PROMPT = "A conversation between a user and a helpful assistant."


class SlimOrcaDatasetConfig(pydantic.BaseModel):
    data_dir: Path = Path("data/slimorca")
    subset_size: Optional[int] = 1000
    seq_len: int = 512
    tokenizer_path: Optional[str] = None
    test_ratio: float = 0.02
    seed: int = 0

    class Config:
        arbitrary_types_allowed = True


class SlimOrcaDataset(Dataset):
    """
    Lightweight in-memory dataset for SlimOrca conversations.
    """

    def __init__(self, config: SlimOrcaDatasetConfig, split: str = "train"):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        self.config = config
        self.split = split

        self.data_dir = Path(self.config.data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.data_dir / SLIMORCA_FILENAME

        self._ensure_dataset()

        self.tokenizer = load_llama_tokenizer(self.config.tokenizer_path)
        self.pad_id = self.tokenizer.pad_id()
        if self.pad_id < 0:
            self.pad_id = self.tokenizer.eos_id()

        records = self._load_split_records()
        total = self._progress_total_estimate()
        record_iter = tqdm(
            records,
            total=total,
            desc=f"SlimOrca[{self.split}]",
            unit="ex",
            disable=os.environ.get("SLIMORCA_TQDM", "").lower() in {"0", "false"},
        )
        samples = []
        skipped = 0
        for record in record_iter:
            sample = self._encode_record(record)
            if sample is None:
                skipped += 1
                continue
            samples.append(sample)
        record_iter.close()

        if not samples:
            raise RuntimeError(
                f"No SlimOrca records available for split='{split}'. "
                "Try increasing subset_size or adjusting test_ratio."
            )

        if skipped and split == "train":
            print(f"[SlimOrcaDataset] Skipped {skipped} record(s) lacking user/assistant messages.")

        self.samples = samples

        total_examples = len(self.samples)
        self.metadata = PuzzleDatasetMetadata(
            pad_id=self.pad_id,
            ignore_label_id=IGNORE_LABEL_ID,
            vocab_size=self.tokenizer.get_piece_size(),
            seq_len=self.config.seq_len,
            num_puzzle_identifiers=1,
            blank_identifier_id=0,
            total_groups=total_examples,
            mean_puzzle_examples=1.0,
            total_puzzles=total_examples,
            sets=[f"{self.split}"],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return {
            "inputs": sample["inputs"],
            "labels": sample["labels"],
            "attention_mask": sample["attention_mask"],
            "puzzle_identifiers": sample["puzzle_identifiers"],
        }

    def _ensure_dataset(self) -> None:
        if self.dataset_path.exists():
            return

        try:
            with urllib.request.urlopen(SLIMORCA_URL) as response, open(
                self.dataset_path, "wb"
            ) as dest:
                shutil.copyfileobj(response, dest)
            return
        except urllib.error.URLError:
            pass

        self._download_from_hf_hub()

    def _download_from_hf_hub(self) -> None:
        filenames = [
            "SlimOrca.jsonl",
            "SlimOrca.jsonl.zst",
            "data/SlimOrca.jsonl",
            "data/SlimOrca.jsonl.zst",
        ]
        try:
            repo_files = list_repo_files("Open-Orca/SlimOrca", repo_type="dataset")
            dynamic_candidates = [
                name
                for name in repo_files
                if name.lower().endswith(".jsonl") or name.lower().endswith(".jsonl.zst")
            ]
            filenames = dynamic_candidates + [fn for fn in filenames if fn not in dynamic_candidates]
        except Exception:
            pass
        last_error: Optional[Exception] = None
        for filename in filenames:
            try:
                downloaded_path = hf_hub_download(
                    repo_id="Open-Orca/SlimOrca",
                    filename=filename,
                    repo_type="dataset",
                )
            except HfHubHTTPError as exc:
                last_error = exc
                continue

            src = Path(downloaded_path)
            if filename.endswith(".zst"):
                try:
                    import zstandard as zstd  # type: ignore
                except ImportError as exc:
                    raise RuntimeError(
                        "Downloaded SlimOrca JSONL is Zstandard-compressed. Install the 'zstandard' package "
                        "or provide an uncompressed file manually."
                    ) from exc

                with src.open("rb") as compressed, self.dataset_path.open("wb") as dest:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(compressed) as reader:
                        shutil.copyfileobj(reader, dest)
            else:
                shutil.copy(src, self.dataset_path)
            return

        raise RuntimeError(
            "Failed to download SlimOrca dataset from Hugging Face. "
            "Tried files: SlimOrca.jsonl, SlimOrca.jsonl.zst"
        ) from last_error

    def _load_split_records(self) -> Iterable[dict]:
        subset_limit = self.config.subset_size
        collected = 0

        with open(self.dataset_path, "r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)

                if self._record_in_split(idx):
                    yield record
                    collected += 1
                    if subset_limit is not None and collected >= subset_limit:
                        return

    def _record_in_split(self, idx: int) -> bool:
        rng = self._split_rng(idx)
        is_test = rng.random() < self.config.test_ratio
        return is_test if self.split == "test" else not is_test

    def _split_rng(self, idx: int) -> random.Random:
        return random.Random((self.config.seed << 32) ^ idx)

    def _encode_record(self, record: dict):
        conversation = record.get("conversations")
        if isinstance(conversation, list) and conversation:
            messages = self._messages_from_conversation(conversation)
            if messages is None:
                return None
        else:
            user_prompt = self._extract_field(record, ("prompt", "question", "input"))
            assistant_response = self._extract_field(record, ("response", "output", "answer"))
            messages = [
                ConversationMessage(role="system", content=SYSTEM_PROMPT),
                ConversationMessage(role="user", content=user_prompt),
                ConversationMessage(role="assistant", content=assistant_response),
            ]

        input_ids, labels, attention_mask = tokenize_conversation(
            tokenizer=self.tokenizer,
            messages=messages,
            seq_len=self.config.seq_len,
            pad_token_id=self.pad_id,
        )

        # Shift assistant tokens so the model predicts them autoregressively.
        original_ids = list(input_ids)
        input_ids = list(input_ids)
        labels = list(labels)
        attention_mask = list(attention_mask)
        for idx, label in enumerate(labels):
            if label == IGNORE_LABEL_ID:
                continue
            if idx == 0:
                continue
            input_ids[idx] = original_ids[idx - 1]

        return {
            "inputs": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "puzzle_identifiers": torch.tensor(0, dtype=torch.long),
        }

    def _messages_from_conversation(self, conversation: Iterable[dict]) -> Optional[List[ConversationMessage]]:
        system_msg = None
        user_parts: List[str] = []
        assistant_parts: List[str] = []

        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role = (turn.get("from") or turn.get("role") or "").strip().lower()
            text = (turn.get("value") or turn.get("content") or "").strip()
            if not text:
                continue
            if role == "system" and system_msg is None:
                system_msg = text
            elif role in ("human", "user"):
                user_parts.append(text)
            elif role in ("assistant", "gpt", "bot"):
                assistant_parts.append(text)

        if not user_parts or not assistant_parts:
            return None
        if system_msg is None:
            system_msg = SYSTEM_PROMPT

        return [
            ConversationMessage(role="system", content=system_msg),
            ConversationMessage(role="user", content="\n\n".join(user_parts)),
            ConversationMessage(role="assistant", content="\n\n".join(assistant_parts)),
        ]

    @staticmethod
    def _extract_field(record: dict, candidates: Iterable[str]) -> str:
        for key in candidates:
            if key in record and isinstance(record[key], str):
                value = record[key].strip()
                if value:
                    return value
        raise KeyError(f"Record missing required text fields: tried {candidates}")

    def _progress_total_estimate(self) -> Optional[int]:
        if self.config.subset_size is not None:
            expected = self.config.subset_size
            test_examples = int(round(expected * self.config.test_ratio))
            if self.split == "test":
                return max(test_examples, 0)
            return max(expected - test_examples, 0)
        return self._compute_full_split_count()

    def _compute_full_split_count(self) -> Optional[int]:
        counts = {"train": 0, "test": 0}
        with open(self.dataset_path, "r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                split_name = "test" if self._record_in_split(idx) else "train"
                counts[split_name] += 1

        return counts.get(self.split)


__all__ = ["SlimOrcaDataset", "SlimOrcaDatasetConfig", "SLIMORCA_URL"]
