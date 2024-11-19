from collections import defaultdict
import math
from typing import List, Dict, Iterator
from logging import getLogger
import json
import torch

from .tokenizer import Tokenizer
from megalodon.utils import pad

logger = getLogger()


class DataLoader:
    def __init__(
        self,
        tokenizer: Tokenizer,
        path: str,
        batch_size: int,
        world_rank: int,
        world_size: int,
        chunk_size: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.world_rank = world_rank
        self.world_size = world_size
        self.path = path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.jsonl_file = open(path, "r", encoding="utf-8")
        self.line_num = 0

    def pad_to_chunk_size(self, length: int) -> int:
        """Calculate padding length to ensure final length is multiple of chunk_size."""
        if length < self.chunk_size:
            return self.chunk_size
        return math.ceil(length / self.chunk_size) * self.chunk_size

    def build_batch(self, batch: Dict) -> Dict:
        key_padding = {"x": self.tokenizer.pad_id, "y": -100}
        for key, padding in key_padding.items():
            assert key in batch
            batch[key] = self.to_tensor(batch[key], padding)
        return batch

    def to_tensor(self, batch_tokens: List[List[int]], pad_value: int) -> torch.Tensor:
        # First find max length of sequences
        max_len = max([len(t) for t in batch_tokens])
        # Then pad to chunk size requirements
        padded_length = self.pad_to_chunk_size(max_len)

        padded_tokens = [
            pad(x, max_length=padded_length, value=pad_value, truncating="pre")
            for x in batch_tokens
        ]
        return torch.tensor(padded_tokens, dtype=torch.long)

    def __iter__(self) -> Iterator[Dict]:
        batch: Dict[str, List] = defaultdict(list)
        curr_bs = 0
        batch_counter = 0
        for line in self.jsonl_file:
            self.line_num = self.line_num + 1
            if (self.line_num - 1) % self.world_size != self.world_rank:
                continue
            try:
                example = json.loads(line)
                text = example["text"]
                x = self.tokenizer.encode(text, bos=True, eos=True)
                batch["x"].append(x[:-1])
                batch["y"].append(x[1:])
                curr_bs += 1
            except Exception as e:
                logger.error(
                    f"Error when trying to load line {self.line_num} in {self.path}: {e}"
                )

            if curr_bs == self.batch_size:
                batch_counter += 1
                yield self.build_batch(batch)
                batch = defaultdict(list)
                curr_bs = 0
        if curr_bs > 0:
            yield self.build_batch(batch)

    def close(self):
        self.jsonl_file.close()
