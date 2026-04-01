from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class CharTokenizer:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @staticmethod
    def from_text(text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = chars
        return CharTokenizer(stoi=stoi, itos=itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def make_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    data: (N,) int64
    返回 x,y: (B, T)
    """
    n = data.numel()
    if n <= block_size + 1:
        raise ValueError("文本太短，无法构造一个 batch；请提供更长的语料或调小 block_size")

    ix = torch.randint(0, n - block_size - 1, (batch_size,), device=device)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

