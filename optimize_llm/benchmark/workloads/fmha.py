from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FmhaWorkload:
    """FMHA 负载：layout [B, H, L, D]。"""

    B: int
    H: int
    L: int
    D: int
    dtype: torch.dtype
    is_causal: bool
    seed: int = 0
    device: str = "cuda"

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)
        q = torch.randn(self.B, self.H, self.L, self.D, device=self.device, dtype=self.dtype, generator=g)
        k = torch.randn(self.B, self.H, self.L, self.D, device=self.device, dtype=self.dtype, generator=g)
        v = torch.randn(self.B, self.H, self.L, self.D, device=self.device, dtype=self.dtype, generator=g)
        return q.contiguous(), k.contiguous(), v.contiguous()
