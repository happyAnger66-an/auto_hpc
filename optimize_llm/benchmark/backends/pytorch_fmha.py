from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BackendState
from ..workloads.fmha import FmhaWorkload


class PyTorchFmhaBackend:
    """``torch.nn.functional.scaled_dot_product_attention`` 参考实现（正确性与可选计时）。"""

    name = "pytorch"

    def supported(self, w: FmhaWorkload) -> bool:
        return w.device == "cuda" and torch.cuda.is_available()

    def prepare(self, state: BackendState) -> None:
        pass

    def run(self, state: BackendState) -> None:
        o = F.scaled_dot_product_attention(
            state.q,
            state.k,
            state.v,
            is_causal=state.workload.is_causal,
        )
        state.out.copy_(o)
