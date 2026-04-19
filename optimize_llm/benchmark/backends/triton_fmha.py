from __future__ import annotations

import torch

from .base import BackendState
from ..workloads.fmha import FmhaWorkload


class TritonFmhaBackend:
    name = "triton"

    def supported(self, w: FmhaWorkload) -> bool:
        if w.device != "cuda" or not torch.cuda.is_available():
            return False
        if w.D not in (32, 64, 128, 256):
            return False
        try:
            import triton  # noqa: F401
        except ImportError:
            return False
        return True

    def prepare(self, state: BackendState) -> None:
        from optimize_llm.backend.triton.fmha_triton import fmha_triton_forward

        fmha_triton_forward(
            state.q,
            state.k,
            state.v,
            is_causal=state.workload.is_causal,
            out=state.out,
        )
        torch.cuda.synchronize()

    def run(self, state: BackendState) -> None:
        from optimize_llm.backend.triton.fmha_triton import fmha_triton_forward

        fmha_triton_forward(
            state.q,
            state.k,
            state.v,
            is_causal=state.workload.is_causal,
            out=state.out,
        )
