from __future__ import annotations

import torch

from .base import BackendState
from ..workloads.fmha import FmhaWorkload


class CuteDSLFmhaBackend:
    name = "cutedsl"

    def supported(self, w: FmhaWorkload) -> bool:
        if w.device != "cuda" or not torch.cuda.is_available():
            return False
        if w.D not in (32, 64, 128, 256):
            return False
        try:
            import cutlass.cute as cute  # noqa: F401
        except ImportError:
            return False
        return True

    def prepare(self, state: BackendState) -> None:
        from optimize_llm.backend.cuteDSL.fmha_cutedsl import fmha_cutedsl_forward

        fmha_cutedsl_forward(
            state.q,
            state.k,
            state.v,
            is_causal=state.workload.is_causal,
            out=state.out,
        )
        torch.cuda.synchronize()

    def run(self, state: BackendState) -> None:
        from optimize_llm.backend.cuteDSL.fmha_cutedsl import fmha_cutedsl_forward

        fmha_cutedsl_forward(
            state.q,
            state.k,
            state.v,
            is_causal=state.workload.is_causal,
            out=state.out,
        )
