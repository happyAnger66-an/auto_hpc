from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..workloads.fmha import FmhaWorkload


@dataclass
class BackendContext:
    workload: FmhaWorkload
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    extra: Any | None = None


class BackendPlugin(Protocol):
    name: str

    def supported(self, w: FmhaWorkload) -> bool: ...

    def prepare(self, ctx: BackendContext) -> None: ...

    def run(self, ctx: BackendContext) -> None: ...

