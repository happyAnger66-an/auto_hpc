from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..workloads.fmha import FmhaWorkload


@dataclass
class BackendState:
    workload: FmhaWorkload
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    extra: Any | None = None


class Backend(Protocol):
    name: str

    def supported(self, w: FmhaWorkload) -> bool: ...

    def prepare(self, state: BackendState) -> None:
        """可选：JIT 编译、缓存等；可修改 state.extra。"""
        ...

    def run(self, state: BackendState) -> None:
        """将结果写入 state.out。"""
        ...
