from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class BackendRef:
    kind: Literal["builtin", "target"]
    value: str
    alias: str | None = None


@dataclass(frozen=True)
class FmhaCaseSpec:
    B: int = 1
    H: int = 4
    L: int = 64
    D: int = 64
    dtype: str = "fp16"
    causal: bool = False
    seed: int = 0
    device: str = "cuda"


@dataclass(frozen=True)
class RunSpec:
    warmup: int = 5
    iters: int = 20
    check: bool = True
    check_cases: int = 5


@dataclass(frozen=True)
class BenchmarkSpec:
    op: str = "fmha"
    backends: list[BackendRef] = field(default_factory=list)
    cases: list[FmhaCaseSpec] = field(default_factory=lambda: [FmhaCaseSpec()])
    run: RunSpec = field(default_factory=RunSpec)

