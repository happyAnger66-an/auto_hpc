from __future__ import annotations

from .backends.base import Backend
from .backends.cutedsl_fmha import CuteDSLFmhaBackend
from .backends.pytorch_fmha import PyTorchFmhaBackend
from .backends.triton_fmha import TritonFmhaBackend

# op 名称 -> 可用 backend 实例（顺序即默认展示顺序）
FMHA_BACKENDS: list[Backend] = [
    PyTorchFmhaBackend(),
    TritonFmhaBackend(),
    CuteDSLFmhaBackend(),
]

REGISTRY: dict[str, list[Backend]] = {
    "fmha": FMHA_BACKENDS,
}


def list_backends(op: str) -> list[Backend]:
    if op not in REGISTRY:
        raise KeyError(f"未知 op: {op!r}，已知: {list(REGISTRY.keys())}")
    return REGISTRY[op]


def resolve_backend_names(op: str, names: list[str]) -> list[Backend]:
    """按名称解析 backend；``all`` 表示该 op 下全部实例。"""
    all_b = list_backends(op)
    by_name: dict[str, Backend] = {b.name: b for b in all_b}
    if len(names) == 1 and names[0].lower() == "all":
        return list(all_b)
    out: list[Backend] = []
    seen: set[str] = set()
    for n in names:
        key = n.strip().lower()
        if key not in by_name:
            raise KeyError(f"未知 backend: {n!r}，可选: {sorted(by_name)} 或 all")
        if key in seen:
            continue
        seen.add(key)
        out.append(by_name[key])
    return out
