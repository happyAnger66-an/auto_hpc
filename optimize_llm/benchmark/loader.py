from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

from .plugins.base import BackendPlugin
from .registry import list_backends
from .spec import BackendRef


@dataclass
class AliasedBackend:
    base: BackendPlugin
    alias: str

    @property
    def name(self) -> str:
        return self.alias

    def supported(self, w):  # noqa: ANN001
        return self.base.supported(w)

    def prepare(self, ctx):  # noqa: ANN001
        return self.base.prepare(ctx)

    def run(self, ctx):  # noqa: ANN001
        return self.base.run(ctx)


def _validate_backend_plugin(obj: Any, source: str) -> BackendPlugin:
    for attr in ("name", "supported", "prepare", "run"):
        if not hasattr(obj, attr):
            raise TypeError(f"插件 {source} 缺少字段/方法: {attr}")
    return obj


def _instantiate_symbol(symbol: Any, source: str) -> BackendPlugin:
    if inspect.isclass(symbol):
        inst = symbol()
        return _validate_backend_plugin(inst, source)
    if callable(symbol) and not hasattr(symbol, "run"):
        inst = symbol()
        return _validate_backend_plugin(inst, source)
    return _validate_backend_plugin(symbol, source)


def load_backend_target(target: str) -> BackendPlugin:
    if ":" not in target:
        raise ValueError(f"target 必须为 module.path:Symbol 形式，当前: {target!r}")
    module_name, symbol_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, symbol_name):
        raise AttributeError(f"模块 {module_name!r} 不包含符号 {symbol_name!r}")
    symbol = getattr(module, symbol_name)
    return _instantiate_symbol(symbol, source=target)


def resolve_backends(op: str, refs: list[BackendRef]) -> list[BackendPlugin]:
    builtins = {b.name: b for b in list_backends(op)}
    out: list[BackendPlugin] = []
    seen: set[str] = set()

    for ref in refs:
        if ref.kind == "builtin":
            key = ref.value.strip().lower()
            if key == "all":
                for b in builtins.values():
                    nm = ref.alias or b.name
                    if nm in seen:
                        continue
                    seen.add(nm)
                    out.append(AliasedBackend(b, nm) if ref.alias else b)
                continue
            if key not in builtins:
                raise KeyError(f"未知 builtin backend: {ref.value!r}，可选: {sorted(builtins)} 或 all")
            b = builtins[key]
            nm = ref.alias or b.name
            if nm in seen:
                continue
            seen.add(nm)
            out.append(AliasedBackend(b, nm) if ref.alias else b)
            continue

        if ref.kind == "target":
            plugin = load_backend_target(ref.value)
            nm = ref.alias or plugin.name
            if nm in seen:
                continue
            seen.add(nm)
            out.append(AliasedBackend(plugin, nm) if ref.alias else plugin)
            continue

        raise ValueError(f"未知 backend ref kind: {ref.kind!r}")

    return out

