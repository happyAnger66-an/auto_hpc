from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .spec import BackendRef, BenchmarkSpec, FmhaCaseSpec, RunSpec


def _parse_backend_ref(entry: Any) -> BackendRef:
    if isinstance(entry, str):
        if ":" in entry:
            return BackendRef(kind="target", value=entry)
        return BackendRef(kind="builtin", value=entry)
    if isinstance(entry, dict):
        if "target" in entry:
            return BackendRef(kind="target", value=str(entry["target"]), alias=entry.get("alias"))
        if "builtin" in entry:
            return BackendRef(kind="builtin", value=str(entry["builtin"]), alias=entry.get("alias"))
    raise ValueError(f"非法 backend 声明: {entry!r}")


def _parse_case(data: dict[str, Any]) -> FmhaCaseSpec:
    return FmhaCaseSpec(
        B=int(data.get("B", 1)),
        H=int(data.get("H", 4)),
        L=int(data.get("L", 64)),
        D=int(data.get("D", 64)),
        dtype=str(data.get("dtype", "fp16")),
        causal=bool(data.get("causal", False)),
        seed=int(data.get("seed", 0)),
        device=str(data.get("device", "cuda")),
    )


def parse_case_expr(expr: str) -> FmhaCaseSpec:
    """
    解析 CLI case 表达式，例如:
    B=1,H=8,L=128,D=64,dtype=fp16,causal=true,seed=42
    """
    kv: dict[str, str] = {}
    for part in expr.split(","):
        p = part.strip()
        if not p:
            continue
        if "=" not in p:
            raise ValueError(f"非法 case 项: {p!r}，格式应为 key=value")
        k, v = p.split("=", 1)
        kv[k.strip()] = v.strip()
    if not kv:
        raise ValueError("空 case 表达式")
    data: dict[str, Any] = dict(kv)
    if "causal" in data:
        data["causal"] = str(data["causal"]).lower() in ("1", "true", "yes", "y", "on")
    return _parse_case(data)


def load_yaml_spec(path: str) -> BenchmarkSpec:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML 顶层必须是对象")

    op = str(raw.get("op", "fmha"))
    backend_entries = raw.get("backends", ["all"])
    if not isinstance(backend_entries, list):
        raise ValueError("YAML 字段 backends 必须是列表")
    backends = [_parse_backend_ref(e) for e in backend_entries]

    run_raw = raw.get("run", {}) or {}
    if not isinstance(run_raw, dict):
        raise ValueError("YAML 字段 run 必须是对象")
    run = RunSpec(
        warmup=int(run_raw.get("warmup", 5)),
        iters=int(run_raw.get("iters", 20)),
        check=bool(run_raw.get("check", True)),
        check_cases=int(run_raw.get("check_cases", 5)),
    )

    cases_raw = raw.get("cases")
    if cases_raw is None:
        default_case = raw.get("case", {})
        if not isinstance(default_case, dict):
            raise ValueError("YAML 字段 case 必须是对象")
        cases = [_parse_case(default_case)]
    else:
        if not isinstance(cases_raw, list) or not cases_raw:
            raise ValueError("YAML 字段 cases 必须是非空列表")
        cases = [_parse_case(c) for c in cases_raw]

    return BenchmarkSpec(op=op, backends=backends, cases=cases, run=run)


def apply_cli_overrides(spec: BenchmarkSpec, args: Any) -> BenchmarkSpec:
    run = replace(
        spec.run,
        warmup=int(args.warmup) if args.warmup is not None else spec.run.warmup,
        iters=int(args.iters) if args.iters is not None else spec.run.iters,
        check=(not bool(args.no_check)) if getattr(args, "no_check", False) else spec.run.check,
        check_cases=int(args.check_cases) if args.check_cases is not None else spec.run.check_cases,
    )

    if getattr(args, "no_check", False):
        run = replace(run, check=False)

    if getattr(args, "backends", None):
        names = [x.strip() for x in str(args.backends).split(",") if x.strip()]
        if names:
            spec = replace(
                spec,
                backends=[BackendRef(kind="builtin", value=n) for n in names],
            )

    if getattr(args, "backend_targets", None):
        targets = [x.strip() for x in str(args.backend_targets).split(",") if x.strip()]
        if targets:
            merged = list(spec.backends) + [BackendRef(kind="target", value=t) for t in targets]
            spec = replace(spec, backends=merged)

    cli_cases: list[FmhaCaseSpec] = []
    for expr in getattr(args, "case", []) or []:
        cli_cases.append(parse_case_expr(expr))
    if cli_cases:
        spec = replace(spec, cases=cli_cases)
    else:
        # 单 case 参数覆盖（仅当明确传了某些 case 级参数）
        single_overrides = any(
            getattr(args, k) is not None for k in ("B", "H", "L", "D", "dtype", "seed", "device")
        ) or bool(getattr(args, "causal", False)) or bool(getattr(args, "no_causal", False))
        if single_overrides and spec.cases:
            c0 = spec.cases[0]
            causal = c0.causal
            if getattr(args, "causal", False):
                causal = True
            if getattr(args, "no_causal", False):
                causal = False
            c0 = replace(
                c0,
                B=int(args.B) if args.B is not None else c0.B,
                H=int(args.H) if args.H is not None else c0.H,
                L=int(args.L) if args.L is not None else c0.L,
                D=int(args.D) if args.D is not None else c0.D,
                dtype=str(args.dtype) if args.dtype is not None else c0.dtype,
                seed=int(args.seed) if args.seed is not None else c0.seed,
                device=str(args.device) if args.device is not None else c0.device,
                causal=causal,
            )
            spec = replace(spec, cases=[c0])

    return replace(spec, run=run)


def build_spec_from_cli(args: Any) -> BenchmarkSpec:
    case = FmhaCaseSpec(
        B=int(args.B) if args.B is not None else 1,
        H=int(args.H) if args.H is not None else 4,
        L=int(args.L) if args.L is not None else 64,
        D=int(args.D) if args.D is not None else 64,
        dtype=str(args.dtype) if args.dtype is not None else "fp16",
        causal=bool(args.causal) and not bool(args.no_causal),
        seed=int(args.seed) if args.seed is not None else 0,
        device=str(args.device) if args.device is not None else "cuda",
    )
    backends = [BackendRef(kind="builtin", value=n) for n in str(args.backends or "all").split(",") if n.strip()]
    targets = [x.strip() for x in str(args.backend_targets or "").split(",") if x.strip()]
    for t in targets:
        backends.append(BackendRef(kind="target", value=t))
    run = RunSpec(
        warmup=int(args.warmup) if args.warmup is not None else 5,
        iters=int(args.iters) if args.iters is not None else 20,
        check=not bool(args.no_check),
        check_cases=int(args.check_cases) if args.check_cases is not None else 5,
    )
    cli_cases: list[FmhaCaseSpec] = [parse_case_expr(expr) for expr in (args.case or [])]
    cases = cli_cases if cli_cases else [case]
    return BenchmarkSpec(op=str(args.op or "fmha"), backends=backends, cases=cases, run=run)

