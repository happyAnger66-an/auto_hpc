from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch

from .config import apply_cli_overrides, build_spec_from_cli, load_yaml_spec
from .loader import resolve_backends
from .plugins.base import BackendContext, BackendPlugin
from .spec import BenchmarkSpec, FmhaCaseSpec
from .workloads.fmha import FmhaWorkload


@dataclass
class BackendResult:
    name: str
    skipped: bool
    skip_reason: str | None
    mean_ms: float | None
    median_ms: float | None
    stdev_ms: float | None
    max_abs_err_mean: float | None
    max_abs_err_max: float | None
    mae_mean: float | None
    mse_mean: float | None


def _reference_fmha(w: FmhaWorkload, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q, k, v, is_causal=w.is_causal)


def _error_metrics(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    diff = (out - ref).float()
    abs_diff = diff.abs()
    max_abs = float(abs_diff.max().item())
    mae = float(abs_diff.mean().item())
    mse = float((diff * diff).mean().item())
    return max_abs, mae, mse


def _dtype_from_str(s: str) -> torch.dtype:
    m = s.strip().lower()
    if m in ("fp16", "float16", "f16"):
        return torch.float16
    if m in ("bf16", "bfloat16"):
        return torch.bfloat16
    if m in ("fp32", "float32", "f32"):
        return torch.float32
    raise ValueError(f"未知 dtype: {s!r}，可选 fp16 bf16 fp32")


def _workload_from_case(case: FmhaCaseSpec) -> FmhaWorkload:
    return FmhaWorkload(
        B=case.B,
        H=case.H,
        L=case.L,
        D=case.D,
        dtype=_dtype_from_str(case.dtype),
        is_causal=case.causal,
        seed=case.seed,
        device=case.device,
    )


def run_fmha_benchmark(
    w: FmhaWorkload,
    backends: list[BackendPlugin],
    *,
    warmup: int,
    iters: int,
    check: bool,
    check_cases: int,
) -> tuple[list[BackendResult], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("FMHA benchmark 需要 CUDA。")

    q, k, v = w.materialize()
    meta: dict[str, Any] = {
        "op": "fmha",
        "B": w.B,
        "H": w.H,
        "L": w.L,
        "D": w.D,
        "dtype": str(w.dtype),
        "is_causal": w.is_causal,
        "warmup": warmup,
        "iters": iters,
        "check_cases": check_cases if check else 0,
    }

    results: list[BackendResult] = []
    for b in backends:
        if not b.supported(w):
            results.append(
                BackendResult(
                    name=b.name,
                    skipped=True,
                    skip_reason="unsupported(workload)",
                    mean_ms=None,
                    median_ms=None,
                    stdev_ms=None,
                    max_abs_err_mean=None,
                    max_abs_err_max=None,
                    mae_mean=None,
                    mse_mean=None,
                )
            )
            continue

        out = torch.empty_like(q)
        state = BackendContext(workload=w, q=q, k=k, v=v, out=out)

        try:
            b.prepare(state)
        except Exception as e:  # noqa: BLE001
            results.append(
                BackendResult(
                    name=b.name,
                    skipped=True,
                    skip_reason=f"prepare: {e}",
                    mean_ms=None,
                    median_ms=None,
                    stdev_ms=None,
                    max_abs_err_mean=None,
                    max_abs_err_max=None,
                    mae_mean=None,
                    mse_mean=None,
                )
            )
            continue

        torch.cuda.synchronize()
        for _ in range(max(0, warmup)):
            b.run(state)
            torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(max(1, iters)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            b.run(state)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = statistics.mean(times_ms)
        median_ms = statistics.median(times_ms)
        stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

        max_abs_err_mean: float | None = None
        max_abs_err_max: float | None = None
        mae_mean: float | None = None
        mse_mean: float | None = None
        if check:
            case_max_abs: list[float] = []
            case_mae: list[float] = []
            case_mse: list[float] = []
            for case_idx in range(max(1, check_cases)):
                case_w = FmhaWorkload(
                    B=w.B,
                    H=w.H,
                    L=w.L,
                    D=w.D,
                    dtype=w.dtype,
                    is_causal=w.is_causal,
                    seed=w.seed + 1000 + case_idx,
                    device=w.device,
                )
                cq, ck, cv = case_w.materialize()
                cref = _reference_fmha(case_w, cq, ck, cv)
                state.q, state.k, state.v = cq, ck, cv
                state.out = torch.empty_like(cq)
                b.run(state)
                torch.cuda.synchronize()
                max_abs, mae, mse = _error_metrics(state.out, cref)
                case_max_abs.append(max_abs)
                case_mae.append(mae)
                case_mse.append(mse)
            max_abs_err_mean = statistics.mean(case_max_abs)
            max_abs_err_max = max(case_max_abs)
            mae_mean = statistics.mean(case_mae)
            mse_mean = statistics.mean(case_mse)

        results.append(
            BackendResult(
                name=b.name,
                skipped=False,
                skip_reason=None,
                mean_ms=mean_ms,
                median_ms=median_ms,
                stdev_ms=stdev_ms,
                max_abs_err_mean=max_abs_err_mean,
                max_abs_err_max=max_abs_err_max,
                mae_mean=mae_mean,
                mse_mean=mse_mean,
            )
        )

    return results, meta


def print_report(results: list[BackendResult], meta: dict[str, Any]) -> None:
    print("=== FMHA benchmark ===")
    for k in ("B", "H", "L", "D", "dtype", "is_causal", "warmup", "iters", "check_cases"):
        if k in meta:
            print(f"  {k}: {meta[k]}")
    print()
    hdr = (
        f"{'backend':<10} {'mean_ms':>9} {'median_ms':>9} {'stdev_ms':>9} "
        f"{'max_abs_mean':>13} {'max_abs_max':>12} {'mae_mean':>10} {'mse_mean':>10} {'note':<16}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.skipped:
            print(
                f"{r.name:<10} {'—':>9} {'—':>9} {'—':>9} {'—':>13} {'—':>12} {'—':>10} {'—':>10} "
                f"{(r.skip_reason or ''):<16}"
            )
            continue
        max_abs_mean_s = f"{r.max_abs_err_mean:.4e}" if r.max_abs_err_mean is not None else "—"
        max_abs_max_s = f"{r.max_abs_err_max:.4e}" if r.max_abs_err_max is not None else "—"
        mae_s = f"{r.mae_mean:.4e}" if r.mae_mean is not None else "—"
        mse_s = f"{r.mse_mean:.4e}" if r.mse_mean is not None else "—"
        print(
            f"{r.name:<10} {r.mean_ms:9.4f} {r.median_ms:9.4f} {r.stdev_ms:9.4f} "
            f"{max_abs_mean_s:>13} {max_abs_max_s:>12} {mae_s:>10} {mse_s:>10} {'':<16}"
        )
    print()


def print_summary(all_results: list[tuple[dict[str, Any], list[BackendResult]]]) -> None:
    if len(all_results) <= 1:
        return
    by_backend: dict[str, list[BackendResult]] = {}
    for _, results in all_results:
        for r in results:
            by_backend.setdefault(r.name, []).append(r)

    print("=== Summary Across Cases ===")
    hdr = (
        f"{'backend':<10} {'cases':>7} {'ok_cases':>9} {'avg_mean_ms':>12} "
        f"{'worst_max_abs':>14} {'avg_mae':>10} {'avg_mse':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, rows in by_backend.items():
        ok_rows = [r for r in rows if not r.skipped and r.mean_ms is not None]
        avg_ms = statistics.mean([r.mean_ms for r in ok_rows]) if ok_rows else None
        worst_max_abs = max([r.max_abs_err_max for r in ok_rows if r.max_abs_err_max is not None], default=None)
        avg_mae = statistics.mean([r.mae_mean for r in ok_rows if r.mae_mean is not None]) if ok_rows else None
        avg_mse = statistics.mean([r.mse_mean for r in ok_rows if r.mse_mean is not None]) if ok_rows else None
        avg_ms_s = f"{avg_ms:.4f}" if avg_ms is not None else "—"
        worst_s = f"{worst_max_abs:.4e}" if worst_max_abs is not None else "—"
        mae_s = f"{avg_mae:.4e}" if avg_mae is not None else "—"
        mse_s = f"{avg_mse:.4e}" if avg_mse is not None else "—"
        print(f"{name:<10} {len(rows):7d} {len(ok_rows):9d} {avg_ms_s:>12} {worst_s:>14} {mae_s:>10} {mse_s:>10}")
    print()


def _build_spec(args: Any) -> BenchmarkSpec:
    if args.config:
        spec = load_yaml_spec(args.config)
        return apply_cli_overrides(spec, args)
    return build_spec_from_cli(args)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="统一 FMHA benchmark（可插件化加载 backend）")
    p.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    p.add_argument("--op", default="fmha", choices=["fmha"], help="算子类型（当前仅 fmha）")
    p.add_argument(
        "--backends",
        type=str,
        default=None,
        help="内置 backend，逗号分隔，如 triton,cutedsl,pytorch；all 为全部",
    )
    p.add_argument(
        "--backend-targets",
        type=str,
        default=None,
        help="动态 backend 插件目标，逗号分隔，格式 module.path:Symbol",
    )
    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="追加 case，格式 B=1,H=8,L=128,D=64,dtype=fp16,causal=true,seed=0",
    )
    p.add_argument("--B", type=int, default=None)
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--D", type=int, default=None)
    p.add_argument("--dtype", default=None, help="fp16 bf16 fp32")
    p.add_argument("--device", default=None, help="默认 cuda")
    p.add_argument("--causal", action="store_true", help="因果注意力")
    p.add_argument("--no-causal", action="store_true", help="非因果（默认）")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--check-cases", type=int, default=None, help="正确性校验随机样本组数")
    p.add_argument(
        "--no-check",
        action="store_true",
        help="关闭与 PyTorch SDPA 的误差检查（默认开启）",
    )
    args = p.parse_args(argv)

    try:
        spec = _build_spec(args)
    except Exception as e:  # noqa: BLE001
        print(f"配置解析失败: {e}", flush=True)
        return 2

    if spec.op != "fmha":
        print("当前仅支持 --op fmha", flush=True)
        return 2

    try:
        backends = resolve_backends(spec.op, spec.backends)
    except Exception as e:  # noqa: BLE001
        print(f"backend 加载失败: {e}", flush=True)
        return 2

    all_results: list[tuple[dict[str, Any], list[BackendResult]]] = []
    failures: list[tuple[int, str, float, float]] = []
    for idx, case in enumerate(spec.cases):
        w = _workload_from_case(case)
        results, meta = run_fmha_benchmark(
            w,
            backends,
            warmup=max(0, spec.run.warmup),
            iters=max(1, spec.run.iters),
            check=spec.run.check,
            check_cases=max(1, spec.run.check_cases),
        )
        meta["case_index"] = idx
        print(f"\n--- Case #{idx} ---")
        print_report(results, meta)
        all_results.append((meta, results))

        if spec.run.check:
            tol = 1e-2 if w.dtype == torch.float16 else (2e-2 if w.dtype == torch.bfloat16 else 1e-4)
            for r in results:
                if r.skipped or r.max_abs_err_max is None:
                    continue
                if r.max_abs_err_max > tol:
                    failures.append((idx, r.name, r.max_abs_err_max, tol))

    print_summary(all_results)
    if failures:
        msg = ", ".join([f"case#{cid}:{name}({val:.4e}>{tol:.4e})" for cid, name, val, tol in failures[:10]])
        print(f"[WARN] 误差超阈值 backend: {msg}")
        return 1
    return 0
