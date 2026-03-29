#!/usr/bin/env python3
"""
原生 CUDA（16B cp.async 预取 + 双槽 smem）Linear Y = X @ W.T + b，与 cutedsl 使用相同 pad 规则。

首次运行会 JIT 编译扩展（需 nvcc、与当前 torch 匹配的 CUDA）。

用法:
  cd auto_hpc/linear/cuda_cpasync_linear && python3 benchmark_cpasync_linear.py --m 1024 --n 1024 --k 1024
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_root = Path(__file__).resolve().parent
_cutedsl = _root.parent / "cutedsl"
if str(_cutedsl) not in sys.path:
    sys.path.insert(0, str(_cutedsl))

from benchmark_linear import BM, BN, BK, pad_linear_tensors  # noqa: E402

_cpasync_mod = None
_cpasync_cache_key: tuple[str, str] | None = None


def get_cpasync_module():
    """按 CUDA 架构与是否仅同步 float4 加载区分缓存，避免环境变量切换后仍用旧 .so。"""
    global _cpasync_mod, _cpasync_cache_key
    from torch.utils.cpp_extension import load

    arch = os.environ.get("CUDA_CPASYNC_ARCH", "89")
    sync_flag = os.environ.get("CPASYNC_LINEAR_USE_SYNC_LOAD", "")
    key = (arch, sync_flag)
    if _cpasync_mod is not None and _cpasync_cache_key == key:
        return _cpasync_mod

    gencode = f"-gencode=arch=compute_{arch},code=sm_{arch}"
    cuda_cflags = ["-O3", "--use_fast_math", gencode]
    if sync_flag == "1":
        cuda_cflags.append("-DCPASYNC_LINEAR_USE_SYNC_LOAD=1")
    ext_name = "cpasync_linear_cuda_sync" if sync_flag == "1" else "cpasync_linear_cuda"
    _cpasync_mod = load(
        name=ext_name,
        sources=[str(_root / "cpasync_tiled_linear.cu")],
        extra_cuda_cflags=cuda_cflags,
        extra_cflags=["-std=c++17"],
        verbose=os.environ.get("VERBOSE_CPASYNC_BUILD", "") == "1",
    )
    _cpasync_cache_key = key
    return _cpasync_mod


def flops_linear(m: int, n: int, k: int) -> float:
    return float(2 * m * n * k + m * n)


def main() -> None:
    p = argparse.ArgumentParser(description="CUDA cp.async tiled linear vs torch")
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--machine", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)

    m, n, k = args.m, args.n, args.k
    x = torch.randn(m, k, device="cuda", dtype=torch.float32)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    x_pad, wt_pad, b_pad, _y_pad_unused, mo, no, ko = pad_linear_tensors(x, w, b)
    assert (mo, no, ko) == (m, n, k)

    mod = get_cpasync_module()
    t0 = time.perf_counter()
    y_out = mod.forward(x_pad, wt_pad, b_pad)
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0

    ref = torch.nn.functional.linear(x, w, b)
    torch.testing.assert_close(y_out[:m, :n], ref, rtol=1e-4, atol=1e-3)

    for _ in range(args.warmup):
        y_out = mod.forward(x_pad, wt_pad, b_pad)
    torch.cuda.synchronize()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(args.iters):
        y_out = mod.forward(x_pad, wt_pad, b_pad)
    ev1.record()
    torch.cuda.synchronize()
    ms_total = ev0.elapsed_time(ev1)
    ms_per_iter = ms_total / args.iters

    f = flops_linear(m, n, k)
    gflops = (f / (ms_per_iter / 1000.0)) / 1e9

    if args.machine:
        print(
            f"cpasync_cuda_ms_per_iter {ms_per_iter:.6f} cpasync_cuda_gflops {gflops:.4f} "
            f"cpasync_cuda_compile_s {compile_s:.6f}"
        )
    else:
        mode = (
            "sync_float4_debug"
            if os.environ.get("CPASYNC_LINEAR_USE_SYNC_LOAD", "") == "1"
            else "cp_async_16B_prefetch"
        )
        print(
            f"shape M={m} N={n} K={k}  fp32  cuda_cpasync_linear  mode={mode}  "
            f"(BM={BM} BN={BN} BK={BK})"
        )
        print(f"first_run_includes_jit_s={compile_s:.4f}  ms/iter={ms_per_iter:.6f}  GFLOPS={gflops:.2f}")
        print(f"FLOPs/iter={f:.0f}  (2*M*N*K + M*N)")


if __name__ == "__main__":
    main()
