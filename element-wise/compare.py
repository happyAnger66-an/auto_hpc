#!/usr/bin/env python3
"""
统一对比 FP32 element-wise add（C = A + B）三种实现：CuTeDSL、cuBLAS、cuDNN。

性能指标（三者一致）：
  GFLOPS = (M * N) / time_seconds / 1e9
  其中按每个输出元素 1 次浮点加法计 1 FLOP。

依赖：
  - CuTeDSL：已安装 cutlass DSL，本目录 cutedsl/benchmark_layouts.py
  - cuBLAS / cuDNN：在子目录内 cmake 编译出 elementwise_add_bench

示例:
  cmake -S element-wise/cublas -B element-wise/cublas/build -DCMAKE_BUILD_TYPE=Release && cmake --build element-wise/cublas/build
  cmake -S element-wise/cudnn -B element-wise/cudnn/build -DCMAKE_BUILD_TYPE=Release && cmake --build element-wise/cudnn/build
  python3 element-wise/compare.py --m 4096 --n 4096
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_MS_GFLOPS = re.compile(
    r"ms_per_iter\s+([\d.]+)\s+gflops\s+([\d.]+)", re.IGNORECASE
)
_CUTEDSL = re.compile(
    r"cutedsl_ms_per_iter\s+([\d.]+)\s+cutedsl_gflops\s+([\d.]+)\s+cutedsl_compile_s\s+([\d.]+)"
)


def run_cutedsl(
    m: int, n: int, warmup: int, iters: int, layout_id: int
) -> tuple[float, float, float] | None:
    script = ROOT / "cutedsl" / "benchmark_layouts.py"
    if not script.is_file():
        print(f"[skip] 未找到 {script}", file=sys.stderr)
        return None
    cmd = [
        sys.executable,
        str(script),
        "--m",
        str(m),
        "--n",
        str(n),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--only-id",
        str(layout_id),
        "--machine",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[CuTeDSL failed] {r.stderr or r.stdout}", file=sys.stderr)
        return None
    line = (r.stdout or "").strip().split("\n")[-1]
    m2 = _CUTEDSL.search(line)
    if not m2:
        print(f"[CuTeDSL parse fail] {line!r}", file=sys.stderr)
        return None
    return float(m2.group(1)), float(m2.group(2)), float(m2.group(3))


def run_native(
    exe: Path, m: int, n: int, warmup: int, iters: int
) -> tuple[float, float] | None:
    if not exe.is_file():
        return None
    r = subprocess.run(
        [str(exe), str(m), str(n), str(warmup), str(iters)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"[{exe.name} failed] {r.stderr}", file=sys.stderr)
        return None
    text = (r.stdout or "").strip()
    for line in text.split("\n"):
        m2 = _MS_GFLOPS.search(line)
        if m2:
            return float(m2.group(1)), float(m2.group(2))
    print(f"[parse fail {exe}] {text!r}", file=sys.stderr)
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare elementwise add: CuTeDSL vs cuBLAS vs cuDNN (GFLOPS)"
    )
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument(
        "--layout-id",
        type=int,
        default=0,
        help="CuTeDSL benchmark_layouts 的 layout_id（默认 0）",
    )
    p.add_argument(
        "--cublas-exe",
        type=Path,
        default=ROOT / "cublas" / "build" / "elementwise_add_bench",
    )
    p.add_argument(
        "--cudnn-exe",
        type=Path,
        default=ROOT / "cudnn" / "build" / "elementwise_add_bench",
    )
    args = p.parse_args()

    print(
        f"问题规模 (M,N)=({args.m},{args.n})  fp32  warmup={args.warmup} iters={args.iters}\n"
        f"指标: GFLOPS = (M×N) / time_s / 1e9（每元素 1 次加法 = 1 FLOP）\n"
    )

    rows: list[tuple[str, str, float | None, float | None, str]] = []

    cd = run_cutedsl(args.m, args.n, args.warmup, args.iters, args.layout_id)
    if cd:
        rows.append(
            (
                "CuTeDSL",
                f"layout_id={args.layout_id}",
                cd[0],
                cd[1],
                f"compile_s={cd[2]:.3f}（仅首次）",
            )
        )
    else:
        rows.append(("CuTeDSL", f"layout_id={args.layout_id}", None, None, "失败或跳过"))

    cb = run_native(args.cublas_exe, args.m, args.n, args.warmup, args.iters)
    if cb:
        rows.append(("cuBLAS", "Scopy+Saxpy", cb[0], cb[1], "两调用/迭代"))
    else:
        rows.append(("cuBLAS", "Scopy+Saxpy", None, None, f"未找到: {args.cublas_exe}"))

    dn = run_native(args.cudnn_exe, args.m, args.n, args.warmup, args.iters)
    if dn:
        rows.append(("cuDNN", "OpTensor ADD", dn[0], dn[1], "单调用/迭代"))
    else:
        rows.append(("cuDNN", "OpTensor ADD", None, None, f"未找到: {args.cudnn_exe}"))

    hdr = f"{'backend':<12} {'impl':<22} {'ms/iter':>12} {'GFLOPS':>12}  note"
    print(hdr)
    print("-" * len(hdr))
    best_gf = None
    best_name = ""
    for name, impl, ms, gflops, note in rows:
        if ms is not None and gflops is not None:
            if best_gf is None or gflops > best_gf:
                best_gf = gflops
                best_name = name
            print(f"{name:<12} {impl:<22} {ms:>12.4f} {gflops:>12.2f}  {note}")
        else:
            print(f"{name:<12} {impl:<22} {'-':>12} {'-':>12}  {note}")
    print("-" * len(hdr))
    if best_gf is not None:
        print(f"最高 GFLOPS: {best_name} @ {best_gf:.2f} GFLOPS")

    print(
        "\n说明: cuBLAS 为 Scopy+Saxpy 两调用/迭代，与单 kernel 的 CuTeDSL/cuDNN 形态不同，仅作参考。"
    )


if __name__ == "__main__":
    main()
