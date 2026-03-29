#!/usr/bin/env python3
"""
统一对比 FP32 Linear：Y = X @ W^T + b（CuTeDSL / cuBLAS 列主序 Sgemm / cuDNN）。

性能指标（三者一致）：
  FLOPs = 2*M*N*K + M*N（matmul 按 MAC 计 2 FLOPs，bias 按每输出 1 次加法）
  GFLOPS = FLOPs / time_seconds / 1e9

示例:
  cmake -S linear/cublas -B linear/cublas/build -DCMAKE_BUILD_TYPE=Release && cmake --build linear/cublas/build
  cmake -S linear/cudnn -B linear/cudnn/build -DCMAKE_BUILD_TYPE=Release && cmake --build linear/cudnn/build
  python3 linear/compare.py --m 1024 --n 1024 --k 1024
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


def flops_linear(m: int, n: int, k: int) -> float:
    return float(2 * m * n * k + m * n)


def run_cutedsl(
    m: int, n: int, k: int, warmup: int, iters: int
) -> tuple[float, float, float] | None:
    script = ROOT / "cutedsl" / "benchmark_linear.py"
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
        "--k",
        str(k),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
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
    exe: Path, m: int, n: int, k: int, warmup: int, iters: int
) -> tuple[float, float] | None:
    if not exe.is_file():
        return None
    r = subprocess.run(
        [str(exe), str(m), str(n), str(k), str(warmup), str(iters)],
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
        description="Compare linear layer: CuTeDSL vs cuBLAS vs cuDNN (GFLOPS)"
    )
    p.add_argument("--m", type=int, default=1024, help="batch size（X 的行数）")
    p.add_argument("--n", type=int, default=1024, help="out_features（W 的行数）")
    p.add_argument("--k", type=int, default=1024, help="in_features（X/W 的列数）")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument(
        "--cublas-exe",
        type=Path,
        default=ROOT / "cublas" / "build" / "linear_bench",
    )
    p.add_argument(
        "--cudnn-exe",
        type=Path,
        default=ROOT / "cudnn" / "build" / "linear_bench",
    )
    args = p.parse_args()

    m, n, k = args.m, args.n, args.k
    f = flops_linear(m, n, k)

    print(
        f"Linear Y = X@W^T+b  形状 X[{m},{k}]  W[{n},{k}]  b[{n}]  fp32\n"
        f"warmup={args.warmup} iters={args.iters}\n"
        f"指标: GFLOPS = FLOPs/time_s/1e9，FLOPs/iter = 2*M*N*K + M*N = {f:.0f}\n"
    )

    rows: list[tuple[str, str, float | None, float | None, str]] = []

    cd = run_cutedsl(m, n, k, args.warmup, args.iters)
    if cd:
        rows.append(
            (
                "CuTeDSL",
                "block-tiled smem GEMM (Wt)",
                cd[0],
                cd[1],
                f"compile_s={cd[2]:.3f}（仅首次）；见 cutedsl/benchmark_linear.py",
            )
        )
    else:
        rows.append(("CuTeDSL", "block-tiled smem GEMM", None, None, "失败或跳过"))

    cb = run_native(args.cublas_exe, m, n, k, args.warmup, args.iters)
    if cb:
        rows.append(
            (
                "cuBLAS",
                "Sgemm(col-major)+Saxpy(bias)",
                cb[0],
                cb[1],
                "Sgemm+Saxpy/迭代",
            )
        )
    else:
        rows.append(
            ("cuBLAS", "Sgemm(col-major)+Saxpy(bias)", None, None, f"未找到: {args.cublas_exe}")
        )

    dn = run_native(args.cudnn_exe, m, n, k, args.warmup, args.iters)
    if dn:
        rows.append(("cuDNN", "conv1x1 + AddTensor", dn[0], dn[1], "两调用/迭代"))
    else:
        rows.append(("cuDNN", "conv1x1 + AddTensor", None, None, f"未找到: {args.cudnn_exe}"))

    hdr = f"{'backend':<12} {'impl':<32} {'ms/iter':>12} {'GFLOPS':>12}  note"
    print(hdr)
    print("-" * len(hdr))
    best_gf = None
    best_name = ""
    for name, impl, ms, gflops, note in rows:
        if ms is not None and gflops is not None:
            if best_gf is None or gflops > best_gf:
                best_gf = gflops
                best_name = name
            print(f"{name:<12} {impl:<32} {ms:>12.4f} {gflops:>12.2f}  {note}")
        else:
            print(f"{name:<12} {impl:<32} {'-':>12} {'-':>12}  {note}")
    print("-" * len(hdr))
    if best_gf is not None:
        print(f"最高 GFLOPS: {best_name} @ {best_gf:.2f} GFLOPS")

    print(
        "\n说明: cuBLAS 路径在设备上使用列主序（cublasSetMatrix）；"
        "bias 用 Saxpy 与预展开向量相加。cuDNN 为卷积/GEMM 融合路径。"
    )


if __name__ == "__main__":
    main()
