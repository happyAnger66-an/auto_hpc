#!/usr/bin/env python3
"""CuTeDSL 小示例：Y = X @ W^T + b，使用与 benchmark_linear 相同的 tiled 路径。"""

import argparse
import sys
import time
from pathlib import Path

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

_cutedsl_dir = Path(__file__).resolve().parent
if str(_cutedsl_dir) not in sys.path:
    sys.path.insert(0, str(_cutedsl_dir))
from benchmark_linear import linear_entry_tiled, pad_linear_tensors


def main() -> None:
    p = argparse.ArgumentParser(description="CuTeDSL linear hello")
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--k", type=int, default=32)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    m, n, k = args.m, args.n, args.k
    x = torch.randn(m, k, device="cuda", dtype=torch.float32)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    x_pad, wt_pad, b_pad, y_pad, mo, no, ko = pad_linear_tensors(x, w, b)
    assert (mo, no, ko) == (m, n, k)

    y_t = from_dlpack(y_pad).mark_layout_dynamic()
    x_t = from_dlpack(x_pad).mark_layout_dynamic()
    wt_t = from_dlpack(wt_pad).mark_layout_dynamic()
    b_t = from_dlpack(b_pad).mark_layout_dynamic()

    t0 = time.time()
    compiled = cute.compile(
        linear_entry_tiled, y_t, x_t, wt_t, b_t, options="--generate-line-info"
    )
    t1 = time.time()
    compiled(y_t, x_t, wt_t, b_t)
    torch.cuda.synchronize()
    t2 = time.time()

    ref = torch.nn.functional.linear(x, w, b)
    torch.testing.assert_close(y_pad[:m, :n], ref, rtol=1e-4, atol=1e-3)
    print(f"[OK] verify passed M={m} N={n} K={k}")
    print(f"[INFO] compile time: {(t1 - t0):.3f}s, first run: {(t2 - t1):.3f}s")
    print("[INFO] y[0,0:4] =", y_pad[0, 0:4].tolist())


if __name__ == "__main__":
    main()
