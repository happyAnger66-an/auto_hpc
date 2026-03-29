#!/usr/bin/env python3
"""
CuTeDSL 基准：Linear Y = X @ W^T + b（FP32）。

优化路径（默认）：**块tile + 共享内存** 的 GEMM，与 PyTorch/CUTLASS 常见实践一致：
  - 将权重整理为 **Wt = W.T**，形状 [K, N] 行主序，使 B 块在 K 维上连续、利于合并读
  - CTA tile (BM×BN)，沿 K 以 BK 为步长分块；sA[BM,BK]、sB[BK,BN] 放在 shared memory
  - 256 线程 / block：每个线程用 **4×4 寄存器 tile**（rmem fragment）累加，共覆盖 64×64 输出子块
  - **GMEM→SMEM**：``local_tile`` 取出当前 K 步的 (64×16)/(16×64) 块，再用 ``composition(tile, tv_layout)`` 将 256 线程映射为各 4 个连续 float，``cute.copy`` + ``CopyUniversalOp`` 一次搬 4 元组（动态 stride 的全局张量不宜强行 ``num_bits_per_copy=128``，由编译器自动向量化）

可选 ``--kernel naive``：每输出一线程 + 全 K 循环（仅作对照）。

可选 ``--pipeline double``：双槽 + 同步 ``CopyUniversalOp``（见 ``linear_tiled_kernel_double_buffer``），占用率常更差。**``double_cpasync32``**：双槽布局与 ``double`` 相同；预取目前用 **同步** ``CopyUniversalOp``（见 kernel 内说明）。在已安装的 CuTeDSL 上 ``cute.copy(CopyG2SOp, …)`` 与 ``composition(local_tile, (256,4))`` 的 rank-1 视图结合后 **数值与参考不一致**，按 float 的 ``slice_`` 亦无法通过 layout congruence，故真正的 32b ``cp.async`` 预取暂不能接。

算术量（与 cuBLAS/cuDNN 一致）：
  FLOPs = 2*M*N*K + M*N，GFLOPS = FLOPs / time_s / 1e9
  （计时在 pad 后的张量上；报告 GFLOPS 仍按**原始** M,N,K 计 FLOPs，便于与 compare 其它列对齐）
"""

from __future__ import annotations

import argparse
import sys
import time

import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

# 静态 tile 尺寸（与 SmemAllocator 的静态 layout 要求一致）
BM = 64
BN = 64
BK = 16
THREADS = 256  # 16×16 输出子网格 × 每线程 4×4 = 64×64


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def pad_linear_tensors(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """返回 pad 后的 X, Wt, b_pad, Y_pad 及原始 m,n,k。"""
    m, k = x.shape
    n, k2 = w.shape
    if k != k2:
        raise ValueError(f"K mismatch: x has {k}, w has {k2}")
    mp = ceil_div(m, BM) * BM
    np_ = ceil_div(n, BN) * BN
    kp = ceil_div(k, BK) * BK

    x_pad = torch.zeros(mp, kp, device=x.device, dtype=x.dtype)
    x_pad[:m, :k] = x

    wt = w.transpose(0, 1).contiguous()
    wt_pad = torch.zeros(kp, np_, device=w.device, dtype=w.dtype)
    wt_pad[:k, :n] = wt

    b_pad = torch.zeros(np_, device=b.device, dtype=b.dtype)
    b_pad[:n] = b

    y_pad = torch.empty(mp, np_, device=x.device, dtype=x.dtype)
    return x_pad, wt_pad, b_pad, y_pad, m, n, k


@cute.kernel
def linear_tiled_kernel(
    mY: cute.Tensor,
    mX: cute.Tensor,
    mWt: cute.Tensor,
    mB: cute.Tensor,
):
    """
    Y = X @ Wt + b，X[M,K], Wt[K,N], b[N], Y[M,N]，均为行主序；Wt 即 W.T。
    假定 M,N,K 已 pad 到 64/64/16 倍数（与模块常量 BM/BN/BK 一致）。
    Tile 尺寸必须为源码字面量，避免 @cute.kernel 闭包捕获导致 JIT 失败。
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    dtype = mY.element_type
    zero = dtype(0.0)

    bm = bidx * 64
    bn = bidy * 64
    kdim = cute.size(mX, mode=[1])

    smem = utils.SmemAllocator()
    lay_a = cute.make_layout((64, 16))
    lay_b = cute.make_layout((16, 64))
    sA = smem.allocate_tensor(dtype, lay_a, byte_alignment=4)
    sB = smem.allocate_tensor(dtype, lay_b, byte_alignment=4)

    ti = tidx // 16
    tj = tidx % 16

    frag = cute.make_rmem_tensor((4, 4), dtype)
    for ii in range(4):
        for jj in range(4):
            frag[ii, jj] = zero

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)
    tv_g2s = cute.make_layout((256, 4), stride=(4, 1))
    frg_g2s = cute.make_rmem_tensor((4,), dtype)

    for k_tile in range(0, kdim, 16):
        kb = k_tile // 16
        g_a = cute.local_tile(mX, (64, 16), (bidx, kb))
        t_g_a = cute.composition(g_a, tv_g2s)
        t_s_a = cute.composition(sA, tv_g2s)
        thr_ga = t_g_a[(tidx, None)]
        thr_sa = t_s_a[(tidx, None)]
        cute.copy(copy_atom, thr_ga, frg_g2s)
        cute.copy(copy_atom, frg_g2s, thr_sa)

        g_b = cute.local_tile(mWt, (16, 64), (kb, bidy))
        t_g_b = cute.composition(g_b, tv_g2s)
        t_s_b = cute.composition(sB, tv_g2s)
        thr_gb = t_g_b[(tidx, None)]
        thr_sb = t_s_b[(tidx, None)]
        cute.copy(copy_atom, thr_gb, frg_g2s)
        cute.copy(copy_atom, frg_g2s, thr_sb)

        cute.arch.sync_threads()

        for kk in range(16):
            for ii in range(4):
                for jj in range(4):
                    frag[ii, jj] = frag[ii, jj] + sA[4 * ti + ii, kk] * sB[kk, 4 * tj + jj]

        cute.arch.sync_threads()

    for ii in range(4):
        for jj in range(4):
            gi = bm + 4 * ti + ii
            gj = bn + 4 * tj + jj
            mY[gi, gj] = frag[ii, jj] + mB[gj]


@cute.kernel
def linear_tiled_kernel_double_buffer(
    mY: cute.Tensor,
    mX: cute.Tensor,
    mWt: cute.Tensor,
    mB: cute.Tensor,
):
    """
    与 ``linear_tiled_kernel`` 相同的 GEMM 与向量化 GMEM→SMEM 拷贝，但使用 **双份 smem**：
    奇偶 K 块交替写入 ``sA[0]`` / ``sA[1]``（``sB`` 同理）。Prologue 装入第 0 条 K；
    每个 ``k_tile`` 上先预取下一条（若存在）到「写缓冲」，再对「读缓冲」做 MMA 子块累加。
    在同步 ``CopyUniversalOp`` 下无法真正重叠 GMEM 与计算；**双倍 smem** 往往降低占用率，整体常 **慢于** ``linear_tiled_kernel``。栅栏仅在有协作预取/尚有下一轮时插入，以略减多余 ``sync``。

    ``--pipeline double_cpasync32`` 当前 **与此 kernel 相同**（入口复用）：已尝试
    ``CopyG2SOp``+``num_bits_per_copy=32`` 直连 smem，编译通过但数值错误；按 float 切分视图则
    ``slice_`` 与 rank-1 构图不 weakly congruent。真 4070 / sm_89 上 32b ``cp.async`` 需等 DSL 修复或手写 PTX/cu 辅助。
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    dtype = mY.element_type
    zero = dtype(0.0)

    bm = bidx * 64
    bn = bidy * 64
    kdim = cute.size(mX, mode=[1])

    smem = utils.SmemAllocator()
    lay_a2 = cute.make_layout((2, 64, 16))
    lay_b2 = cute.make_layout((2, 16, 64))
    sA = smem.allocate_tensor(dtype, lay_a2, byte_alignment=4)
    sB = smem.allocate_tensor(dtype, lay_b2, byte_alignment=4)

    ti = tidx // 16
    tj = tidx % 16

    frag = cute.make_rmem_tensor((4, 4), dtype)
    for ii in range(4):
        for jj in range(4):
            frag[ii, jj] = zero

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)
    tv_g2s = cute.make_layout((256, 4), stride=(4, 1))
    frg_g2s = cute.make_rmem_tensor((4,), dtype)

    # --- Prologue：K 条 0 装入 buffer 0
    g_a0 = cute.local_tile(mX, (64, 16), (bidx, 0))
    slab_a0 = cute.slice_(sA, (0, None, None))
    t_g_a0 = cute.composition(g_a0, tv_g2s)
    t_s_a0 = cute.composition(slab_a0, tv_g2s)
    cute.copy(copy_atom, t_g_a0[(tidx, None)], frg_g2s)
    cute.copy(copy_atom, frg_g2s, t_s_a0[(tidx, None)])

    g_b0 = cute.local_tile(mWt, (16, 64), (0, bidy))
    slab_b0 = cute.slice_(sB, (0, None, None))
    t_g_b0 = cute.composition(g_b0, tv_g2s)
    t_s_b0 = cute.composition(slab_b0, tv_g2s)
    cute.copy(copy_atom, t_g_b0[(tidx, None)], frg_g2s)
    cute.copy(copy_atom, frg_g2s, t_s_b0[(tidx, None)])

    cute.arch.sync_threads()

    for k_tile in range(0, kdim, 16):
        kn = k_tile // 16
        if cute.elem_less(k_tile + 16, kdim):
            kb_n = kn + 1
            wp = 1 - (kn % 2)
            slab_aw = cute.slice_(sA, (wp, None, None))
            slab_bw = cute.slice_(sB, (wp, None, None))
            g_an = cute.local_tile(mX, (64, 16), (bidx, kb_n))
            g_bn = cute.local_tile(mWt, (16, 64), (kb_n, bidy))
            t_g_an = cute.composition(g_an, tv_g2s)
            t_s_aw = cute.composition(slab_aw, tv_g2s)
            cute.copy(copy_atom, t_g_an[(tidx, None)], frg_g2s)
            cute.copy(copy_atom, frg_g2s, t_s_aw[(tidx, None)])
            t_g_bn = cute.composition(g_bn, tv_g2s)
            t_s_bw = cute.composition(slab_bw, tv_g2s)
            cute.copy(copy_atom, t_g_bn[(tidx, None)], frg_g2s)
            cute.copy(copy_atom, frg_g2s, t_s_bw[(tidx, None)])
            cute.arch.sync_threads()

        rp = kn % 2
        for kk in range(16):
            for ii in range(4):
                for jj in range(4):
                    frag[ii, jj] = (
                        frag[ii, jj]
                        + sA[rp, 4 * ti + ii, kk] * sB[rp, kk, 4 * tj + jj]
                    )

        if cute.elem_less(k_tile + 16, kdim):
            cute.arch.sync_threads()

    for ii in range(4):
        for jj in range(4):
            gi = bm + 4 * ti + ii
            gj = bn + 4 * tj + jj
            mY[gi, gj] = frag[ii, jj] + mB[gj]


@cute.kernel
def linear_naive_kernel(
    mY: cute.Tensor, mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor
):
    """未 pad：W 为 [N,K]，每输出点内层 K 循环（慢，仅对照）。"""
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()

    linear_id = bx * bdx + tx
    m = cute.size(mX, mode=[0])
    n = cute.size(mY, mode=[1])
    kdim = cute.size(mX, mode=[1])
    mi = linear_id // n
    ni = linear_id % n

    dtype = mY.element_type
    zero = dtype(0.0)

    if cute.elem_less((mi, ni), (m, n)):
        acc = zero
        for kk in range(kdim):
            acc = acc + mX[(mi, kk)] * mW[(ni, kk)]
        mY[(mi, ni)] = acc + mB[ni]


@cute.jit
def linear_entry_tiled(mY, mX, mWt, mB):
    m = cute.size(mX, mode=[0])
    n = cute.size(mY, mode=[1])
    grid_x = (m + 64 - 1) // 64
    grid_y = (n + 64 - 1) // 64
    linear_tiled_kernel(mY, mX, mWt, mB).launch(
        grid=[grid_x, grid_y, 1],
        block=[256, 1, 1],
    )


@cute.jit
def linear_entry_tiled_double_buffer(mY, mX, mWt, mB):
    m = cute.size(mX, mode=[0])
    n = cute.size(mY, mode=[1])
    grid_x = (m + 64 - 1) // 64
    grid_y = (n + 64 - 1) // 64
    linear_tiled_kernel_double_buffer(mY, mX, mWt, mB).launch(
        grid=[grid_x, grid_y, 1],
        block=[256, 1, 1],
    )


@cute.jit
def linear_entry_naive(mY, mX, mW, mB):
    threads_per_block = 256
    m = cute.size(mX, mode=[0])
    n = cute.size(mY, mode=[1])
    total_mn = m * n
    blocks = (total_mn + threads_per_block - 1) // threads_per_block
    linear_naive_kernel(mY, mX, mW, mB).launch(
        grid=[blocks, 1, 1],
        block=[256, 1, 1],
    )


def flops_linear(m: int, n: int, k: int) -> float:
    return float(2 * m * n * k + m * n)


def main() -> None:
    p = argparse.ArgumentParser(description="CuTeDSL linear benchmark (Y = X @ W.T + b)")
    p.add_argument("--m", type=int, default=1024, help="batch / rows of X")
    p.add_argument("--n", type=int, default=1024, help="out_features / rows of W")
    p.add_argument("--k", type=int, default=1024, help="in_features / cols of X,W")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--machine", action="store_true", help="单行机器可读输出（供 compare.py）")
    p.add_argument(
        "--kernel",
        choices=("tiled", "naive"),
        default="tiled",
        help="tiled: smem 块 GEMM + Wt 布局（默认）；naive: 旧实现",
    )
    p.add_argument(
        "--pipeline",
        choices=("single", "double", "double_cpasync32"),
        default="single",
        help="single|double|double_cpasync32（double_cpasync32 当前等同 double；CopyG2S 构图待 DSL）",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)

    m, n, k = args.m, args.n, args.k
    x = torch.randn(m, k, device="cuda", dtype=torch.float32)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    if args.kernel == "tiled":
        x_pad, wt_pad, b_pad, y_pad, mo, no, ko = pad_linear_tensors(x, w, b)
        assert (mo, no, ko) == (m, n, k)
        y_t = from_dlpack(y_pad).mark_layout_dynamic()
        x_t = from_dlpack(x_pad).mark_layout_dynamic()
        wt_t = from_dlpack(wt_pad).mark_layout_dynamic()
        b_t = from_dlpack(b_pad).mark_layout_dynamic()
        if args.pipeline == "double":
            entry = linear_entry_tiled_double_buffer
        elif args.pipeline == "double_cpasync32":
            # 与 double 同内核：CopyG2SOp+当前构图运行结果错误，见模块说明与 linear_tiled_kernel_double_buffer 注释
            entry = linear_entry_tiled_double_buffer
        else:
            entry = linear_entry_tiled
        compile_args = (y_t, x_t, wt_t, b_t)
        run_args = (y_t, x_t, wt_t, b_t)
        y_out = y_pad
    else:
        y = torch.empty(m, n, device="cuda", dtype=torch.float32)
        y_t = from_dlpack(y).mark_layout_dynamic()
        x_t = from_dlpack(x).mark_layout_dynamic()
        w_t = from_dlpack(w).mark_layout_dynamic()
        b_t = from_dlpack(b).mark_layout_dynamic()
        entry = linear_entry_naive
        compile_args = (y_t, x_t, w_t, b_t)
        run_args = (y_t, x_t, w_t, b_t)
        y_out = y

    t0 = time.perf_counter()
    compiled = cute.compile(entry, *compile_args, options="--generate-line-info")
    compile_s = time.perf_counter() - t0

    compiled(*run_args)
    torch.cuda.synchronize()
    ref = torch.nn.functional.linear(x, w, b)
    if args.kernel == "tiled":
        torch.testing.assert_close(y_out[:m, :n], ref, rtol=1e-4, atol=1e-3)
    else:
        torch.testing.assert_close(y_out, ref, rtol=1e-4, atol=1e-4)

    for _ in range(args.warmup):
        compiled(*run_args)
    torch.cuda.synchronize()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(args.iters):
        compiled(*run_args)
    ev1.record()
    torch.cuda.synchronize()
    ms_total = ev0.elapsed_time(ev1)
    ms_per_iter = ms_total / args.iters

    f = flops_linear(m, n, k)
    gflops = (f / (ms_per_iter / 1000.0)) / 1e9

    if args.machine:
        print(
            f"cutedsl_ms_per_iter {ms_per_iter:.6f} cutedsl_gflops {gflops:.4f} cutedsl_compile_s {compile_s:.6f}"
        )
    else:
        pipe = args.pipeline if args.kernel == "tiled" else "-"
        print(
            f"shape M={m} N={n} K={k}  fp32  kernel={args.kernel}  pipeline={pipe}  "
            f"(tiled uses BM={BM} BN={BN} BK={BK})"
        )
        print(f"compile_s={compile_s:.4f}  ms/iter={ms_per_iter:.6f}  GFLOPS={gflops:.2f}")
        print(f"FLOPs/iter={f:.0f}  (2*M*N*K + M*N)")


if __name__ == "__main__":
    main()
