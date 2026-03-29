#!/usr/bin/env python3
"""
扫描多组 (thr_layout, val_layout) 形状，对 elementwise add kernel 做计时并输出带宽。

说明：make_layout_tv 需要编译期静态 layout；DSL 也不接受 exec 生成的函数。
因此对每种 (thr,val) 使用本文件内一个独立的 @cute.jit 入口（见 _add_entry_layout_XX）。

用法（在已安装 CuTeDSL 的环境下）:
  cd auto_hpc/element-wise/cutedsl && python3 benchmark_layouts.py --m 4096 --n 4096
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


# 与 hello_cutedsl_elementwise_add.py 中设备端逻辑一致（避免跨目录 import 问题）。
@cute.kernel
def add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkCrd = cC[blk_coord]

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    thrA = thr_copy.partition_S(blkA)
    thrB = thr_copy.partition_S(blkB)
    thrC = thr_copy.partition_S(blkC)
    thrCrd = thr_copy.partition_S(blkCrd)

    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)

    for i in range(cute.size(frgPred)):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    cute.copy(copy_atom, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom, thrB, frgB, pred=frgPred)

    frgC.store(frgA.load() + frgB.load())
    cute.copy(copy_atom, frgC, thrC, pred=frgPred)


# 以下每个函数内布局均为源码字面量；顺序与 default_candidates 去重后的 layout_id 一致。
@cute.jit
def _add_entry_layout_00(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_01(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((8, 16), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_02(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((16, 8), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_03(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((32, 4), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_04(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((2, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_05(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((64, 2), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_06(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((128, 1), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_07(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((1, 128), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_08(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((8, 2), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_09(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((2, 8), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_10(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, 1), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_11(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, 16), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_12(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((8, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@cute.jit
def _add_entry_layout_13(mA, mB, mC):
    thr_layout = cute.make_ordered_layout((16, 16), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


ADD_ENTRY_FUNCS = (
    _add_entry_layout_00,
    _add_entry_layout_01,
    _add_entry_layout_02,
    _add_entry_layout_03,
    _add_entry_layout_04,
    _add_entry_layout_05,
    _add_entry_layout_06,
    _add_entry_layout_07,
    _add_entry_layout_08,
    _add_entry_layout_09,
    _add_entry_layout_10,
    _add_entry_layout_11,
    _add_entry_layout_12,
    _add_entry_layout_13,
)


@dataclass(frozen=True)
class LayoutCandidate:
    name: str
    layout_id: int
    thr_m: int
    thr_n: int
    val_m: int
    val_n: int

    @property
    def num_threads(self) -> int:
        return self.thr_m * self.thr_n

    @property
    def vals_per_thread(self) -> int:
        return self.val_m * self.val_n


def default_candidates() -> list[LayoutCandidate]:
    """
    候选说明：
    - thr_m*thr_n：每 CTA 线程数（建议为 32 倍数，常用 128/256）。
    - val_m*val_n：每线程逻辑 value 数；与 hello 默认 (4,4)=16 对齐的变体便于对比访存形态。
    """
    c: list[LayoutCandidate] = []
    # 固定 128 线程，扫 thr 形状
    for tm, tn in ((4, 32), (8, 16), (16, 8), (32, 4), (2, 64), (64, 2), (128, 1), (1, 128)):
        c.append(LayoutCandidate(f"thr_{tm}x{tn}_val_4x4", -1, tm, tn, 4, 4))
    # 固定 baseline thr，扫 val（保持每线程 16 个元素）
    for vm, vn in ((4, 4), (8, 2), (2, 8), (16, 1), (1, 16)):
        c.append(LayoutCandidate(f"thr_4x32_val_{vm}x{vn}", -1, 4, 32, vm, vn))
    # 256 线程
    c.append(LayoutCandidate("thr_8x32_val_4x4", -1, 8, 32, 4, 4))
    c.append(LayoutCandidate("thr_16x16_val_4x4", -1, 16, 16, 4, 4))
    # 去重：同一 (thr, val) 只保留第一次出现的 name；layout_id 与 ADD_ENTRY_FUNCS 下标一致
    seen: set[tuple[int, int, int, int]] = set()
    out: list[LayoutCandidate] = []
    for x in c:
        key = (x.thr_m, x.thr_n, x.val_m, x.val_n)
        if key in seen:
            continue
        seen.add(key)
        lid = len(out)
        out.append(
            LayoutCandidate(
                name=x.name,
                layout_id=lid,
                thr_m=x.thr_m,
                thr_n=x.thr_n,
                val_m=x.val_m,
                val_n=x.val_n,
            )
        )
    return out


def validate_candidate(x: LayoutCandidate) -> str | None:
    nt = x.num_threads
    if nt % 32 != 0:
        return f"threads={nt} 不是 32 的倍数"
    if nt > 1024:
        return f"threads={nt} 超过每 block 上限 1024"
    if x.val_m < 1 or x.val_n < 1:
        return "val 维度必须为正"
    return None


def bench_one(
    cand: LayoutCandidate,
    a_t,
    b_t,
    c_t,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    m: int,
    n: int,
    warmup: int,
    iters: int,
    compile_options: str,
) -> tuple[float | None, float | None, float | None, str | None]:
    """
    返回 (compile_s, kernel_ms_per_iter, gflops, error_message)。
    gflops：按每个输出元素 1 次加法计 FLOPs，GFLOPS = (M*N) / time_s / 1e9。
    """
    err = validate_candidate(cand)
    if err:
        return None, None, None, err

    try:
        t0 = time.perf_counter()
        entry_fn = ADD_ENTRY_FUNCS[cand.layout_id]
        compiled = cute.compile(
            entry_fn,
            a_t,
            b_t,
            c_t,
            options=compile_options,
        )
        compile_s = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - 基准脚本需捕获 JIT 失败
        return None, None, None, f"compile failed: {e!s}"

    try:
        compiled(a_t, b_t, c_t)
        torch.cuda.synchronize()
        torch.testing.assert_close(c, a + b, rtol=1e-5, atol=1e-5)
    except Exception as e:  # noqa: BLE001
        return compile_s, None, None, f"correctness failed: {e!s}"

    for _ in range(warmup):
        compiled(a_t, b_t, c_t)
    torch.cuda.synchronize()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    for _ in range(iters):
        compiled(a_t, b_t, c_t)
    ev_end.record()
    torch.cuda.synchronize()
    ms_total = ev_start.elapsed_time(ev_end)
    ms_per_iter = ms_total / iters

    flops = float(m * n)
    gflops = (flops / (ms_per_iter / 1000.0)) / 1e9

    return compile_s, ms_per_iter, gflops, None


def run_benchmark(
    candidates: Iterable[LayoutCandidate],
    m: int,
    n: int,
    warmup: int,
    iters: int,
    compile_options: str,
) -> None:
    if not torch.cuda.is_available():
        print("需要 CUDA GPU。", file=sys.stderr)
        sys.exit(1)

    cand_list = list(candidates)
    max_lid = max(c.layout_id for c in cand_list)
    n_entry = len(ADD_ENTRY_FUNCS)
    if max_lid >= n_entry:
        print(
            f"layout_id 最大为 {max_lid}，但只有 {n_entry} 个 ADD_ENTRY_FUNCS，请扩展 _add_entry_layout_XX。",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda")
    a = torch.randn(m, n, device=device, dtype=torch.float32)
    b = torch.randn(m, n, device=device, dtype=torch.float32)
    c = torch.zeros_like(a)

    a_t = from_dlpack(a).mark_layout_dynamic()
    b_t = from_dlpack(b).mark_layout_dynamic()
    c_t = from_dlpack(c).mark_layout_dynamic()

    rows: list[tuple] = []
    for cand in cand_list:
        c.zero_()
        compile_s, ms, gflops, err = bench_one(
            cand, a_t, b_t, c_t, a, b, c, m, n, warmup, iters, compile_options
        )
        rows.append((cand, compile_s, ms, gflops, err))

    hdr = (
        f"{'name':<28} {'id':>3} {'thr':^9} {'val':^9} {'thr#':>5} {'v/thr':>6} "
        f"{'compile_s':>10} {'ms/iter':>10} {'GFLOPS':>10}  note"
    )
    print(hdr)
    print("-" * len(hdr))

    best_ms = None
    best_name = None
    for cand, compile_s, ms, gflops, err in rows:
        thr_s = f"{cand.thr_m}x{cand.thr_n}"
        val_s = f"{cand.val_m}x{cand.val_n}"
        if err:
            line = (
                f"{cand.name:<28} {cand.layout_id:>3} {thr_s:^9} {val_s:^9} {cand.num_threads:>5} "
                f"{cand.vals_per_thread:>6} {'-':>10} {'-':>10} {'-':>10}  {err}"
            )
            print(line)
            continue
        assert ms is not None and gflops is not None and compile_s is not None
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_name = cand.name
        print(
            f"{cand.name:<28} {cand.layout_id:>3} {thr_s:^9} {val_s:^9} {cand.num_threads:>5} "
            f"{cand.vals_per_thread:>6} {compile_s:>10.3f} {ms:>10.4f} {gflops:>10.2f}  OK"
        )

    print("-" * len(hdr))
    if best_name and best_ms is not None:
        print(f"最快（按 ms/iter）: {best_name}  @ {best_ms:.4f} ms/iter")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark thr_layout / val_layout for CuTeDSL add")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--only-id",
        type=int,
        default=None,
        help="只跑指定 layout_id（用于与 cuBLAS/cuDNN 对比脚本对接）",
    )
    p.add_argument(
        "--machine",
        action="store_true",
        help="与 --only-id 合用：单行输出 ms_per_iter gflops compile_s，便于脚本解析",
    )
    p.add_argument(
        "--compile-options",
        type=str,
        default="--generate-line-info",
        help="传给 cute.compile 的选项字符串",
    )
    args = p.parse_args()

    print(f"problem (m,n)=({args.m},{args.n})  fp32  warmup={args.warmup} iters={args.iters}")

    if args.only_id is not None:
        cand_list = [c for c in default_candidates() if c.layout_id == args.only_id]
        if not cand_list:
            print(f"未知 layout_id={args.only_id}", file=sys.stderr)
            sys.exit(1)
        if not torch.cuda.is_available():
            print("需要 CUDA GPU。", file=sys.stderr)
            sys.exit(1)
        cand = cand_list[0]
        device = torch.device("cuda")
        a = torch.randn(args.m, args.n, device=device, dtype=torch.float32)
        b = torch.randn(args.m, args.n, device=device, dtype=torch.float32)
        c = torch.zeros_like(a)
        a_t = from_dlpack(a).mark_layout_dynamic()
        b_t = from_dlpack(b).mark_layout_dynamic()
        c_t = from_dlpack(c).mark_layout_dynamic()
        compile_s, ms, gflops, err = bench_one(
            cand,
            a_t,
            b_t,
            c_t,
            a,
            b,
            c,
            args.m,
            args.n,
            args.warmup,
            args.iters,
            args.compile_options,
        )
        if err:
            print(err, file=sys.stderr)
            sys.exit(1)
        assert ms is not None and gflops is not None and compile_s is not None
        if args.machine:
            print(
                f"cutedsl_ms_per_iter {ms:.6f} cutedsl_gflops {gflops:.4f} cutedsl_compile_s {compile_s:.6f}"
            )
        else:
            print(
                f"{cand.name}  compile_s={compile_s:.3f}  ms/iter={ms:.4f}  GFLOPS={gflops:.2f}"
            )
        return

    run_benchmark(
        default_candidates(),
        args.m,
        args.n,
        args.warmup,
        args.iters,
        args.compile_options,
    )


if __name__ == "__main__":
    main()
