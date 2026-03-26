#!/usr/bin/env python3
"""
CuTeDSL Hello World:
Compute C = A + B on GPU with a tiny custom kernel.
"""

import argparse
import time

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


# @cute.kernel: 定义设备端 kernel（在 GPU 上执行）。
@cute.kernel
def add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,  # coordinate tensor used for predication
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    # thread_idx/block_idx: 获取当前线程与当前 CTA(block) 的索引。
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkCrd = cC[blk_coord]

    # make_copy_atom: 声明一次“拷贝原语”（这里是通用拷贝）。
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    # make_tiled_copy_tv: 基于线程布局 + value布局，构建 tiled copy 映射规则。
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    # get_slice: 取出“当前线程”在这个 tiled copy 里的视图。
    thr_copy = tiled_copy.get_slice(tidx)

    # partition_S: 按当前线程的映射切分 source tensor。
    thrA = thr_copy.partition_S(blkA)
    thrB = thr_copy.partition_S(blkB)
    thrC = thr_copy.partition_S(blkC)
    thrCrd = thr_copy.partition_S(blkCrd)

    # make_fragment_like: 按目标形状分配寄存器片段（fragment）。
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)
    # make_rmem_tensor: 在寄存器地址空间创建 pred tensor，做越界保护。
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)

    for i in range(cute.size(frgPred)):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    # cute.copy: 执行真实拷贝；pred 控制哪些元素有效（避免 OOB）。
    cute.copy(copy_atom, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom, thrB, frgB, pred=frgPred)

    frgC.store(frgA.load() + frgB.load())
    cute.copy(copy_atom, frgC, thrC, pred=frgPred)


# @cute.jit: 定义主机侧 DSL 入口，供 cute.compile 做 JIT 编译。
@cute.jit
def add_entry(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    # 128 threads per CTA, each thread handles a small vector.
    # make_ordered_layout: 构建线程/值的逻辑布局。
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    # make_layout_tv: 生成 tile 形状 + thread-value 映射布局。
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # zipped_divide: 把原始 tensor 按 tile 进行分块，得到 CTA 级视图。
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    # make_identity_tensor: 创建坐标张量；用于后续越界判定。
    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)

    # launch: 指定 grid/block 后发射 kernel。
    add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def main():
    parser = argparse.ArgumentParser(description="CuTeDSL hello world: C=A+B")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    m, n = args.m, args.n
    a = torch.randn(m, n, device="cuda", dtype=torch.float32)
    b = torch.randn(m, n, device="cuda", dtype=torch.float32)
    c = torch.zeros_like(a)

    # from_dlpack: 把 torch tensor 封装为 CuTeDSL 可识别的运行时 tensor 视图（零拷贝语义）。
    # mark_layout_dynamic: 告诉编译器“布局信息在运行时决定”，减少布局写死导致的特化限制。
    a_t = from_dlpack(a).mark_layout_dynamic()
    b_t = from_dlpack(b).mark_layout_dynamic()
    c_t = from_dlpack(c).mark_layout_dynamic()

    t0 = time.time()
    # cute.compile: 对 @cute.jit 入口按当前参数做 JIT 编译，返回可调用对象。
    compiled = cute.compile(add_entry, a_t, b_t, c_t, options="--generate-line-info")
    t1 = time.time()

    # JIT 返回的 callable 只接受预期的位置参数，这里不要传 keyword 参数（如 stream=...）。
    # 如果需要控制 stream，建议在外层切换 torch.cuda.stream(...) 上下文后再调用 compiled(...)。
    compiled(a_t, b_t, c_t)
    torch.cuda.synchronize()
    t2 = time.time()

    torch.testing.assert_close(c, a + b, rtol=1e-5, atol=1e-5)
    print(f"[OK] verify passed for shape=({m}, {n})")
    print(f"[INFO] compile time: {(t1 - t0):.3f}s")
    print(f"[INFO] first launch+run: {(t2 - t1):.3f}s")
    print("[INFO] C[0,0:4] =", c[0, 0:4].tolist())


if __name__ == "__main__":
    main()

