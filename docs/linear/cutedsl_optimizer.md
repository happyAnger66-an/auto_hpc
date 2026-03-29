# CuTeDSL Linear（Y = X @ W^T + b）优化记录

本文档归纳 `auto_hpc/linear/cutedsl/benchmark_linear.py` 中 **CuTeDSL 实现从「教学向朴素 kernel」到「块 tile + 共享内存 GEMM」** 的动机、思路与踩坑，便于日后对照代码复习。

---

## 1. 背景与目标

- **算子语义**：与 `torch.nn.functional.linear(x, w, b)` 一致：`x ∈ ℝ^{M×K}`，`w ∈ ℝ^{N×K}`，`b ∈ ℝ^N`，`y = x @ w^T + b`。
- **对比环境**：与 `linear/cublas`、`linear/cudnn` 共用 `linear/compare.py`，**GFLOPS** 按 `FLOPs = 2MNK + MN`（MAC 计 2 FLOPs，bias 每输出 1 次加法）。
- **目标**：在仍用 CuTeDSL 手写 kernel 的前提下，把性能从「远低于 cuBLAS」拉到「可接受的量级」，并沉淀可复用的 DSL 习惯。

---

## 2. 第一版：Naive 实现（`--kernel naive`）

### 2.1 做法

- 每个输出标量 `(m, n)` 映射到一个线程（1D grid + 256 threads/block，带边界判断）。
- 内层对 **K** 做动态长度循环：`acc += X[m,k] * W[n,k]`，最后 `+ b[n]`。
- 利用 DSL 的 **AST 预处理**：`if cute.elem_less(...)` + `for kk in range(kdim)` 生成 `scf.if` / `scf.for`，由编译器维护 **K 维累加的 SSA**（手写 `for_generate` + `yield_out` 做规约更容易踩「block 无 terminator / dominate」类错误）。

### 2.2 主要问题（思路）

1. **无重用**：每个线程对 `X` 的行、`W` 的行重复读全局内存，无共享内存复用。
2. **指令与访存比差**：内层是标量乘加，无向量化、无 TensorCore、无块级并行组织。
3. **W 的访问形态**：数学上需要 `W[n, k]`；在 `w` 为 `[N,K]` 行主序时，沿 `n` 变化步长为 `K`，与「warp 合并读」的理想模式不一致（但 naive 阶段主要瓶颈仍是「无 tiling」）。

**结论**：适合作为 **正确性基线** 与教学演示，不适合作为性能基线。

---

## 3. 第二版：块 Tile + 共享内存 GEMM（默认 `tiled`）

### 3.1 核心思路（与经典 CUDA GEMM / CUTLASS 一致）

将 `y = x @ w^T` 改写为 **`y = x @ Wt`**，其中 **`Wt = w^T`，形状 `[K, N]` 行主序**：

- `x` 子块：`BM × BK`
- `Wt` 子块：`BK × BN`
- 输出子块：`BM × BN`

对每个 **K 方向条带** `k_tile : k_tile+BK`：

1. 协作把 `x[bm:bm+BM, …]` 与 `Wt[…, bn:bn+BN]` 装入 **shared memory**（`sA`、`sB`）。
2. `sync_threads()`。
3. 各线程在寄存器里对当前条带做子块乘加；多条带结果累加在同一组寄存器上。
4. 再 `sync_threads()`，进入下一 `k_tile`。

这样 **全局内存流量** 从「每输出点 O(K) 次无结构复用」变为 **块级复用**，是性能跃升的主要来源。

### 3.2 具体参数（当前代码）

| 符号 | 取值 | 含义 |
|------|------|------|
| BM, BN | 64 | CTA 在 M、N 方向的输出 tile 边长 |
| BK | 16 | K 方向条带厚度 |
| Threads | 256 | 16×16 线程子网格 × 每线程负责 4×4 输出标量 = 64×64 |

**寄存器累加**：使用 `cute.make_rmem_tensor((4, 4), dtype)` 表示每线程的 **4×4 fragment**，避免手写 16 个独立标量 SSA。

**共享内存**：`cutlass.utils.SmemAllocator` + `cute.make_layout((64,16))` / `((16,64))`；`allocate_tensor` 要求 **静态 layout**，因此 **BK/BM/BN 在 layout 里必须是编译期可静态化的形状**。

### 3.3 向量化 GMEM→SMEM（`local_tile` + `composition` + `cute.copy`）

**动机**：按元素写 `sA[ai,ak] = mX[…]` 指令多、合并度差；希望每线程一次搬 **4 个连续 float**（与行主序下 K/N 向连续一致）。

**做法**：

- 用 `cute.local_tile(mX, (64, 16), (bidx, k_tile // 16))`（及 `mWt` 上 `(16, 64)`、`(k_tile//16, bidy)`）取出当前 K 步的 **静态形状** 子块，避免对整块 `[M,K]` 做易触发 stride 问题的动态切片。
- `tv = cute.make_layout((256, 4), stride=(4, 1))`：将 64×16（或 16×64）tile 展平后，线程 `tidx` 负责线性下标 `[4*tidx : 4*tidx+4)`，与原先「每线程 4 连续元」的映射一致。
- `cute.composition(tile, tv)` 后取 `[(tidx, None)]` 得到 4 元子视图；`cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)` + 两次 `cute.copy`（gmem→rmem→smem）完成搬运。

**注意**：对 `mark_layout_dynamic()` 的全局张量，强行 `num_bits_per_copy=128` 可能在 IR 校验阶段报错（gmem 侧 stride 非完全静态）；当前依赖 **默认** `CopyUniversalOp` 的自动向量化。若需硬 128-bit，通常要对齐布局或使用更底层的 TMA/cp.async 路径。

### 3.4 Python 侧：Pad 与 `Wt`

- **Pad**：将 `M, N, K` 向上取整到 `BM, BN, BK` 的倍数，简化 kernel（无需 tile 内复杂边界谓词）。**无效 pad 区**不参与与 `torch` 的数值对比（只校验 `y[:M,:N]`）。
- **GFLOPS 报告**：计时在 **pad 后** 张量上执行，但 FLOPs 仍按 **用户输入的原始 M,N,K** 计算，以便与 `compare.py` 中 cuBLAS/cuDNN **同一算术量口径**对齐（pad 会带来少量「白算」，大问题上通常可忽略）。

### 3.5 重要踩坑：`@cute.kernel` / `@cute.jit` 与闭包

**现象**：`cute.compile(linear_entry_tiled, …)` 报错类似  
`requires a code object with 0 free vars, not N`。

**原因**：若在 `linear_entry_tiled` 或 `linear_tiled_kernel` 里引用 **模块级** `BM`、`BK` 等变量，Python 会生成 **闭包自由变量**，当前 CuTeDSL 的 JIT 预处理不接受。

**做法**：在 **被 `@cute.jit` / `@cute.kernel` 装饰的函数体内**，对 tile 尺寸使用 **源码字面量**（如 `64`、`16`、`256`）；模块级 `BM/BN/BK` **仅用于** Python 侧 `pad_linear_tensors` 与文档/打印，与 kernel 内数字 **人工保持一致**。

---

## 4. 与官方 CuTeDSL 示例的关系

仓库旁路 `cutlass` 中的高性能范例（如 `examples/python/CuTeDSL/hopper/dense_gemm.py`、Blackwell `tutorial_gemm`）普遍依赖 **TMA、WGMMA/UMMA、流水线** 等 **架构绑定** 能力，学习与迁移成本更高。

本项目的 **tiled 路径** 刻意选择：

- **SIMT + smem tiling**，不绑定 Hopper/Blackwell；
- 代码量与概念量适合作为 **从 naive 到「像样 GEMM」的阶梯**。

若要在特定架构上继续逼近 cuBLAS，应优先阅读对应架构的 **官方 dense GEMM 教程**，再考虑把 TMA/MMA 接回当前问题规模。

---

## 5. 命令行与代码入口

| 用途 | 命令 / 位置 |
|------|-------------|
| 默认 tiled 基准 | `python3 linear/cutedsl/benchmark_linear.py --m M --n N --k K` |
| 对照 naive | `… --kernel naive` |
| 机器可读一行（给 compare） | `--machine` |
| Hello | `linear/cutedsl/hello_cutedsl_linear.py`（tiled + pad） |
| 三后端对比 | `python3 linear/compare.py` |

---

## 6. 后续可优化方向（未实现）

按投入与收益大致排序，供后续迭代：

1. **双缓冲 / 多 stage smem**：重叠全局读与计算（需更仔细的同步与流水线）。
2. ~~**向量化全局加载**~~：**已实现** 为每线程 4×fp32 的 `cute.copy` 路径（见 §3.3）；若仍要 TMA/`cp.async`/显式 128-bit，可再迭代。
3. **更大 BK 或调整 BM/BN**：在 smem 容量与寄存器压力下扫参。
4. **架构专用路径**：Hopper+ 使用 TMA + WGMMA 等（直接对照 `cutlass` 官方示例）。

---

## 7. 小结

| 阶段 | 要点 |
|------|------|
| Naive | 正确、易读；每点一线程 + 全长 K 循环，性能差。 |
| Tiled | `Wt` 布局 + CTA smem tile + 线程子 tile + `sync_threads`；注意 **JIT 内用字面量避免闭包**。 |
| 指标 | Pad 执行、原始 MNK 计 FLOPs，与 compare 其它列一致。 |

若本文与实现漂移，以 `linear/cutedsl/benchmark_linear.py` 与 `linear/cutedsl/README.md` 为准，并建议在本文件末尾 **更新日期** 或补一节变更记录。
