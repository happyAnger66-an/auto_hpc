# CuTeDSL Linear

- `benchmark_linear.py`：默认 **64×64×16 块 tile + 共享内存 GEMM**，权重侧使用 **`Wt = W.T`（`[K,N]`）** 以改善全局内存合并；`--kernel naive` 可切回每输出点 K 循环的对照实现。`--pipeline double` 为 **双槽 smem**（`linear_tiled_kernel_double_buffer`），供流水线实验；**同步拷贝下 smem 加倍、占用率下降，GFLOPS 通常明显低于默认 `single`**，与 cuBLAS 对比时请用默认 pipeline。
- `hello_cutedsl_linear.py`：小形状正确性演示（tiled 路径）

**GPU 架构说明（例：RTX 4070，compute capability 8.9）**

- CUDA **8.9** 对应 CuTeDSL / NVPTX 目标 **Ada Lovelace**，常记为 **`sm_89`**（与 Ampere 同属 8.x 家族，驱动会映射兼容链如 `sm_89` → `sm_86` 等，以你本机 CuTeDSL 为准）。
- 本目录当前 **SIMT + smem tiled linear** 在 8.9 上可直接使用；**Hopper 起的 TMA / `cp.async.bulk.tensor` 等路径**在 DSL 里往往要求 **`sm_90` 及以上**，4070 上不可用，异步优化应走 **非 bulk 的 `cp.async`（如 `CopyG2SOp`）** 等 Ada 仍支持的指令，并注意 **指针对齐**（此前 128b 全局向量 async 在动态 stride 视图上易触 IR 校验失败）。
- 若 JIT 报架构不匹配或加载 kernel 失败，可显式设置环境变量（具体取值以 [CuTeDSL 文档](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/) 为准），例如：`export CUTE_DSL_ARCH=sm_89`。

**实践要点（与 CUTLASS/CuTe 习惯一致）**

1. **`@cute.kernel` / `@cute.jit` 内不要用模块级变量当 tile 尺寸**（会生成闭包，`cute.compile` 可能报 `free vars`）；BM/BN/BK 在 kernel 内写成 **字面量**。
2. **共享内存**用 `cutlass.utils.SmemAllocator` + **`cute.make_layout` 静态 shape**；tile 间 **`cute.arch.sync_threads()`**。
3. **寄存器累加**可用 `cute.make_rmem_tensor((4,4), dtype)` 表示每线程子 tile，避免手写 16 个标量。
4. **布局**：`Y = X @ W^T` 在内存上实现为 `X @ Wt`（`Wt` 为 `[K,N]` 行主序），与经典 GEMM 的 A、B 访问模式一致。

语义与 `torch.nn.functional.linear(x, w, b)` 一致：`x [M,K]`, `w [N,K]`, `b [N]`（内部对 `w` 转置并 pad）。

```bash
cd auto_hpc/linear/cutedsl
python3 benchmark_linear.py --m 512 --n 512 --k 512
python3 benchmark_linear.py --m 512 --n 512 --k 512 --pipeline double   # 双缓冲 tiled
python3 benchmark_linear.py --m 512 --n 512 --k 512 --pipeline double_cpasync32   # 占位：当前等同 double（真 cp.async 受阻，见 benchmark 注释）
python3 benchmark_linear.py --kernel naive --m 256 --n 256 --k 256   # 对照
python3 hello_cutedsl_linear.py
```

统一对比：

```bash
python3 ../compare.py
```
