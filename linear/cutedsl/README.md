# CuTeDSL Linear

- `benchmark_linear.py`：默认 **64×64×16 块 tile + 共享内存 GEMM**，权重侧使用 **`Wt = W.T`（`[K,N]`）** 以改善全局内存合并；`--kernel naive` 可切回每输出点 K 循环的对照实现。
- `hello_cutedsl_linear.py`：小形状正确性演示（tiled 路径）

**实践要点（与 CUTLASS/CuTe 习惯一致）**

1. **`@cute.kernel` / `@cute.jit` 内不要用模块级变量当 tile 尺寸**（会生成闭包，`cute.compile` 可能报 `free vars`）；BM/BN/BK 在 kernel 内写成 **字面量**。
2. **共享内存**用 `cutlass.utils.SmemAllocator` + **`cute.make_layout` 静态 shape**；tile 间 **`cute.arch.sync_threads()`**。
3. **寄存器累加**可用 `cute.make_rmem_tensor((4,4), dtype)` 表示每线程子 tile，避免手写 16 个标量。
4. **布局**：`Y = X @ W^T` 在内存上实现为 `X @ Wt`（`Wt` 为 `[K,N]` 行主序），与经典 GEMM 的 A、B 访问模式一致。

语义与 `torch.nn.functional.linear(x, w, b)` 一致：`x [M,K]`, `w [N,K]`, `b [N]`（内部对 `w` 转置并 pad）。

```bash
cd auto_hpc/linear/cutedsl
python3 benchmark_linear.py --m 512 --n 512 --k 512
python3 benchmark_linear.py --kernel naive --m 256 --n 256 --k 256   # 对照
python3 hello_cutedsl_linear.py
```

统一对比：

```bash
python3 ../compare.py
```
