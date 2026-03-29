# auto_hpc

在 GPU 上对比多种实现路径的 **高性能算子** 与 **统一 GFLOPS 口径**，覆盖 **逐元素加法** 与 **全连接 / 矩阵乘加（Linear）**。代码按算子分目录，同一算子下再按 **CuTeDSL、cuBLAS、cuDNN、手写 CUDA** 等分子工程，便于横向对照与扩展。

## 仓库结构

| 路径 | 说明 |
|------|------|
| `element-wise/` | 逐元素 `C = A + B`（FP32） |
| `linear/` | `Y = X @ W^T + b`（FP32） |
| `docs/` | 补充说明（如 Linear 与 CuTeDSL 优化相关笔记） |

各算子目录内另有 `README.md`，含编译命令与指标定义。

## 当前支持的计算

以下均为 **FP32**；性能统一为 **GFLOPS**（各子目录 README 中给出 FLOPs 公式）。

### 逐元素加法 `C = A + B`

| 实现 | 形式 | 说明 |
|------|------|------|
| CuTeDSL | Python | 示例与布局扫描 |
| cuBLAS | CUDA + CMake | `Scopy` + `Saxpy` |
| cuDNN | CUDA + CMake | `OpTensor` ADD |

一键对比（需先按 `element-wise/README.md` 编译 C++ 基准）：

```bash
python3 element-wise/compare.py --m 4096 --n 4096 --warmup 10 --iters 100
```

根目录转发：`python3 compare_elementwise_add.py ...`

### 线性层 / GEMM + bias `Y = X @ W^T + b`

| 实现 | 形式 | 说明 |
|------|------|------|
| CuTeDSL | Python | 块 tile + smem GEMM，可与 `torch.nn.functional.linear` 对齐校验 |
| cuBLAS | CUDA + CMake | 列主序 `Sgemm` + bias |
| cuDNN | CUDA + CMake | 1×1 卷积等价 matmul + bias |
| `cpasync_microbench/` | CUDA + CMake | 独立 **`cp.async`** 微基准（与 PyTorch 无关） |
| `cuda_cpasync_linear/` | PyTorch JIT 扩展 | 双槽共享内存 + **16B `cp.async`** 预取 tiled Linear，pad/校验与 CuTeDSL 侧约定一致 |

一键对比 **CuTeDSL / cuBLAS / cuDNN**（不含 cp.async 扩展；需先编译 `linear/cublas` 与 `linear/cudnn`）：

```bash
python3 linear/compare.py --m 1024 --n 1024 --k 1024 --warmup 10 --iters 100
```

根目录转发：`python3 compare_linear.py ...`

`cuda_cpasync_linear` 单独运行见 `linear/cuda_cpasync_linear/README.md`。

## 依赖（概要）

- **CUDA**、对应架构下的 **nvcc**
- **CuTeDSL**：需本机已按 CUTLASS 文档安装 `python/CuTeDSL`（与 `element-wise/cutedsl`、`linear/cutedsl` 一致）
- **PyTorch（CUDA 版）**：`cuda_cpasync_linear` 的 JIT 扩展与 CuTeDSL 基准中的 PyTorch 校验
- **cuBLAS / cuDNN**：CMake 子工程链接系统或指定 `CUDNN_ROOT` 等（见各子目录 README）

## 构建产物与 Git

所有名为 `build/` 的目录（任意层级）已写入 `.gitignore`，请在本机执行 `cmake -B .../build` 生成，勿提交二进制与 CMake 缓存。
