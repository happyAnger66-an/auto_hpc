# Linear（Y = X @ W^T + b）

与 `element-wise/` 并列：按实现分子目录，统一 **GFLOPS** 对比。

| 子目录 | 内容 |
|--------|------|
| `cutedsl/` | CuTeDSL **块 tile + smem GEMM**（默认）或 `--kernel naive` 对照 + `benchmark_linear.py` |
| `cublas/` | CMake + `linear_bench.cu`：`cublasSgemm`（**设备列主序**）+ `cublasSaxpy` 加 bias |
| `cudnn/` | CMake + `linear_bench.cu`：1×1 卷积实现 matmul + `cudnnAddTensor` 加 bias |
| `cpasync_microbench/` | 独立 **cp.async**（4B）正确性 + 与同步 kernel 计时的最小 CUDA 程序（4070 用 `sm_89`） |
| `cuda_cpasync_linear/` | **PyTorch JIT 扩展**：双槽 smem + **16B cp.async** 预取的 tiled Linear，pad/校验与 `cutedsl` 一致；见目录内 `README.md` |

## 性能指标

- **FLOPs** = `2*M*N*K + M*N`（与 PyTorch `nn.Linear` 常见计法一致）
- **GFLOPS** = `FLOPs / time_s / 1e9`

## 编译 cuBLAS / cuDNN 基准

```bash
cmake -S linear/cublas -B linear/cublas/build -DCMAKE_BUILD_TYPE=Release
cmake --build linear/cublas/build -j

cmake -S linear/cudnn -B linear/cudnn/build -DCMAKE_BUILD_TYPE=Release
cmake --build linear/cudnn/build -j

cmake -S linear/cpasync_microbench -B linear/cpasync_microbench/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build linear/cpasync_microbench/build -j
./linear/cpasync_microbench/build/cpasync_verify

python3 linear/cuda_cpasync_linear/benchmark_cpasync_linear.py --m 1024 --n 1024 --k 1024
```

## 一键对比

```bash
# 仓库根目录
python3 linear/compare.py --m 1024 --n 1024 --k 1024 --warmup 10 --iters 100
```

或使用根目录转发：`python3 compare_linear.py ...`

## 依赖

- CuTeDSL：同 `element-wise/cutedsl`（CUTLASS `python/CuTeDSL` 安装）
- cuDNN：系统 `libcudnn` 或 `-DCUDNN_ROOT=`
