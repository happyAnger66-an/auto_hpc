# Element-wise Add（C = A + B）

本目录按**实现方式**分子目录，便于对照与扩展。

| 子目录 | 内容 |
|--------|------|
| `cutedsl/` | CuTeDSL 示例 `hello_cutedsl_elementwise_add.py`、布局扫描 `benchmark_layouts.py` |
| `cublas/` | `cublasScopy` + `cublasSaxpy` 基准（CMake + `.cu`） |
| `cudnn/` | `cudnnOpTensor` ADD 基准（CMake + `.cu`） |

## 统一性能指标：GFLOPS

三种实现均报告：

- **GFLOPS** = `(M × N) / time_seconds / 10^9`
- 算术量：每个输出元素 **1 次浮点加法** → **M×N FLOPs**（与 `benchmark_layouts`、两个 C++ 基准一致）。

## 一键对比

```bash
# 自仓库根目录 auto_hpc（或本目录上级）
cmake -S element-wise/cublas -B element-wise/cublas/build -DCMAKE_BUILD_TYPE=Release
cmake --build element-wise/cublas/build -j

cmake -S element-wise/cudnn -B element-wise/cudnn/build -DCMAKE_BUILD_TYPE=Release
cmake --build element-wise/cudnn/build -j

python3 element-wise/compare.py --m 4096 --n 4096 --warmup 10 --iters 100
```

## 兼容旧入口

仓库根目录的 `compare_elementwise_add.py` 会转发到 `element-wise/compare.py`。
