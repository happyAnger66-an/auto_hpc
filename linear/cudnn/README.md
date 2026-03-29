# cuDNN Linear 基准

`linear_bench`：用 **1×1 卷积**（NCHW `[M,K,1,1]` × 滤波器 `[N,K,1,1]`）计算 `X @ W^T`，再用 **`cudnnAddTensor`** 加 bias。

```bash
cd auto_hpc/linear/cudnn
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/linear_bench 1024 1024 1024 10 100
```

统一对比（自 `auto_hpc` 根目录）：

```bash
python3 linear/compare.py --cudnn-exe linear/cudnn/build/linear_bench
```
