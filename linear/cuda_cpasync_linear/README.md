# CUDA 原生双槽 smem + 16B `cp.async` Linear

与 `cutedsl/benchmark_linear.py` **相同的 pad 规则**（`pad_linear_tensors`）与 **BM/BN/BK=64/64/16** 分块；预取路径使用内联 PTX `cp.async.ca.shared.global`（**16B / float4**），`commit` 后与当前条带的 MMA **重叠**，条带结束后再 `wait_group 0` + `syncthreads`。

## 依赖

- PyTorch（CUDA 版）、nvcc，架构默认 **sm_89**（4070）。其它 GPU：

```bash
export CUDA_CPASYNC_ARCH=80   # 示例
```

## 运行

```bash
cd auto_hpc/linear/cuda_cpasync_linear
python3 benchmark_cpasync_linear.py --m 1024 --n 1024 --k 1024
```

首次会 JIT 编译扩展，缓存于 `~/.cache/torch_extensions/`。

## 调试：仅同步 float4 加载（无 cp.async）

用于对照 GEMM 索引是否正确（编译宏 `CPASYNC_LINEAR_USE_SYNC_LOAD`）：

```bash
CPASYNC_LINEAR_USE_SYNC_LOAD=1 python3 benchmark_cpasync_linear.py --m 512 --n 512 --k 512
```

注意：环境变量必须写在 **`python3` 同一命令前**，不要写成 `VAR=1 cd dir && python3 ...`（`VAR` 不会传给 Python）。

## 与 CuTeDSL 的关系

此处不依赖 CuTeDSL；数值与 `torch.nn.functional.linear` 对齐，可与 `cutedsl/benchmark_linear.py` 对比 GFLOPS。
