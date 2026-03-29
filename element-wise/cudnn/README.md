# cuDNN：element-wise add 基准

`C = A + B`（FP32）使用 **`cudnnOpTensor`**，`CUDNN_OP_TENSOR_ADD`，系数 `α1=α2=1`、`β=0`。

张量描述为 **NCHW** 形状 `1×1×M×N`，与行主序 `M×N` 连续内存布局一致。

## 依赖

需安装 **cuDNN**。CMake 会按顺序查找：

- **发行版包**（如 `libcudnn9` / `libcudnn9-dev`）：头文件常在 `/usr/include/x86_64-linux-gnu`，库在 `/usr/lib/x86_64-linux-gnu/libcudnn.so.9`（无需再设 `CUDNN_ROOT`）。
- **NVIDIA 压缩包**：设置 `-DCUDNN_ROOT=/path/to/cudnn`，或环境变量 `CUDNN_PATH` / `CUDNN_ROOT`。

## 编译

```bash
cd auto_hpc/element-wise/cudnn
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

可执行文件：`build/elementwise_add_bench`

## 运行

```bash
./build/elementwise_add_bench 4096 4096 10 100
```

输出含 **gflops**（与 `compare.py` 一致）。

## 统一对比

```bash
cd auto_hpc
python3 element-wise/compare.py --cudnn-exe element-wise/cudnn/build/elementwise_add_bench
```
