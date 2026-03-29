# cuBLAS Linear 基准（列主序）

主机上 `X`、`W` 仍为行主序 `[M,K]`、`[N,K]`（与 PyTorch 一致）。`cublasSetMatrix` 将数据拷到 GPU 后为 **列主序**；`cublasSgemm(OP_N, OP_N, M, N, K)` 计算 `C = A_{M×K} B_{K×N}`，其中 `B` 为数学上的 `W^T`。

bias 在设备上为预展开的 `M×N` 列主序向量（每列常数为 `b[n]`），每次迭代用 `cublasSaxpy` 做 `Y += broadcast(b)`。

```bash
cd auto_hpc/linear/cublas
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/linear_bench 1024 1024 1024 10 100
```

对比脚本默认可执行文件：`linear/cublas/build/linear_bench`。

```bash
python3 ../compare.py --cublas-exe cublas/build/linear_bench
```
