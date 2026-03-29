# cuBLAS：element-wise add 基准

`C = A + B`（FP32，`M×N` 行主序连续存储）通过 **`cublasScopy` + `cublasSaxpy`** 实现：`C ← A`，再 `C ← C + B`。

cuBLAS 经典接口没有“两向量相加写入第三向量”的单调用，这是常见写法。

## 编译

```bash
cd auto_hpc/element-wise/cublas
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

可执行文件：`build/elementwise_add_bench`

## 运行

```bash
./build/elementwise_add_bench 4096 4096 10 100
```

输出：`OK ms_per_iter ... gflops ... method cublas_scopy_saxpy`（GFLOPS = M×N / time_s / 1e9）

## 统一对比

```bash
cd auto_hpc
python3 element-wise/compare.py --cublas-exe element-wise/cublas/build/elementwise_add_bench
```
