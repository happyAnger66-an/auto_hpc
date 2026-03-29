# cp.async 微基准（原生 CUDA）

与 `cutedsl/benchmark_linear.py` 无关的**最小**程序：单 block、每线程一条 **4 字节** `cp.async.ca.shared.global`，随后 `cp.async.commit_group` 与 `cp.async.wait_group 0`，再把 shared 写回 global，与 host 填充对比；并打印与「同步标量写 smem」kernel 的计时。

## 构建（4070 / Ada，sm_89）

```bash
cd auto_hpc/linear/cpasync_microbench
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

其它 GPU：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80   # 例：Ampere
```

## 运行

```bash
./build/cpasync_verify
./build/cpasync_verify --iters 5000 --warmup 100 --n 256
```

退出码 0 表示 cp.async 结果与参考一致。
