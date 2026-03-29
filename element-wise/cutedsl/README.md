# CuTeDSL 入门示例

本目录为 `element-wise/` 下的 CuTeDSL 实现。

- `hello_cutedsl_elementwise_add.py`：自定义 CuTeDSL kernel，计算 `C = A + B`

## 1) 环境准备

建议先在 CUTLASS 仓库下完成 CuTeDSL 安装（按官方脚本）：

```bash
cd /path/to/cutlass/python/CuTeDSL
bash setup.sh
```

CUDA 13 可尝试：`bash setup.sh --cu13`

## 2) 运行示例

```bash
cd /path/to/auto_hpc/element-wise/cutedsl
python3 hello_cutedsl_elementwise_add.py --m 128 --n 256
```

## 2.1) 布局扫描基准（thr_layout / val_layout）

`benchmark_layouts.py` 会编译多组 `(thr_m, thr_n)`、`(val_m, val_n)`，用 CUDA Event 统计 **ms/iter** 与 **GFLOPS**（每输出元素 1 次加法 → M×N FLOPs）。

```bash
cd /path/to/auto_hpc/element-wise/cutedsl
python3 benchmark_layouts.py --m 4096 --n 4096 --warmup 10 --iters 100
```

某一组若 **编译失败** 或 **数值校验失败**，会在表格 `note` 列给出原因。

扩展更多 `(thr, val)` 时：在 `benchmark_layouts.py` 中按现有模式增加 `_add_entry_layout_XX`，并把函数加入 `ADD_ENTRY_FUNCS`，同时调整 `default_candidates()` 的去重顺序使 `layout_id` 与元组下标一致。

## 与 cuBLAS / cuDNN 对比

在 `element-wise/` 下运行统一入口（指标均为 GFLOPS）：

```bash
python3 ../compare.py --m 4096 --n 4096
```

（自 `auto_hpc` 根目录则为 `python3 element-wise/compare.py`。）

## 3) 学习重点

- `@cute.kernel` 定义设备端 kernel
- `@cute.jit` 定义 host 侧 launch 封装
- `cute.make_tiled_copy_tv` + `cute.copy` 做拷贝
- `cute.compile(...)` JIT 编译并执行
- `torch.testing.assert_close` 做数值校验
