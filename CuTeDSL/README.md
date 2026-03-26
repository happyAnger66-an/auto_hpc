# CuTeDSL 入门示例

这个目录提供一个最小的 CuTeDSL "hello world"：

- `hello_cutedsl_elementwise_add.py`：自定义 CuTeDSL kernel，计算 `C = A + B`

## 1) 环境准备

建议先在 CUTLASS 仓库下完成 CuTeDSL 安装（按官方脚本）：

```bash
cd /home/zhangxa/codes/hpc/cutlass/python/CuTeDSL
bash setup.sh
```

如果你使用 CUDA 13 环境，可尝试：

```bash
bash setup.sh --cu13
```

## 2) 运行示例

```bash
cd /home/zhangxa/codes/hpc/auto_hpc/CuTeDSL
python hello_cutedsl_elementwise_add.py --m 128 --n 256
```

预期输出包含：

- `[OK] verify passed ...`
- 编译耗时和首次执行耗时

## 3) 学习重点

这个示例覆盖了 CuTeDSL 最关键的最小路径：

- `@cute.kernel` 定义设备端 kernel
- `@cute.jit` 定义 host 侧 launch 封装
- `cute.make_tiled_copy_tv` + `cute.copy` 做拷贝
- `cute.compile(...)` JIT 编译并执行
- `torch.testing.assert_close` 做数值校验

