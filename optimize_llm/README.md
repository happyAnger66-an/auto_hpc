# optimize_llm

本目录提供基于 **Triton** 与 **CuTeDSL** 的 FMHA（多头注意力）实现及 **TensorRT Python 插件**（`tensorrt.plugin`）注册，并附带可插件化的 benchmark 框架。

## 依赖概要

- **CUDA** 与 **PyTorch（CUDA 版）**：benchmark 与算子均在 GPU 上运行。
- **Triton backend**：需能 `import triton`；`head_dim` 支持 32 / 64 / 128 / 256。
- **CuTeDSL backend**：需能 `import cutlass.cute`（与当前环境匹配的 `nvidia-cutlass-dsl` 等）。
- 导入 **`optimize_llm.backend.triton`** 时会加载 TensorRT 插件模块，终端可能出现 `tensorrt.plugin` experimental 提示，属正常现象。

## Benchmark 框架能力

- 内置 backend：`pytorch`、`triton`、`cutedsl`。
- 动态 backend 加载：支持从任意模块路径按 `module.path:Symbol` 方式导入。
- 配置方式：支持 **YAML + CLI**，CLI 可覆盖 YAML 字段。
- 支持多组 case 批量执行，并输出跨 case 汇总统计。

## 运行方式

在仓库根目录执行：

```bash
cd /path/to/auto_hpc
python3 -m optimize_llm.benchmark --help
```

### 1) 纯 CLI（单 case 或多 `--case`）

默认内置 backend：

```bash
python3 -m optimize_llm.benchmark --backends all
```

指定参数：

```bash
python3 -m optimize_llm.benchmark --backends triton,cutedsl --B 2 --H 8 --L 128 --D 64 --dtype fp16
```

多 case（可重复传 `--case`）：

```bash
python3 -m optimize_llm.benchmark \
  --backends pytorch,triton \
  --case B=1,H=4,L=64,D=64,dtype=fp16,causal=false,seed=0 \
  --case B=1,H=8,L=128,D=128,dtype=fp16,causal=true,seed=1
```

### 2) YAML 配置

示例配置见：`optimize_llm/benchmark/examples/fmha.yaml`

```bash
python3 -m optimize_llm.benchmark --config optimize_llm/benchmark/examples/fmha.yaml
```

CLI 覆盖 YAML：

```bash
python3 -m optimize_llm.benchmark \
  --config optimize_llm/benchmark/examples/fmha.yaml \
  --iters 50 --warmup 10 --backends triton,cutedsl
```

### 3) 动态加载任意路径 backend

通过 `--backend-targets` 传入插件目标（`module.path:Symbol`）：

```bash
python3 -m optimize_llm.benchmark \
  --backends pytorch \
  --backend-targets my_pkg.my_backend:MyFmhaBackend
```

YAML 里也可以写：

```yaml
backends:
  - builtin: pytorch
  - target: my_pkg.my_backend:MyFmhaBackend
    alias: my_backend
```

> 插件对象需提供：`name`、`supported(workload)`、`prepare(ctx)`、`run(ctx)`。

### 4) 外部仓库插件最小模板

仓库内提供了可直接参考/复制的模板：

- Python 模板：`optimize_llm/benchmark/examples/external_backend_template.py`
- YAML 模板：`optimize_llm/benchmark/examples/external_plugin.yaml`

本地快速验证模板：

```bash
python3 -m optimize_llm.benchmark --config optimize_llm/benchmark/examples/external_plugin.yaml
```

跨仓库使用时，你只需要：

1. 将模板类复制到外部仓库并修改 `run()` 为你的算子调用；
2. 确保该模块可被当前 Python 解释器导入（安装包或设置 `PYTHONPATH`）；
3. 在 YAML/CLI 中将 `target` 改为外部路径 `your_pkg.your_module:YourBackendClass`。

### 5) 同时对比多个 backend（builtin + external）

你可以在一次 benchmark 中混合多个 backend（内置 + 外部插件）进行横向对比。

#### CLI 方式（推荐快速试验）

```bash
python3 -m optimize_llm.benchmark \
  --backends pytorch,triton,cutedsl \
  --backend-targets your_pkg.my_backend:MyFmhaBackend \
  --B 1 --H 8 --L 128 --D 128 --dtype fp16 \
  --check-cases 10
```

说明：
- `--backends`：内置 backend 列表（或 `all`）。
- `--backend-targets`：外部插件列表，格式 `module.path:Symbol`，多个用逗号分隔。
- 输出表会把所有 backend 放在同一张表中，便于直接对比 `mean_ms` 与误差指标。

#### YAML 方式（推荐做可复现实验）

```yaml
op: fmha
backends:
  - builtin: pytorch
  - builtin: triton
  - builtin: cutedsl
  - target: your_pkg.my_backend:MyFmhaBackend
    alias: my_backend
run:
  warmup: 5
  iters: 20
  check: true
  check_cases: 10
cases:
  - B: 1
    H: 8
    L: 128
    D: 128
    dtype: fp16
    causal: true
    seed: 0
    device: cuda
```

运行：

```bash
python3 -m optimize_llm.benchmark --config /path/to/bench_mix.yaml
```

#### YAML + CLI 混合覆盖

先用 YAML 固定实验配置，再临时改 backend 组合：

```bash
python3 -m optimize_llm.benchmark \
  --config /path/to/bench_mix.yaml \
  --backends pytorch,triton \
  --backend-targets your_pkg.my_backend:MyFmhaBackend \
  --iters 50
```

这会用 CLI 覆盖 YAML 的 `backends` 与 `iters`，其余配置保持不变。

## 参数说明（核心）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 无 | YAML 配置路径。 |
| `--backends` | `all`(CLI模式) | 内置 backend 名称列表。 |
| `--backend-targets` | 无 | 动态插件目标列表（`module:Symbol`）。 |
| `--case` | 无 | 追加 case（`key=value,...`）。 |
| `--B --H --L --D --dtype --seed --device --causal` | - | 单 case 参数（未使用 `--case` 时生效）。 |
| `--warmup` | `5` | 预热轮数。 |
| `--iters` | `20` | 计时轮数。 |
| `--check-cases` | `5` | 正确性随机样本组数。 |
| `--no-check` | 关闭 | 关闭与 PyTorch 参考的误差检查。 |

## 输出与判定

- 每个 case 输出一张表：`mean_ms/median_ms/stdev_ms` + `max_abs_mean/max_abs_max/mae_mean/mse_mean`。
- 多 case 时输出 `Summary Across Cases`。
- 退出码：
  - `0`：成功且误差不过阈值；
  - `1`：存在 backend 误差超阈值；
  - `2`：配置或插件加载错误。

## 目录关系

- 算子实现：`optimize_llm/backend/triton/`、`optimize_llm/backend/cuteDSL/`。
- benchmark 框架：`optimize_llm/benchmark/`（入口 `python3 -m optimize_llm.benchmark`）。
