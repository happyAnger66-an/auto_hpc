---
name: auto-loop-hpc
description: >-
  Implements GPU operators conforming to the hpc_bench three-file contract
  (definition.json, workload.jsonl, solution.json): correct run() argument
  order, destination_passing_style (DPS) vs return-value style, per-backend
  entry points (pytorch, triton, cuda_cpp, cudnn, cutlass, cute_dsl, cutile),
  and numerical parity with the embedded Python reference. Bundled references/
  provides runnable templates (rmsnorm multi-backend, elementwise_add). Use
  when adding or porting kernels for hpc-bench or sol-execbench, or when the
  user mentions definition/workload/solution JSON, DPS, SupportedLanguages, or
  hpc_bench benchmarks. 适用于按 hpc_bench 三件套实现或迁移算子、修正 run
  签名与数值对比失败。
---

# 按 hpc_bench 接口实现算子（可 benchmark）

## Quick start

1. 打开与本 skill 同级的 **`references/<算子类型>/`**（或已克隆的 **`hpc_bench/examples/`**）作为目录模板。
2. **先** 定好或修改 **`definition.json`**（含 `reference` 字符串）与 **`workload.jsonl`**，再写 **`solution.json`** 与内核源码。
3. 保证 **`run` 形参顺序** = `definition.inputs` 键顺序；若 DPS，再按 `definition.outputs` 键顺序追加输出张量形参。
4. 用 `references/` 中对应后端复制 **`solution.json`** 的 `spec.languages`、`entry_point`、`sources`，只替换实现。
5. 在具备 GPU 的环境执行 **`hpc-bench <problem_dir> --solution ...`**（或 **`sol-execbench`** / 相关 **`pytest`**），**通过后再交付**。

### 术语（勿混用）

| 说法 | 含义 |
|------|------|
| **`hpc-bench`** | **`hpc_bench` 提供的 CLI 命令**（评测入口之一）。 |

下文 **`references/`** 均指 **与本 `SKILL.md` 同级**的目录，不依赖工作区是否挂载 `hpc_bench` 仓库。

本 skill 指导在 **hpc_bench** 契约下实现算子，使 CLI 能加载、**数值对比参考实现**并**计时**。**优先对照 `references/`**（与契约一致的可跑样例），勿自拟 `run(...)` 形参顺序或 `solution.json` 字段。

### 本 skill 附带的 `references/`（按算子类型分目录）

路径：**`references/`**（与 `SKILL.md` 同级）。更细的说明、与上游同步注意点见 **[references/README.md](references/README.md)**（渐进展开，避免本文件过长）。

| 目录 | 说明 |
|------|------|
| **`references/rmsnorm/`** | 归约 + **标量** `eps`、多后端齐全：`pytorch`、`triton`、`cuda_cpp`、`cudnn`、`cutlass`、`cute_dsl`、`cutile`。 |
| **`references/elementwise_add/`** | **仅张量**输入（无标量）、逐元素加；含 `pytorch` 与 `triton`，可按 `rmsnorm/` 结构扩展 C++ 族后端。 |
| **`references/README.md`** | 上表摘要；若与上游漂移，以 `hpc_bench` 为准并可再同步 `references/rmsnorm/`。 |

**上游对照**（架构与枚举）：在已克隆的 **`hpc_bench`** 仓库中查看 `examples/`、`docs/arch.md`、`src/hpc_bench/core/data/solution.py`。

---

## 1. 契约总览（三件套）

| 产物 | 作用 |
|------|------|
| `definition.json` | **唯一契约**：`axes`、`inputs`、`outputs`（名字、shape、dtype）、以及字符串形式的 **`reference`**（Python `run`）。 |
| `workload.jsonl` | 每行一个评测用例：`uuid`、`axes`、`inputs`（random/scalar/safetensors/…）、`tolerance`。 |
| `solution.json` | 你的实现：`spec.languages`、`spec.entry_point`、`spec.destination_passing_style`、`sources`；`solution.definition` **必须等于** `definition.name`。 |

**禁止**：在 `solution` 里写与 `definition.inputs`/`outputs` 不一致的形参顺序或 dtype；禁止 Python 与 C++ `languages` 混写在同一 `solution.json`（schema 校验会失败）。

---

## 2. `run` 的调用方式（最重要）

框架在 `problem_packager` 中大致等价于：

- **非 DPS**：`ref_fn(*inputs)`；`user_fn(*inputs)`。返回值再按 `definition.outputs` 的 **key 顺序** 与参考输出逐张量比对。
- **DPS（`destination_passing_style: true`，与 `examples` 一致）**：`user_fn(*inputs, *outputs)`，其中 **`outputs` 由框架预分配**，算子 **原地写入**，无返回值。

### 2.1 `*inputs` 的顺序

`inputs` 列表顺序 = **`definition.json` 里 `inputs` 对象中键的声明顺序**（与 JSON 中字段顺序一致；`gen_inputs` 按 `definition.inputs.items()` 迭代）。

- 张量：CUDA 上 `torch.Tensor`（dtype/shape 与契约一致）。
- 标量（`shape: null`）：Python 标量（如 `float`），来自 workload 的 `ScalarInput`。

**示例**（RMSNorm）：契约里顺序为 `input` → `weight` → `eps`，则：

```python
def run(input, weight, eps, output):  # DPS：最后一个是 output
    ...
```

```cpp
void run(torch::Tensor input, torch::Tensor weight, float eps, torch::Tensor output);
```

### 2.2 `*outputs` 的顺序（仅 DPS）

输出缓冲区顺序 = **`definition.json` 里 `outputs` 键的声明顺序**（`allocate_outputs` 使用 `definition.outputs` 的顺序）。多数算子只有一个 `output`；多输出时 **最后一个形参对应 JSON 中最后一个 output 键**（按顺序追加）。

---

## 3. `reference` 与实现的一致性

- **参考实现**必须是可执行 Python 源码字符串：`import torch` 等写在字符串内；函数名 **`run`**。
- **非 DPS**：`run(...)` 返回与 `outputs` 对应的张量或元组（与框架解析方式一致）。
- **DPS**：参考实现常写作 **`return`** 张量；用户算子仍用 **原地写 `output`**。两者数值语义必须一致（同一数学定义、注意 **fp32 中间累加** 与 `dtype` 对齐，否则易 `INCORRECT_NUMERICAL`）。

---

## 4. `solution.json` 与后端选择

`spec.languages` 只能包含 **同一侧**语言（全 Python 族或全 C++ 族），且与 **`entry_point` 文件后缀**一致：

| `languages` | `entry_point` 文件 | 其他常见字段 |
|-------------|---------------------|--------------|
| `pytorch` | `kernel.py::run` | 无 `binding` |
| `triton` | `kernel.py::run` | 依赖 Triton |
| `cute_dsl` | `kernel.py::run` | CuTe / cutlass Python 栈 |
| `cutile` | `kernel.py::run` | `import cuda.tile` |
| `cuda_cpp` | `kernel.cu::run`（推荐设备代码放 `.cu`） | `"binding": "torch"`，`compile_options` 可选 |
| `cutlass` | `kernel.cu::run` | 同上；可选环境变量 `CUTLASS_PATH` |
| `cudnn` | `kernel.cu::run` | `binding` + `ld_flags` 含 `-lcudnn` 等 |

- **`target_hardware`**：含 `LOCAL` 时，驱动会为扩展编译追加 `-gencode=arch=compute_XX,code=sm_XX`。
- **`sources`**：`path` 相对 **`solution.json` 所在目录**；可省略 `content`，由 CLI 读盘。
- **C++ 扩展**：需提供 **`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`** 并把入口 `m.def("run", ...)` 与 `entry_point` 中符号一致（见 `references/rmsnorm/cuda_cpp/kernel.cu`）。

Schema 另有 `cudnn_frontend`、`cublas` 等枚举；未在用户需求中列出时勿随意混用，除非已核对 `SupportedLanguages` 与驱动编译路径。

---

## 5. 推荐目录布局（与 `references/rmsnorm` 一致）

```
<problem>/
├── definition.json
├── workload.jsonl
├── pytorch/solution.json + kernel.py
├── triton/solution.json + kernel.py
├── cuda_cpp/solution.json + kernel.cu
├── cudnn/ ...
├── cutlass/ ...
├── cute_dsl/ ...
└── cutile/ ...
```

评测命令形态：

```bash
hpc-bench <problem_dir> --solution <problem_dir>/<backend>/solution.json
```

---

## 6. 自检清单（写算子后必查）

- [ ] `definition.name` 与 `solution.definition` 一致。
- [ ] `run` 形参顺序 = `inputs` JSON 键顺序；DPS 下最后若干参数为 `outputs` 键顺序。
- [ ] 张量 `device`/`dtype`/`shape` 与契约一致；标量类型与 `workload` 一致。
- [ ] `destination_passing_style` 与参考/用户实现模式一致。
- [ ] Python：`languages` 仅 Python 族 + `entry_point` 以 `.py` 结尾。
- [ ] CUDA：`languages` 仅 C++ 族 + 设备代码在 `.cu`；`binding: "torch"`。
- [ ] `sources` 列出 `entry_point` 中的文件；路径无 `..`、非绝对路径。
- [ ] 数值路径与 `reference` 对齐（尤其 bf16 + fp32 归约顺序）。

### 评测失败时（优先排查）

- **`INCORRECT_NUMERICAL`**：先对齐 **`reference` 的数学定义与累加精度**（如 fp32 中间累加、归约轴、`dtype`），勿先改签名。
- **编译 / schema / 导入错误**：核对 **`solution.json`** 与 **`references/`** 同后端模板字段是否一致。
- **`run` 相关 TypeError/参数个数**：重新对照 **`definition.json` 中 `inputs`/`outputs` 的键顺序**（§2）。

---

## 7. 对 Agent 的硬性要求

1. **先写或改 `definition.json` 与 `reference`，再写 `solution`**，禁止与契约脱节的签名。
2. **以 `references/<算子类型>/` 为模板**复制目录结构，再替换数学实现。
3. 用户指定后端时：**只改 `languages` + 源码文件 + `entry_point`**，不擅自改 `inputs`/`outputs` 顺序。
4. 完成后在具备 GPU 的环境运行 **`hpc-bench`** / 相关 **`pytest`**，确认 **`PASSED`** 再交付；失败时按 §6.1 迭代。

---

## 8. 相关资源

- **契约样例（本 skill，不依赖其他仓库）**：`references/`、[references/README.md](references/README.md)。
