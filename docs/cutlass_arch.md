# CUTLASS 代码架构分析

本文档基于 `../cutlass`（即 `/home/zhangxa/codes/hpc/cutlass`）代码进行梳理，面向在 `auto_hpc` 工作区内快速理解 CUTLASS 4.x 的模块边界与调用链路。

## 1. 总体架构

CUTLASS 本质上是一个 **header-only 的 CUDA C++ 模板库**，外围配套了可选构建目标：

- `include/`：核心库（`cutlass` + `cute`）
- `tools/`：库实例化、性能测试与通用工具
- `examples/`：从基础到新架构特性的示例集合
- `test/`：单元测试与自包含头文件检查
- `python/`：Python 接口、代码生成器、CuTe DSL

在根 `CMakeLists.txt` 中，按开关组合挂接：

- `CUTLASS_ENABLE_TOOLS` -> `add_subdirectory(tools)`
- `CUTLASS_ENABLE_EXAMPLES` -> `add_subdirectory(examples)`
- `CUTLASS_ENABLE_TESTS` -> `add_subdirectory(test)`

这意味着核心库可被独立 include 使用，而工具/示例/测试是“附加构建层”。

## 2. 核心头文件层（include）

## 2.1 `include/cutlass`：算法与算子模板层

`include/cutlass` 目录承载传统 CUTLASS C++ API 与算子实现，主要分层如下：

- `arch/`：硬件/指令级能力封装（Tensor Core、异步拷贝、架构特性）
- `gemm/`：GEMM 主体（device/kernel/threadblock/warp/thread 多层）
- `epilogue/`：GEMM/Conv 后处理（激活、融合、写回策略等）
- `conv/`：卷积体系（implicit GEMM、direct conv、fprop/dgrad/wgrad）
- `layout/`、`transform/`：布局与数据变换
- `reduction/`：非 GEMM 模式的归约
- `thread/`、`pipeline/`：线程级算子与流水线基础设施

其中 `gemm/kernel`、`epilogue/collective`、`conv/kernel` 下可看到大量按架构命名文件：

- `sm70_*`、`sm90_*`、`sm100_*`、`sm103_*`、`sm120_*`

这体现了 CUTLASS 的核心策略：**统一模板接口 + 分架构特化实现**。

## 2.2 `include/cute`：底层抽象代数层

`cute` 是 CUTLASS 3.x 之后引入的底层抽象核心，提供：

- 形状/步长/布局/张量等基础类型（`layout.hpp`、`tensor.hpp` 等）
- `arch/` + `atom/`：PTX 原子操作与元信息
- `algorithm/`：copy、gemm 等组合算法

可理解为：`cute` 负责“表达与组合”，`cutlass` 负责“算子工程化与设备级封装”。

## 3. GEMM/Conv 的纵向组织方式

以 GEMM 为代表，CUTLASS 采用分层分工：

1. **Device 层**：对外可调用接口（如 `gemm/device/*`）
2. **Kernel 层**：网格级/CTA 级 kernel 结构（如 `gemm/kernel/*`）
3. **Collective/Threadblock/Warp/Thread 层**：进一步拆分数据搬运、MMA、epilogue 等职责
4. **Arch 层**：映射到底层指令与硬件能力

同一套路也用于 Conv（`conv/device`、`conv/kernel`、`conv/collective`）。

这使得 CUTLASS 可以在不同架构与数据类型上重用同一设计骨架，仅替换策略模板与特化组件。

## 4. 构建系统架构（CMake）

根 `CMakeLists.txt` 的关键职责：

- 从 `include/cutlass/version.h` 读取版本号
- 统一 C++17 / CUDA17 编译设置
- 按 CUDA 版本动态确定 `CUTLASS_NVCC_ARCHS_SUPPORTED`
- 通过缓存变量控制构建规模（examples/tools/library/tests/profiler）

架构编译能力（`CUTLASS_NVCC_ARCHS`）覆盖从 Volta 到 Blackwell/后续变体（如 `90a`、`100a`、`120f` 等），体现了 CUTLASS 的多代 GPU 兼容策略。

## 5. Tools 层：从模板到可执行生态

`tools/CMakeLists.txt` 显示工具链分为三部分：

- `tools/util`：公用工具与辅助头文件
- `tools/library`：实例库生成（将模板实例化为可枚举/可调用操作）
- `tools/profiler`：命令行性能与正确性测试入口

并且 profiler 依赖 library（未启用 library 时会报构建冲突）。  
说明 CUTLASS 的 benchmarking/profiling 不是直接“裸调用模板”，而是基于实例库统一调度。

## 6. Examples 与 Test 层

## 6.1 `examples/`

`examples/CMakeLists.txt` 通过 `cutlass_example_add_executable()` 批量注册示例，覆盖：

- 早期基础示例（`00`~`39`）
- Hopper/Blackwell 等新架构特性（`48+`、`70+`、`90+`、`111/112`）
- GEMM、Conv、FMHA、MoE、低精度/块缩放等场景

示例统一链接 `CUTLASS` 与 `cutlass_tools_util_includes`，是“最佳实践入口层”。

## 6.2 `test/`

`test/CMakeLists.txt` 结构较简洁：

- `test/unit`：GoogleTest 单元测试
- `test/self_contained_includes`：头文件自包含性检查

其中 unit tests 目录按模块细分（`gemm`、`conv`、`epilogue`、`cute`、`reduction`、`pipeline` 等），对应核心命名空间，保证模板组合的行为稳定性。

## 7. Python 侧架构

`python/` 不是单一包，而是多组件协作：

- `cutlass_cppgen`：Python 高层接口（运行/生成 CUTLASS kernel）
- `cutlass_library`：枚举与生成 C++ kernel 的工具模块（也被 CMake 侧使用）
- `CuTeDSL/`：CuTe Python DSL（4.x 重点）
- `pycute/`：与 CuTe 相关的 Python 组件

`python/CuTeDSL/cutlass` 下进一步拆分：

- `base_dsl/`：基础 DSL 能力
- `cute/`：CuTe 语义映射
- `cutlass_dsl/`：CUTLASS DSL 层
- `pipeline/`、`utils/`、`jax/`、`torch.py`：运行时和框架集成

可见 CUTLASS 4.x 的方向是：**C++ 模板内核 + Python DSL 入口并行发展**。

## 8. 代码阅读建议（面向二次开发）

建议按以下顺序建立全局认知：

1. 根 `README.md` 与根 `CMakeLists.txt`（看构建开关和能力边界）
2. `include/cute`（理解 layout/tensor/atom 的抽象基石）
3. `include/cutlass/gemm` + `include/cutlass/epilogue`（主路径）
4. `tools/library` + `tools/profiler`（实例化与性能验证机制）
5. 对应 `examples/*` 与 `test/unit/*`（找可运行样例和验证参考）
6. 若考虑 Python 工作流，再进入 `python/CuTeDSL`

## 9. 一句话总结

CUTLASS 的架构核心是：**以 `cute` 抽象为底座、以 `cutlass` 模板算子为主体、以 tools/examples/test/python 构成工程化闭环，并通过按架构特化文件族持续扩展到新 GPU 代际。**

