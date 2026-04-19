---
name: optimize-llm
description: Compile, validate, and benchmark a CUDA, CUTLASS, or Triton operator against an optional Python reference by driving `skills/optimized-skill/kernel-benchmark/scripts/benchmark.py`. Use when Claude needs to 验证算子正确性、做 baseline 性能测试、比较 operator 与 reference 的 speedup、对 CUDA/CUTLASS 读取 `extern "C" void solve(...)` 签名推断参数，或在进入后续分析前先得到稳定 benchmark 结果。
---

# Kernel Benchmark

通过 `skills/optimized-skill/kernel-benchmark/scripts/benchmark.py` 编译、验证、压测三类算子：
