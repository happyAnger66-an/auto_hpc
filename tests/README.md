# tests

集成与自检脚本，在**仓库根目录**执行，确保 `llm` 包可被导入（当前工作目录应在 `PYTHONPATH` 中，从根目录运行 `python -m ...` 即可）。

## llm

| 模块 | 说明 |
|------|------|
| `tests.llm.test_onnx_export` | 小模型导出 ONNX，ORT 与 PyTorch 对齐 |
| `tests.llm.test_trt_build_run` | ONNX → TensorRT engine，与 PyTorch 对齐（无 CUDA / 无 TensorRT 时跳过） |

示例：

```bash
python3 -m tests.llm.test_onnx_export
python3 -m tests.llm.test_trt_build_run
```

当前自检以 `python -m tests.llm...` 方式运行（脚本入口为 `main()`），未写成 pytest 用例；若改为 pytest，可将逻辑拆成 `test_*` 函数。
