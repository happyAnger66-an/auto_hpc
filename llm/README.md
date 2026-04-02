## llm：最小 PyTorch Decoder LLM

这是一个**最小可运行**的 decoder-only Transformer（因果自注意力）示例，用于跑通：

- **next-token（这里是 next-character）预测训练**
- **自回归生成**（温度采样 + top-k）

### 运行训练（默认内置小语料）

在仓库根目录执行：

```bash
python3 -m llm.train_tiny --device cpu --steps 200
```

有 CUDA 且装了 CUDA 版 PyTorch：

```bash
python3 -m llm.train_tiny --device cuda --steps 500
```

### 使用自己的文本

```bash
python3 -m llm.train_tiny --text_path /path/to/text.txt --device cuda --steps 1000
```

### 导出 ONNX

依赖：`pip install onnx onnxruntime`（校验需要 `onnxruntime`）。

在仓库根目录：

```bash
python3 -m llm.export_onnx --out /tmp/decoder_llm.onnx --verify
```

可指定与训练一致的结构（默认 `vocab_size=64` 等仅用于随机初始化导出；真实训练后请传 `--checkpoint your.pt` 与对应 `--vocab_size` 等）。

关闭数值校验：`python3 -m llm.export_onnx --out model.onnx --no-verify`

一键自检（临时文件导出 + ORT 对齐，需 `onnxruntime`），在仓库根目录：

```bash
python3 -m tests.llm.test_onnx_export
```

详见 `tests/README.md`。

### 文件说明

- `llm/model.py`：decoder-only Transformer（forward + generate）
- `llm/tokenizer.py`：最小字符级 tokenizer 与 batch 构造
- `llm/train_tiny.py`：训练脚本（打印 loss 与采样生成）
- `llm/onnx_export.py`：ONNX 导出与 PyTorch/ORT 对齐校验
- `llm/export_onnx.py`：命令行导出脚本
- `llm/tensorrt_engine.py` / `llm/trt_build_run.py`：TensorRT 构建、推理与对比
- `tests/llm/`：ONNX / TensorRT 集成自检（见 `tests/README.md`）

### TensorRT：ONNX → engine → GPU 推理，并与 PyTorch / ONNX Runtime 对比

依赖：本机已安装与 CUDA 版本匹配的 **TensorRT**（可 `import tensorrt`）、**CUDA 版 PyTorch**。与 ONNX Runtime 对比时需 `onnxruntime`。

在仓库根目录，先准备好与训练/导出一致的 ONNX；**与 PyTorch 对齐**时请传入导出该 ONNX 时使用的 `state_dict`（`--checkpoint`）及相同结构参数。

```bash
python3 -m llm.trt_build_run \
  --onnx /path/to/model.onnx \
  --engine /path/to/model.engine \
  --min_shape 1x1 --opt_shape 1x32 --max_shape 8x128 \
  --checkpoint /path/to/weights.pt \
  --vocab_size 64 --block_size 128
```

- 不传 `--checkpoint` 时，默认做 **TensorRT vs ONNX Runtime（CPU）** 对比（`--vocab_size` 等需与模型一致，用于随机 `input_ids` 上界）。
- `--fp16` 构建 FP16 engine，对比容差会自动放宽。
- 已有 engine 仅做对比：`--skip_build --engine /path/to/model.engine`（仍需 `--onnx` 供 ONNX 对比或记录路径；PyTorch 对比只需 checkpoint + 结构参数）。

一键自检（无 CUDA / 无 TensorRT 时打印 SKIP 并退出 0），在仓库根目录：

```bash
python3 -m tests.llm.test_trt_build_run
```

