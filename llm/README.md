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

### 文件说明

- `llm/model.py`：decoder-only Transformer（forward + generate）
- `llm/tokenizer.py`：最小字符级 tokenizer 与 batch 构造
- `llm/train_tiny.py`：训练脚本（打印 loss 与采样生成）

