"""自检：构建小模型、导出临时 ONNX、ORT 与 PyTorch 对齐。需 onnx、onnxruntime。"""
from __future__ import annotations

import os
import sys
import tempfile

import torch

from llm.model import DecoderConfig, DecoderOnlyTransformer
from llm.onnx_export import export_decoder_to_onnx, verify_onnx_against_torch


def main() -> int:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("请安装: pip install onnx onnxruntime", file=sys.stderr)
        return 1

    cfg = DecoderConfig(
        vocab_size=48,
        block_size=64,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=256,
        dropout=0.0,
    )
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(cfg)
    model.eval()

    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    try:
        export_decoder_to_onnx(
            model,
            path,
            opset_version=17,
            example_batch=1,
            example_seq_len=24,
        )
        max_diff, ok = verify_onnx_against_torch(
            path, model, batch_size=2, seq_len=12
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    print(f"test_onnx_export: max|pt-ort|={max_diff:.6e} allclose={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
