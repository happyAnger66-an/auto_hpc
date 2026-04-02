"""在具备 CUDA + TensorRT 时：导出小 ONNX、构建 engine、与 PyTorch 对比；否则跳过。"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import torch

from llm.model import DecoderConfig, DecoderOnlyTransformer
from llm.onnx_export import export_decoder_to_onnx


def _repo_root() -> str:
    # tests/llm/test_trt_build_run.py -> 仓库根目录
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> int:
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("SKIP: 未安装 tensorrt")
        return 0
    if not torch.cuda.is_available():
        print("SKIP: 无 CUDA")
        return 0

    cfg = DecoderConfig(
        vocab_size=48,
        block_size=64,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=256,
        dropout=0.0,
    )
    m = DecoderOnlyTransformer(cfg).eval()
    root = _repo_root()
    d = tempfile.mkdtemp()
    onnx_p = os.path.join(d, "m.onnx")
    eng_p = os.path.join(d, "m.engine")
    ckpt = os.path.join(d, "ckpt.pt")
    torch.save(m.state_dict(), ckpt)
    export_decoder_to_onnx(m, onnx_p, example_seq_len=24)

    cmd = [
        sys.executable,
        "-m",
        "llm.trt_build_run",
        "--onnx",
        onnx_p,
        "--engine",
        eng_p,
        "--min_shape",
        "1x1",
        "--opt_shape",
        "1x24",
        "--max_shape",
        "4x64",
        "--vocab_size",
        "48",
        "--block_size",
        "64",
        "--n_layers",
        "2",
        "--n_heads",
        "4",
        "--d_model",
        "128",
        "--d_ff",
        "256",
        "--dropout",
        "0",
        "--checkpoint",
        ckpt,
        "--compare_batch",
        "2",
        "--compare_seq_len",
        "12",
    ]
    r = subprocess.run(cmd, cwd=root)
    if r.returncode != 0:
        print(f"test_trt_build_run: subprocess exit {r.returncode}", file=sys.stderr)
        return r.returncode
    print("test_trt_build_run: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
