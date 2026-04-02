"""将 DecoderOnlyTransformer 导出为 ONNX（仅推理 logits），并提供与 PyTorch 的数值校验。"""
from __future__ import annotations

import inspect
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from llm.model import DecoderOnlyTransformer


class DecoderONNXWrapper(nn.Module):
    """ONNX 只支持固定返回值：输入 token ids，输出 logits。"""

    def __init__(self, model: DecoderOnlyTransformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids, None)
        return logits


def export_decoder_to_onnx(
    model: DecoderOnlyTransformer,
    onnx_path: str,
    *,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    dynamic_seq: bool = True,
    example_batch: int = 1,
    example_seq_len: int = 32,
) -> None:
    """
    在 CPU 上导出；请先 model.eval()，dropout 在 eval 下为恒等映射。
    """
    model.eval()
    wrapper = DecoderONNXWrapper(model).cpu()
    cfg = model.cfg
    if example_seq_len > cfg.block_size or example_seq_len < 1:
        raise ValueError(
            f"example_seq_len 须在 [1, block_size] 内，当前 block_size={cfg.block_size}"
        )

    idx = torch.randint(
        0,
        cfg.vocab_size,
        (example_batch, example_seq_len),
        dtype=torch.long,
        device="cpu",
    )

    dynamic_axes = None
    if dynamic_batch or dynamic_seq:
        dynamic_axes = {"input_ids": {}, "logits": {}}
        if dynamic_batch:
            dynamic_axes["input_ids"][0] = "batch"
            dynamic_axes["logits"][0] = "batch"
        if dynamic_seq:
            dynamic_axes["input_ids"][1] = "seq"
            dynamic_axes["logits"][1] = "seq"

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)) or ".", exist_ok=True)

    export_kw: dict = {
        "model": wrapper,
        "args": idx,
        "f": onnx_path,
        "input_names": ["input_ids"],
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        export_kw["dynamo"] = False

    torch.onnx.export(**export_kw)


def verify_onnx_against_torch(
    onnx_path: str,
    model: DecoderOnlyTransformer,
    *,
    batch_size: int = 2,
    seq_len: int = 16,
    rtol: float = 1e-4,
    atol: float = 1e-3,
) -> Tuple[float, bool]:
    """
    使用 ONNX Runtime（CPU）与 PyTorch（CPU 权重副本）对比 logits，避免 GPU/CPU 数值差导致误判。
    返回 (max_abs_diff, ok)。
    """
    import onnxruntime as ort

    cfg = model.cfg
    if seq_len > cfg.block_size:
        raise ValueError(f"seq_len 不能超过 block_size={cfg.block_size}")

    cpu_model = DecoderOnlyTransformer(cfg)
    cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    cpu_model.eval()

    idx = torch.randint(
        0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long, device="cpu"
    )
    with torch.no_grad():
        pt_logits, _ = cpu_model(idx, None)

    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    in_name = sess.get_inputs()[0].name
    ort_out = sess.run(
        None, {in_name: idx.cpu().numpy().astype(np.int64)}
    )[0]

    pt_np = pt_logits.detach().cpu().numpy()
    max_diff = float(np.max(np.abs(pt_np - ort_out)))
    ok = np.allclose(pt_np, ort_out, rtol=rtol, atol=atol)
    return max_diff, ok
