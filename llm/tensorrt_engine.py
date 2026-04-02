"""从 ONNX 构建 TensorRT engine，在 GPU 上推理，并与 PyTorch / ONNX Runtime 对比 logits。"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch

from llm.model import DecoderOnlyTransformer

# I/O 名称与 export_onnx 一致；若 ONNX 不同，可用 onnx 解析首输入名
DEFAULT_INPUT_NAME = "input_ids"


def _onnx_first_input_name(onnx_path: str) -> str:
    try:
        import onnx

        model = onnx.load(onnx_path)
        return model.graph.input[0].name
    except Exception:
        return DEFAULT_INPUT_NAME


def _trt_logger():
    import tensorrt as trt

    return trt.Logger(trt.Logger.WARNING)


def build_trt_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    *,
    min_shape: Tuple[int, int],
    opt_shape: Tuple[int, int],
    max_shape: Tuple[int, int],
    fp16: bool = False,
    workspace_bytes: int = 1 << 30,
    input_name: Optional[str] = None,
) -> None:
    """
    解析 ONNX，为动态 batch/seq 设置 optimization profile，序列化 engine。
    min/opt/max_shape 均为 (batch, seq_len)。
    """
    import tensorrt as trt

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)

    in_name = input_name or _onnx_first_input_name(onnx_path)
    logger = _trt_logger()
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        err = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT 解析 ONNX 失败:\n{err}")

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        except Exception:
            pass
    else:
        config.max_workspace_size = workspace_bytes

    profile = builder.create_optimization_profile()
    profile.set_shape(in_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("build_serialized_network 返回 None（构建失败）")
        blob = bytes(serialized)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("build_engine 返回 None（构建失败）")
        blob = engine.serialize()

    out_dir = os.path.dirname(os.path.abspath(engine_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(blob)


def _infer_trt_legacy(engine, context, input_ids: torch.Tensor) -> torch.Tensor:
    """TensorRT 8.x：binding API。"""
    import tensorrt as trt

    in_idx = out_idx = None
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            in_idx = i
        else:
            out_idx = i
    if in_idx is None or out_idx is None:
        raise RuntimeError("无法解析 engine 输入/输出 binding")

    context.set_binding_shape(in_idx, tuple(input_ids.shape))
    out_shape = tuple(context.get_binding_shape(out_idx))
    if any(d < 0 for d in out_shape):
        raise RuntimeError(f"输出 shape 非法: {out_shape}")

    output = torch.empty(
        out_shape, dtype=torch.float32, device=input_ids.device, memory_format=torch.contiguous_format
    )
    bindings = [0] * engine.num_bindings
    bindings[in_idx] = int(input_ids.data_ptr())
    bindings[out_idx] = int(output.data_ptr())

    stream = torch.cuda.Stream()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
    stream.synchronize()
    return output


def _infer_trt_v10(engine, context, input_ids: torch.Tensor) -> torch.Tensor:
    """TensorRT 10.x：tensor 名称 API。"""
    import tensorrt as trt

    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            in_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            out_name = name
    if not in_name or not out_name:
        raise RuntimeError("无法解析 engine 输入/输出 tensor 名")

    context.set_input_shape(in_name, tuple(input_ids.shape))
    out_shape = tuple(context.get_tensor_shape(out_name))
    if any(d < 0 for d in out_shape):
        raise RuntimeError(f"输出 shape 非法: {out_shape}")

    output = torch.empty(
        out_shape, dtype=torch.float32, device=input_ids.device, memory_format=torch.contiguous_format
    )
    context.set_tensor_address(in_name, int(input_ids.data_ptr()))
    context.set_tensor_address(out_name, int(output.data_ptr()))

    stream = torch.cuda.Stream()
    ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    if not ok:
        raise RuntimeError("execute_async_v3 返回 False")
    return output


def load_trt_engine(engine_path: str):
    import tensorrt as trt

    logger = _trt_logger()
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine 失败")
    return engine


def run_trt_logits(engine, input_ids: torch.Tensor) -> torch.Tensor:
    """
    input_ids: CUDA int64，shape (B, T)，须在 engine profile 范围内。
    返回 float32 logits，与 PyTorch 同设备。
    """
    if input_ids.dtype != torch.int64:
        input_ids = input_ids.to(dtype=torch.int64)
    if input_ids.device.type != "cuda":
        raise ValueError("TensorRT 推理需要 input_ids 在 CUDA 上")

    context = engine.create_execution_context()
    if hasattr(engine, "num_io_tensors") and engine.num_io_tensors > 0:
        return _infer_trt_v10(engine, context, input_ids)
    return _infer_trt_legacy(engine, context, input_ids)


def compare_trt_with_torch(
    engine,
    model: DecoderOnlyTransformer,
    *,
    batch_size: int,
    seq_len: int,
    rtol: float = 1e-3,
    atol: float = 1e-2,
) -> Tuple[float, bool]:
    """同一随机 input_ids：PyTorch（CUDA）与 TRT logits 对比。model 应在 cuda 上且 eval。"""
    model.eval()
    cfg = model.cfg
    if seq_len > cfg.block_size:
        raise ValueError(f"seq_len 不能超过 block_size={cfg.block_size}")

    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("compare_trt_with_torch 需要模型在 CUDA 上")

    idx = torch.randint(
        0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device
    )
    with torch.no_grad():
        pt_logits, _ = model(idx, None)

    trt_logits = run_trt_logits(engine, idx)

    d = (pt_logits - trt_logits).abs().max().item()
    ok = torch.allclose(pt_logits, trt_logits, rtol=rtol, atol=atol)
    return float(d), bool(ok)


def compare_trt_with_onnx(
    engine,
    onnx_path: str,
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    rtol: float = 1e-3,
    atol: float = 1e-2,
) -> Tuple[float, bool]:
    """TRT（CUDA）与 ONNX Runtime（CPU）对比；用于无 checkpoint 时校验 engine。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    in_name = in_meta.name

    idx_np = np.random.randint(
        0, max(vocab_size, 1), size=(batch_size, seq_len), dtype=np.int64
    )
    idx = torch.from_numpy(idx_np).to(device="cuda", dtype=torch.long)

    ref = sess.run(None, {in_name: idx_np})[0]

    trt_logits = run_trt_logits(engine, idx)
    trt_np = trt_logits.detach().cpu().numpy()

    d = float(np.max(np.abs(ref - trt_np)))
    ok = np.allclose(ref, trt_np, rtol=rtol, atol=atol)
    return d, ok
