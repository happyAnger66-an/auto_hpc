"""
从 ONNX 构建 TensorRT engine，在 GPU 上推理，并与 PyTorch 或 ONNX Runtime 对比 logits。

需要：NVIDIA GPU、CUDA、PyTorch（CUDA 版）、Python 包 `tensorrt`；与 ONNX 对比时需 `onnxruntime`。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from llm.model import DecoderConfig, DecoderOnlyTransformer
from llm.tensorrt_engine import (
    build_trt_engine_from_onnx,
    compare_trt_with_onnx,
    compare_trt_with_torch,
    load_trt_engine,
)


def _parse_bt(s: str) -> tuple[int, int]:
    parts = s.lower().replace("*", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"形状应为 BxT，例如 1x32，收到: {s!r}")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ONNX -> TensorRT engine，GPU 推理，与 PyTorch / ONNX Runtime 数值对比"
    )
    p.add_argument("--onnx", type=str, required=True, help="输入 ONNX 路径")
    p.add_argument("--engine", type=str, required=True, help="输出 .engine 路径")
    p.add_argument(
        "--skip_build",
        action="store_true",
        help="不构建，仅加载已有 engine 做对比/推理",
    )
    p.add_argument("--fp16", action="store_true", help="构建 FP16 engine（对比容差自动放宽）")
    p.add_argument("--workspace_mb", type=int, default=1024, help="TensorRT workspace（MiB）")
    p.add_argument("--min_shape", type=_parse_bt, default=(1, 1))
    p.add_argument("--opt_shape", type=_parse_bt, default=(1, 32))
    p.add_argument("--max_shape", type=_parse_bt, default=(8, 128))
    p.add_argument(
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="构建后做数值对比（默认开启）",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="若指定：与 PyTorch（CUDA）对比（state_dict，结构参数须与导出 ONNX 时一致）；否则与 ONNX Runtime 对比",
    )
    p.add_argument("--compare_batch", type=int, default=2)
    p.add_argument("--compare_seq_len", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    # 与 export_onnx 一致（仅 compare_torch 时用于构造模型）
    p.add_argument("--vocab_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("错误: 未安装 TensorRT Python 包（import tensorrt）", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("错误: 需要 CUDA（TensorRT 推理与 PyTorch 对比在 GPU 上执行）", file=sys.stderr)
        return 1

    ws = max(1, args.workspace_mb) << 20

    if not args.skip_build:
        build_trt_engine_from_onnx(
            args.onnx,
            args.engine,
            min_shape=args.min_shape,
            opt_shape=args.opt_shape,
            max_shape=args.max_shape,
            fp16=args.fp16,
            workspace_bytes=ws,
        )
        print(f"已写入 TensorRT engine: {args.engine}")
    elif not os.path.isfile(args.engine):
        print(f"错误: --skip_build 但 engine 不存在: {args.engine}", file=sys.stderr)
        return 1

    if not args.compare:
        return 0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    engine = load_trt_engine(args.engine)

    rtol, atol = (1e-2, 1e-2) if args.fp16 else (1e-3, 1e-2)

    if args.checkpoint:
        cfg = DecoderConfig(
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )
        if args.compare_seq_len > cfg.block_size:
            print(
                f"错误: compare_seq_len ({args.compare_seq_len}) > block_size ({cfg.block_size})",
                file=sys.stderr,
            )
            return 2
        model = DecoderOnlyTransformer(cfg).cuda().eval()
        load_kw = {"map_location": "cpu"}
        try:
            state = torch.load(args.checkpoint, **load_kw, weights_only=True)
        except TypeError:
            state = torch.load(args.checkpoint, **load_kw)
        model.load_state_dict(state, strict=True)

        max_diff, ok = compare_trt_with_torch(
            engine,
            model,
            batch_size=args.compare_batch,
            seq_len=args.compare_seq_len,
            rtol=rtol,
            atol=atol,
        )
        print(f"TRT vs PyTorch: max|diff|={max_diff:.6e}, allclose={ok} (rtol={rtol}, atol={atol})")
        return 0 if ok else 3

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print(
            "错误: 未安装 onnxruntime，无法做 TRT vs ONNX 对比（或与 PyTorch 对比请传 --checkpoint）",
            file=sys.stderr,
        )
        return 1

    max_diff, ok = compare_trt_with_onnx(
        engine,
        args.onnx,
        vocab_size=args.vocab_size,
        batch_size=args.compare_batch,
        seq_len=args.compare_seq_len,
        rtol=rtol,
        atol=atol,
    )
    print(
        f"TRT vs ONNXRuntime: max|diff|={max_diff:.6e}, allclose={ok} (rtol={rtol}, atol={atol})"
    )
    print(
        "提示: 与 PyTorch 数值对齐请传与导出 ONNX 相同的 --checkpoint 及结构参数。",
        file=sys.stderr,
    )
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
