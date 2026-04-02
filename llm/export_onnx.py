"""导出 Decoder LLM 为 ONNX，并可选用 ONNX Runtime 与 PyTorch 对齐校验。"""
from __future__ import annotations

import argparse
import sys

import torch

from llm.model import DecoderConfig, DecoderOnlyTransformer
from llm.onnx_export import export_decoder_to_onnx, verify_onnx_against_torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出 llm.DecoderOnlyTransformer 为 ONNX")
    p.add_argument("--out", type=str, required=True, help="输出 .onnx 路径")
    p.add_argument("--checkpoint", type=str, default="", help="可选：torch.save 的 state_dict 路径")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--example_batch", type=int, default=1)
    p.add_argument("--example_seq_len", type=int, default=32)
    p.add_argument("--no_dynamic_batch", action="store_true")
    p.add_argument("--no_dynamic_seq", action="store_true")
    p.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="导出后用 onnxruntime 与 PyTorch 对比（默认开启）",
    )
    p.add_argument("--verify_batch", type=int, default=2)
    p.add_argument("--verify_seq_len", type=int, default=16)
    # 与 train_tiny 一致的模型结构参数
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

    cfg = DecoderConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = DecoderOnlyTransformer(cfg)
    if args.checkpoint:
        load_kw = {"map_location": "cpu"}
        try:
            state = torch.load(args.checkpoint, **load_kw, weights_only=True)
        except TypeError:
            state = torch.load(args.checkpoint, **load_kw)
        model.load_state_dict(state, strict=True)
    model.eval()

    if args.example_seq_len > cfg.block_size:
        print(
            f"错误: example_seq_len ({args.example_seq_len}) > block_size ({cfg.block_size})",
            file=sys.stderr,
        )
        return 1

    export_decoder_to_onnx(
        model,
        args.out,
        opset_version=args.opset,
        dynamic_batch=not args.no_dynamic_batch,
        dynamic_seq=not args.no_dynamic_seq,
        example_batch=args.example_batch,
        example_seq_len=args.example_seq_len,
    )
    print(f"已写入 ONNX: {args.out}")

    if args.verify:
        try:
            max_diff, ok = verify_onnx_against_torch(
                args.out,
                model,
                batch_size=args.verify_batch,
                seq_len=args.verify_seq_len,
            )
        except ImportError as e:
            print(f"校验跳过（缺少依赖）: {e}", file=sys.stderr)
            return 0
        print(f"校验 max|pt-ort| = {max_diff:.6e}, allclose={ok}")
        if not ok:
            print("校验未通过：logits 与 ONNX Runtime 输出差异超出阈值", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
