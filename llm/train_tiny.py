import argparse
import time

import torch

from llm.model import DecoderConfig, DecoderOnlyTransformer
from llm.tokenizer import CharTokenizer, make_batch


def get_default_corpus() -> str:
    # 保持极简：默认语料只是一个小段落，便于快速跑通。
    return (
        "auto_hpc: compare high-performance operators on GPU.\n"
        "This is a tiny decoder-only transformer demo.\n"
        "The model learns next-character prediction.\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text_path", type=str, default="", help="可选：训练文本路径（UTF-8）")
    p.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--gen_every", type=int, default=150)
    p.add_argument("--gen_tokens", type=int, default=200)
    p.add_argument("--top_k", type=int, default=50)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.text_path:
        with open(args.text_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = get_default_corpus()

    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    cfg = DecoderConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = DecoderOnlyTransformer(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(data.to(device), args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            print(f"step {step:4d}/{args.steps} | loss {loss.item():.4f} | {dt:.1f}s")

        if step % args.gen_every == 0 or step == args.steps:
            prompt = "auto_hpc:"
            idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
            out = model.generate(idx, max_new_tokens=args.gen_tokens, temperature=1.0, top_k=args.top_k)
            print("\n--- sample ---")
            print(tok.decode(out[0].tolist()))
            print("--------------\n")


if __name__ == "__main__":
    main()

