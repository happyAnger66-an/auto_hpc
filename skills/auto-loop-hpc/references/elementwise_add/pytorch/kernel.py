import torch


def run(a, b, output):
    """a + b with fp32 accumulation, bf16 DPS (contract: a, b, output)."""
    output[:] = (a.to(torch.float32) + b.to(torch.float32)).to(a.dtype)
