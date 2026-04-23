"""Element-wise add via Triton (hidden_size=4096, bfloat16)."""

import torch
import triton
import triton.language as tl

_HIDDEN = 4096


@triton.jit
def _elemadd_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    stride_m,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    row_off = row * stride_m

    x = tl.load(a_ptr + row_off + offs, mask=mask, other=0.0)
    y = tl.load(b_ptr + row_off + offs, mask=mask, other=0.0)
    z = (x.to(tl.float32) + y.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + row_off + offs, z, mask=mask)


def run(a, b, output):
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("a and b must be bfloat16")
    if output.dtype != torch.bfloat16:
        raise TypeError("output must be bfloat16")
    if not a.is_cuda or not b.is_cuda or not output.is_cuda:
        raise RuntimeError("Triton elemadd expects CUDA tensors")
    if (
        a.shape[1] != _HIDDEN
        or b.shape != a.shape
        or output.shape != a.shape
    ):
        raise ValueError(
            f"expected a,b,output [*, {_HIDDEN}] with matching shapes"
        )

    n_rows, n_cols = a.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _elemadd_kernel[grid](
        a,
        b,
        output,
        a.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
