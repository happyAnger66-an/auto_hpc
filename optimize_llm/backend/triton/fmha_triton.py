"""
Triton 前向 FMHA（fused multi-head attention）：layout 为 [B, H, L, D]。
数值路径为在线 softmax（FlashAttention 风格）。
"""

import torch
import triton
import triton.language as tl

SUPPORTED_HEAD_DIMS = (32, 64, 128, 256)


@triton.jit
def _fmha_fwd_kernel(
    Q,
    K,
    V,
    Out,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    num_heads,
    seq_len,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // num_heads
    off_h = off_hz % num_heads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q_ptrs = (
        Q
        + off_z * stride_qb
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + tl.arange(0, HEAD_DIM)[None, :] * stride_qd
    )
    mask_m = offs_m < seq_len

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    lo = 0
    hi = seq_len
    if IS_CAUSAL:
        hi = tl.minimum(seq_len, (start_m + 1) * BLOCK_M)

    for start_n in tl.range(lo, hi, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seq_len

        k_ptrs = (
            K
            + off_z * stride_kb
            + off_h * stride_kh
            + offs_n_curr[None, :] * stride_km
            + tl.arange(0, HEAD_DIM)[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + off_z * stride_vb
            + off_h * stride_vh
            + offs_n_curr[:, None] * stride_vm
            + tl.arange(0, HEAD_DIM)[None, :] * stride_vd
        )

        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, k).to(tl.float32) * sm_scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        qk = tl.where(mask_m[:, None], qk, float("-inf"))
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        pv = tl.dot(p.to(q.dtype), v)
        acc = acc * alpha[:, None] + pv.to(tl.float32)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    out = tl.where(mask_m[:, None], out, 0.0)

    o_ptrs = (
        Out
        + off_z * stride_ob
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + tl.arange(0, HEAD_DIM)[None, :] * stride_od
    )
    tl.store(o_ptrs, out.to(q.dtype), mask=mask_m[:, None])


def fmha_triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    q, k, v: float16 / bfloat16 / float32，shape [B, H, L, D]，且 stride 与 shape 一致（连续最后一维）。
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape
    assert q.dim() == 4
    b, h, l, d = q.shape
    if d not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim 仅支持 {SUPPORTED_HEAD_DIMS}，当前为 {d}")
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    if sm_scale is None:
        sm_scale = float(d ** (-0.5))

    if out is None:
        out = torch.empty_like(q)
    else:
        if out.shape != q.shape or out.dtype != q.dtype:
            raise ValueError("out 必须与 q 同 shape / dtype")
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 8 if d == 256 else 4
    # D=256 时将 stage 降为 1，避免 shared memory 超限（部分消费级卡上 2-stage 会 OOR）
    num_stages = 1 if d == 256 else 2

    grid = (triton.cdiv(l, BLOCK_M), b * h)
    _fmha_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        h,
        l,
        sm_scale,
        HEAD_DIM=d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
