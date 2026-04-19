"""
CuTeDSL 前向 FMHA：与 ``backend.triton.fmha_triton`` 相同 layout [B,H,L,D] 与在线 softmax 语义。

张量视图 ``[BH, L, D]``（BH=B*H），每个 CUDA block 处理一条 query；维度由 ``cute.size`` 取得。
因果与非因果分别 JIT 为两个 kernel，循环上界分别为 ``range(row + 1)`` 与 ``range(L_dim)``。
"""

from __future__ import annotations

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

SUPPORTED_HEAD_DIMS = (32, 64, 128, 256)
MAX_HEAD_DIM = max(SUPPORTED_HEAD_DIMS)


def _make_row_kernel(*, causal: bool):
    if causal:

        @cute.kernel
        def fmha_row_kernel(
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mO: cute.Tensor,
            sm_buf: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == 0:
                L_dim = cute.size(mQ, mode=[1])
                D_dim = cute.size(mQ, mode=[2])
                flat = cute.arch.block_idx()[0]
                row = flat % L_dim
                bh_id = flat // L_dim
                dtype = mQ.element_type
                neg_large = cutlass.Float32(-1.0e30)
                m_v = neg_large
                l_v = cutlass.Float32(0.0)
                acc = cute.make_rmem_tensor((MAX_HEAD_DIM,), cutlass.Float32)
                for kk in range(MAX_HEAD_DIM):
                    acc[kk] = cutlass.Float32(0.0)
                sm_scale = cutlass.Float32(sm_buf[0])

                for j in range(row + 1):
                    s = cutlass.Float32(0.0)
                    for kk in range(D_dim):
                        qv = cutlass.Float32(mQ[bh_id, row, kk])
                        kv = cutlass.Float32(mK[bh_id, j, kk])
                        s = s + qv * kv
                    s = s * sm_scale
                    m_new = s if s > m_v else m_v
                    exp_m = cute.math.exp(m_v - m_new)
                    exp_s = cute.math.exp(s - m_new)
                    l_v = l_v * exp_m + exp_s
                    for kk in range(D_dim):
                        vv = cutlass.Float32(mV[bh_id, j, kk])
                        acc[kk] = acc[kk] * exp_m + vv * exp_s
                    m_v = m_new

                inv_l = cutlass.Float32(1.0) / l_v
                for kk in range(D_dim):
                    outv = acc[kk] * inv_l
                    mO[bh_id, row, kk] = dtype(outv)

    else:

        @cute.kernel
        def fmha_row_kernel(
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mO: cute.Tensor,
            sm_buf: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == 0:
                L_dim = cute.size(mQ, mode=[1])
                D_dim = cute.size(mQ, mode=[2])
                flat = cute.arch.block_idx()[0]
                row = flat % L_dim
                bh_id = flat // L_dim
                dtype = mQ.element_type
                neg_large = cutlass.Float32(-1.0e30)
                m_v = neg_large
                l_v = cutlass.Float32(0.0)
                acc = cute.make_rmem_tensor((MAX_HEAD_DIM,), cutlass.Float32)
                for kk in range(MAX_HEAD_DIM):
                    acc[kk] = cutlass.Float32(0.0)
                sm_scale = cutlass.Float32(sm_buf[0])

                for j in range(L_dim):
                    s = cutlass.Float32(0.0)
                    for kk in range(D_dim):
                        qv = cutlass.Float32(mQ[bh_id, row, kk])
                        kv = cutlass.Float32(mK[bh_id, j, kk])
                        s = s + qv * kv
                    s = s * sm_scale
                    m_new = s if s > m_v else m_v
                    exp_m = cute.math.exp(m_v - m_new)
                    exp_s = cute.math.exp(s - m_new)
                    l_v = l_v * exp_m + exp_s
                    for kk in range(D_dim):
                        vv = cutlass.Float32(mV[bh_id, j, kk])
                        acc[kk] = acc[kk] * exp_m + vv * exp_s
                    m_v = m_new

                inv_l = cutlass.Float32(1.0) / l_v
                for kk in range(D_dim):
                    outv = acc[kk] * inv_l
                    mO[bh_id, row, kk] = dtype(outv)

    @cute.jit
    def fmha_entry(
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        sm_buf: cute.Tensor,
    ):
        bh = cute.size(mQ, mode=[0])
        ld = cute.size(mQ, mode=[1])
        fmha_row_kernel(mQ, mK, mV, mO, sm_buf).launch(
            grid=[bh * ld, 1, 1],
            block=[1, 1, 1],
        )

    return fmha_entry


@functools.lru_cache(maxsize=8)
def _compile_fmha(is_causal: bool, dtype_torch: torch.dtype) -> Any:
    dev = torch.device("cuda")
    fmha_entry = _make_row_kernel(causal=is_causal)
    bh0, l0, d0 = 2, 8, 64
    q0 = torch.zeros(bh0, l0, d0, device=dev, dtype=dtype_torch)
    sm0 = torch.zeros(1, device=dev, dtype=torch.float32)
    q_t = from_dlpack(q0).mark_layout_dynamic()
    k_t = from_dlpack(q0).mark_layout_dynamic()
    v_t = from_dlpack(q0).mark_layout_dynamic()
    o_t = from_dlpack(q0).mark_layout_dynamic()
    sm_t = from_dlpack(sm0).mark_layout_dynamic()
    return cute.compile(fmha_entry, q_t, k_t, v_t, o_t, sm_t, options="--generate-line-info")


def fmha_cutedsl_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape and q.dim() == 4
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

    bh = b * h
    q3 = q.reshape(bh, l, d)
    k3 = k.reshape(bh, l, d)
    v3 = v.reshape(bh, l, d)
    o3 = out.reshape(bh, l, d)

    sm_buf = torch.tensor([sm_scale], device="cuda", dtype=torch.float32)

    compiled = _compile_fmha(is_causal, q.dtype)
    q_t = from_dlpack(q3).mark_layout_dynamic()
    k_t = from_dlpack(k3).mark_layout_dynamic()
    v_t = from_dlpack(v3).mark_layout_dynamic()
    o_t = from_dlpack(o3).mark_layout_dynamic()
    sm_t = from_dlpack(sm_buf).mark_layout_dynamic()
    compiled(q_t, k_t, v_t, o_t, sm_t)
    return out
