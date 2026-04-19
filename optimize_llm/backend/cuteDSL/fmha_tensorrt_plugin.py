"""
使用 tensorrt.plugin 将 CuTeDSL FMHA 注册为 TensorRT 插件。

插件 ID: ``auto_hpc::fmha_cutedsl``
"""

from typing import Tuple

import tensorrt.plugin as trtp
import torch

from optimize_llm.backend.cuteDSL.fmha_cutedsl import fmha_cutedsl_forward

PLUGIN_ID = "auto_hpc::fmha_cutedsl"


@trtp.register(PLUGIN_ID)
def _fmha_cutedsl_plugin_desc(
    q: trtp.TensorDesc,
    k: trtp.TensorDesc,
    v: trtp.TensorDesc,
    is_causal: bool,
) -> Tuple[trtp.TensorDesc]:
    del is_causal
    return (q.like(),)


@trtp.impl(PLUGIN_ID)
def _fmha_cutedsl_plugin_impl(
    q: trtp.Tensor,
    k: trtp.Tensor,
    v: trtp.Tensor,
    is_causal: bool,
    outputs: Tuple[trtp.Tensor],
    stream: int,
) -> None:
    del stream
    q_t = torch.as_tensor(q, device="cuda")
    k_t = torch.as_tensor(k, device="cuda")
    v_t = torch.as_tensor(v, device="cuda")
    out_t = torch.as_tensor(outputs[0], device="cuda")
    fmha_cutedsl_forward(q_t, k_t, v_t, is_causal=is_causal, out=out_t)


def register_fmha_plugin() -> str:
    return PLUGIN_ID
