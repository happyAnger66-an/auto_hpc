"""Triton 后端：FMHA 等。"""

from optimize_llm.backend.triton.fmha_tensorrt_plugin import PLUGIN_ID, register_fmha_plugin
from optimize_llm.backend.triton.fmha_triton import fmha_triton_forward

__all__ = ["PLUGIN_ID", "fmha_triton_forward", "register_fmha_plugin"]
