"""LLM 优化相关：算子实现已迁至 ``backend/triton`` 与 ``backend/cuteDSL``。"""

from optimize_llm.backend.cuteDSL import (
    PLUGIN_ID as PLUGIN_ID_CUTEDSL,
    fmha_cutedsl_forward,
    register_fmha_plugin as register_fmha_cutedsl_plugin,
)
from optimize_llm.backend.triton import (
    PLUGIN_ID as PLUGIN_ID_TRITON,
    fmha_triton_forward,
    register_fmha_plugin as register_fmha_triton_plugin,
)

# 兼容旧名称：默认指向 Triton 插件 ID
PLUGIN_ID = PLUGIN_ID_TRITON
register_fmha_plugin = register_fmha_triton_plugin

__all__ = [
    "PLUGIN_ID",
    "PLUGIN_ID_TRITON",
    "PLUGIN_ID_CUTEDSL",
    "fmha_triton_forward",
    "fmha_cutedsl_forward",
    "register_fmha_plugin",
    "register_fmha_triton_plugin",
    "register_fmha_cutedsl_plugin",
]
