"""CuTeDSL 后端：FMHA 等。"""

from optimize_llm.backend.cuteDSL.fmha_tensorrt_plugin import PLUGIN_ID, register_fmha_plugin
from optimize_llm.backend.cuteDSL.fmha_cutedsl import fmha_cutedsl_forward

__all__ = ["PLUGIN_ID", "fmha_cutedsl_forward", "register_fmha_plugin"]
