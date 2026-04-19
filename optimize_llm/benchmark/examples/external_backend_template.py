"""
外部仓库 backend 插件最小模板。

使用方式（示例）:
1) 把本文件复制到你的外部仓库（如 /path/to/your_repo/my_bench_plugin.py）
2) 确保外部仓库可被 Python 导入（PYTHONPATH 或安装为包）
3) 通过 benchmark 动态加载:
   python3 -m optimize_llm.benchmark \
     --backends pytorch \
     --backend-targets your_repo.my_bench_plugin:ExternalFmhaBackend
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ExternalFmhaBackend:
    """
    最小可用 backend 插件示例。

    必须实现字段/方法:
    - name: str
    - supported(workload) -> bool
    - prepare(ctx) -> None
    - run(ctx) -> None
    """

    name = "external_template"

    def supported(self, workload) -> bool:  # noqa: ANN001
        # workload 字段参考: B/H/L/D/dtype/is_causal/seed/device
        return workload.device == "cuda" and torch.cuda.is_available()

    def prepare(self, ctx) -> None:  # noqa: ANN001
        # 可选: JIT 编译、缓存初始化、权重预处理等
        # 最小模板无需预处理
        return None

    def run(self, ctx) -> None:  # noqa: ANN001
        # ctx 提供:
        # - ctx.workload
        # - ctx.q / ctx.k / ctx.v / ctx.out
        out = F.scaled_dot_product_attention(
            ctx.q,
            ctx.k,
            ctx.v,
            is_causal=ctx.workload.is_causal,
        )
        ctx.out.copy_(out)

