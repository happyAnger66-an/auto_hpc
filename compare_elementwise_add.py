#!/usr/bin/env python3
"""
兼容入口：转发到 element-wise/compare.py（统一 GFLOPS 对比）。

用法与 `python3 element-wise/compare.py` 相同，例如:
  python3 compare_elementwise_add.py --m 4096 --n 4096
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    target = root / "element-wise" / "compare.py"
    if not target.is_file():
        print(f"错误: 未找到 {target}", file=sys.stderr)
        sys.exit(1)
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
