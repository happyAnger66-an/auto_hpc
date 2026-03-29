#!/usr/bin/env python3
"""
兼容入口：转发到 linear/compare.py。

  python3 compare_linear.py --m 1024 --n 1024 --k 1024
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    target = root / "linear" / "compare.py"
    if not target.is_file():
        print(f"错误: 未找到 {target}", file=sys.stderr)
        sys.exit(1)
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
