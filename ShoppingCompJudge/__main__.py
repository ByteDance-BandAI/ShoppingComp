#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
`python -m ShoppingCompJudge` 的统一入口。

目标：把零散的 judgers/runner 聚合为一个清晰的子命令式 CLI，同时保持历史入口可用。
"""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())




