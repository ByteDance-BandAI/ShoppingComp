# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
ShoppingCompJudge: Simple, composable evaluation framework for ShoppingComp.

核心目标：
- 统一 Answer Match F1 / SoP / Scenario Coverage / Rationale Validity / Safety Pass Rate 的接口
- 提供类似 simple-evals 的命令行使用体验
"""

from .cli import main as cli_main
from .runner import main as runner_main

__all__ = ["cli_main", "runner_main"]


