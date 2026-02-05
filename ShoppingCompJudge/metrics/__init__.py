#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
ShoppingCompJudge.metrics

封装所有聚合层指标：
- Answer Match F1
- SoP (Selection Accuracy)
- Scenario Coverage
- Rationale Validity
- Safety Rubric Pass Rate
"""

from .answer_match import AnswerMatchEval
from .sop import SoPEval
from .scenario_coverage import ScenarioCoverageEval
from .rationale_validity import RationaleValidityEval
from .safety_pass_rate import SafetyPassRateEval

__all__ = [
    "AnswerMatchEval",
    "SoPEval",
    "ScenarioCoverageEval",
    "RationaleValidityEval",
    "SafetyPassRateEval",
]


