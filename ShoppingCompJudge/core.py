#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
核心评估抽象类与结果结构。

设计思路参考：
- OpenAI simple-evals: https://github.com/openai/simple-evals/blob/main/simple_evals.py
- 但这里专注于 ShoppingComp 的五类指标：
  - Answer Match F1（Browse Products Score）
  - SoP（Selection Accuracy, Satisfaction of Products）
  - Scenario Coverage
  - Rationale Validity
  - Safety Rubric Pass Rate
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MetricScore:
    """单个 metric 的标量结果（支持额外扩展字段）"""

    name: str
    value: float
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        # 将 extra 扁平化到顶层，方便下游直接用列名
        extra = base.pop("extra", {}) or {}
        base.update(extra)
        return base


@dataclass
class EvalResult:
    """
    一次评估的聚合结果（针对一个 jsonl 文件或一个“运行”）。

    与 simple-evals 中的 result.metrics/result.score 类似：
    - metrics: {metric_name: MetricScore}
    - score:   主指标（用于排序/对比），这里默认取第一个 metric。
    """

    metrics: Dict[str, MetricScore]
    score: float

    def to_flat_metrics(self) -> Dict[str, Any]:
        """展开成 {metric_name_subkey: value} 形式，便于写 CSV。"""
        flat: Dict[str, Any] = {}
        for name, metric in self.metrics.items():
            data = metric.to_dict()
            for k, v in data.items():
                if k == "name":
                    continue
                key = f"{name}_{k}" if k != "value" else name
                flat[key] = v
        return flat


class BaseEval(abc.ABC):
    """
    所有 ShoppingComp 评估的基类。

    子类只需要实现 `run()`，从输入路径计算出 EvalResult。
    """

    eval_name: str  # 例如: "answer_match", "sop", "scenario_coverage"

    def __init__(self, **config: Any) -> None:
        self.config = config

    @abc.abstractmethod
    def run(self) -> EvalResult:
        """执行评估并返回聚合结果。"""

    def __call__(self) -> EvalResult:
        return self.run()



