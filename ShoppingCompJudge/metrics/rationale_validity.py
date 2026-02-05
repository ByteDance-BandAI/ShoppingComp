#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
Rationale Validity (RV) 聚合实现。

输入：CoP judge 结果 jsonl（每行有 cop_evaluation）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable

from ..core import BaseEval, EvalResult, MetricScore


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[RationaleValidity] 第{line_num}行 JSON 解析失败: {e}")
                continue


class RationaleValidityEval(BaseEval):
    eval_name = "rationale_validity"

    def __init__(self, cop_file: str) -> None:
        super().__init__(cop_file=cop_file)
        self.cop_file = cop_file

    def run(self) -> EvalResult:
        total_checks = 0
        accurate_checks = 0

        for row in _iter_jsonl(self.cop_file):
            cop_eval = row.get("cop_evaluation", {}) or {}
            t = cop_eval.get("total_checks")
            a = cop_eval.get("accurate_checks")

            if isinstance(t, int) and isinstance(a, int):
                total_checks += t
                accurate_checks += a
                continue

            product_evals = cop_eval.get("product_evaluations", []) or []
            for product in product_evals:
                for check in product.get("check_evaluations", []) or []:
                    total_checks += 1
                    if check.get("accuracy_result") == 1:
                        accurate_checks += 1

        rv = accurate_checks / total_checks if total_checks > 0 else 0.0
        metric = MetricScore(
            name="rationale_validity",
            value=rv,
            extra={
                "overall_accuracy": rv,
                "total_checks": total_checks,
                "accurate_checks": accurate_checks,
            },
        )
        return EvalResult(metrics={"rationale_validity": metric}, score=metric.value)



