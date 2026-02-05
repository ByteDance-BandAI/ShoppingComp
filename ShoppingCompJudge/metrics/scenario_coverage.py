#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
Scenario Coverage 聚合实现。

输入：CoR judge 结果 jsonl（每行有 cor_evaluation.metrics）。
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
                print(f"[ScenarioCoverage] 第{line_num}行 JSON 解析失败: {e}")
                continue


class ScenarioCoverageEval(BaseEval):
    eval_name = "scenario_coverage"

    def __init__(self, cor_file: str) -> None:
        super().__init__(cor_file=cor_file)
        self.cor_file = cor_file

    def run(self) -> EvalResult:
        precisions = []
        recalls = []
        f1s = []

        total_matched_demands = 0
        total_demands = 0
        total_matched_scenes = 0
        total_scenes = 0

        for row in _iter_jsonl(self.cor_file):
            cor_eval = row.get("cor_evaluation", {}) or {}
            metrics = cor_eval.get("metrics", {}) or {}

            current_total_scenes = metrics.get("total_scenes", 0)
            current_total_demands = metrics.get("total_demands", 0)
            if current_total_scenes == 0:
                continue

            p = metrics.get("precision", 0.0)
            r = metrics.get("recall", 0.0)
            f1 = metrics.get("f1", 0.0)

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

            total_matched_demands += metrics.get("matched_demands", 0)
            total_demands += current_total_demands
            total_matched_scenes += metrics.get("matched_scenes", 0)
            total_scenes += current_total_scenes

        if not precisions:
            metric = MetricScore(
                name="scenario_coverage",
                value=0.0,
                extra={
                    "avg_p": 0.0,
                    "avg_r": 0.0,
                    "avg_f": 0.0,
                    "overall_p": 0.0,
                    "overall_r": 0.0,
                    "overall_f": 0.0,
                    "total_demands": 0,
                    "total_scenes": 0,
                },
            )
            return EvalResult(metrics={"scenario_coverage": metric}, score=0.0)

        avg_p = sum(precisions) / len(precisions)
        avg_r = sum(recalls) / len(recalls)
        avg_f = sum(f1s) / len(f1s)

        overall_p = total_matched_demands / total_demands if total_demands > 0 else 0.0
        overall_r = total_matched_scenes / total_scenes if total_scenes > 0 else 0.0
        overall_f = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

        metric = MetricScore(
            name="scenario_coverage",
            value=overall_f,
            extra={
                "avg_p": avg_p,
                "avg_r": avg_r,
                "avg_f": avg_f,
                "overall_p": overall_p,
                "overall_r": overall_r,
                "overall_f": overall_f,
                "total_demands": total_demands,
                "total_scenes": total_scenes,
            },
        )
        return EvalResult(metrics={"scenario_coverage": metric}, score=metric.value)



