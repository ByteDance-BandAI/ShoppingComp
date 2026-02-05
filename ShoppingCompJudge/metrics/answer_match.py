#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
Answer Match F1 实现（Browse Products Score）。

逻辑来源：原 `calculate_matrix_metrics.py`，但抽象成 Eval 类。
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Set

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
                print(f"[AnswerMatch] 第{line_num}行 JSON 解析失败: {e}")
                continue


class AnswerMatchEval(BaseEval):
    eval_name = "answer_match"

    def __init__(self, jsonl_file: str, exclude_uuids: Optional[Set[str]] = None) -> None:
        super().__init__(jsonl_file=jsonl_file)
        self.jsonl_file = jsonl_file
        self.exclude_uuids = exclude_uuids or set()

    def run(self) -> EvalResult:
        stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_questions": 0,
                "valid_questions": 0,
                "total_gt_products": 0,
                "total_model_products": 0,
                "total_correct_products": 0,
                "precision_sum": 0.0,
                "recall_sum": 0.0,
                "precision_count": 0,
                "recall_count": 0,
            }
        )

        for row in _iter_jsonl(self.jsonl_file):
            uuid = row.get("uuid", "")
            if uuid in self.exclude_uuids:
                continue

            model_name = row.get("model_name", "unknown")
            gt_list = row.get("gt_product_list", []) or []
            model_list = row.get("model_product_list", []) or []
            product_set = row.get("product_set", []) or []

            s = stats[model_name]
            s["total_questions"] += 1

            if not gt_list:
                continue

            s["valid_questions"] += 1

            gt_count = len(gt_list)
            model_count = len(model_list)
            correct_count = len(product_set)

            s["total_gt_products"] += gt_count
            s["total_model_products"] += model_count
            s["total_correct_products"] += correct_count

            if model_count > 0:
                prec = correct_count / model_count
                s["precision_sum"] += prec
                s["precision_count"] += 1

            if gt_count > 0:
                rec = correct_count / gt_count
                s["recall_sum"] += rec
                s["recall_count"] += 1

        def _score_from_bucket(bucket: Dict[str, Any]) -> MetricScore:
            """
            从单个 model 的累计桶里计算指标。

            注意：
            - avg_precision/avg_recall 是“逐题平均”，只在对应分母>0的题目上计数
            - overall_precision/overall_recall/overall_f1 是“全量 micro”
            """
            if bucket["valid_questions"] == 0:
                avg_p = avg_r = avg_f1 = 0.0
            else:
                avg_p = (
                    bucket["precision_sum"] / bucket["precision_count"]
                    if bucket["precision_count"] > 0
                    else 0.0
                )
                avg_r = (
                    bucket["recall_sum"] / bucket["recall_count"]
                    if bucket["recall_count"] > 0
                    else 0.0
                )
                avg_f1 = (
                    2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0
                )

            overall_p = (
                bucket["total_correct_products"] / bucket["total_model_products"]
                if bucket["total_model_products"] > 0
                else 0.0
            )
            overall_r = (
                bucket["total_correct_products"] / bucket["total_gt_products"]
                if bucket["total_gt_products"] > 0
                else 0.0
            )
            overall_f1 = (
                2 * overall_p * overall_r / (overall_p + overall_r)
                if (overall_p + overall_r) > 0
                else 0.0
            )
            accuracy = (
                bucket["total_correct_products"] / bucket["total_gt_products"]
                if bucket["total_gt_products"] > 0
                else 0.0
            )

            return MetricScore(
                name="answer_match",
                value=overall_f1,
                extra={
                    "avg_precision": avg_p,
                    "avg_recall": avg_r,
                    "avg_f1": avg_f1,
                    "overall_precision": overall_p,
                    "overall_recall": overall_r,
                    "overall_f1": overall_f1,
                    "accuracy": accuracy,
                    "total_questions": bucket["total_questions"],
                    "valid_questions": bucket["valid_questions"],
                },
            )

        if not stats:
            metric0 = MetricScore(
                name="answer_match",
                value=0.0,
                extra={
                    "avg_precision": 0.0,
                    "avg_recall": 0.0,
                    "avg_f1": 0.0,
                    "overall_precision": 0.0,
                    "overall_recall": 0.0,
                    "overall_f1": 0.0,
                    "accuracy": 0.0,
                    "total_questions": 0,
                    "valid_questions": 0,
                    "model_count": 0,
                },
            )
            return EvalResult(metrics={"answer_match": metric0}, score=0.0)

        # 兼容：若文件只有一个 model_name，保持旧行为（仅返回 answer_match）
        if len(stats) == 1:
            (_model_name, bucket), = stats.items()
            metric = _score_from_bucket(bucket)
            return EvalResult(metrics={"answer_match": metric}, score=metric.value)

        # 新增：同一 jsonl 内包含多个 model_name 时，返回：
        # - answer_match: 将所有行合并后的整体（micro）结果
        # - answer_match__{model}: 每个 model 的结果（便于在一个输出里同时观察多个模型）
        merged = {
            "total_questions": 0,
            "valid_questions": 0,
            "total_gt_products": 0,
            "total_model_products": 0,
            "total_correct_products": 0,
            "precision_sum": 0.0,
            "recall_sum": 0.0,
            "precision_count": 0,
            "recall_count": 0,
        }
        for _m, b in stats.items():
            for k in (
                "total_questions",
                "valid_questions",
                "total_gt_products",
                "total_model_products",
                "total_correct_products",
                "precision_sum",
                "recall_sum",
                "precision_count",
                "recall_count",
            ):
                merged[k] += b[k]

        overall_metric = _score_from_bucket(merged)
        overall_metric.extra["model_count"] = len(stats)

        metrics: Dict[str, MetricScore] = {"answer_match": overall_metric}
        for model_name, bucket in stats.items():
            metrics[f"answer_match__{model_name}"] = _score_from_bucket(bucket)

        return EvalResult(metrics=metrics, score=overall_metric.value)



