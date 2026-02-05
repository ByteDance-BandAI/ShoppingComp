#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
SoP (Selection Accuracy / Satisfaction of Products) 实现。

逻辑来源：`src/report_score/calc_soft_acc.py` 的 calculate_product_satisfaction。
"""

from __future__ import annotations

import json
import re
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
                print(f"[SoP] 第{line_num}行 JSON 解析失败: {e}")
                continue


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<tag>\s*(.*?)\s*</tag>", re.IGNORECASE | re.DOTALL)
_EN_CHOICES_RE = re.compile(
    r"\b(Satisfied|Not satisfied|Not Satisfied|Price not satisfied|Price Not Satisfied|Unable to judge|Unable to Judge|Unable to Determine)\b",
    re.IGNORECASE,
)
_ZH_CHOICES_RE = re.compile(r"(不满足|无法判断|满足)")


def _normalize_judge_answer(raw: Any) -> str:
    """
    Normalize judge answer across zh/en prompts.

    We support:
    - zh: 满足 / 不满足 / 无法判断
    - en: Satisfied / Not Satisfied / Unable to Determine
    - legacy booleans: yes/true/1
    - fallback: if full text contains any allowed choice, extract it
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""

    # If wrapped in <answer>...</answer>, prefer the inner block.
    m = _ANSWER_TAG_RE.search(s)
    if m:
        s = (m.group(1) or "").strip()
        if not s:
            return ""

    # New format: extract <tag>...</tag> if present.
    mt = _TAG_RE.search(s)
    if mt:
        s = (mt.group(1) or "").strip()
        if not s:
            return ""

    # Direct zh choice
    m = _ZH_CHOICES_RE.search(s)
    if m:
        return m.group(1)

    # Direct en choice
    m = _EN_CHOICES_RE.search(s)
    if m:
        # canonicalize
        choice = m.group(1).strip().lower()
        # normalize common variants
        if choice == "satisfied":
            return "satisfied"
        if choice == "not satisfied":
            return "not satisfied"
        if choice == "unable to determine":
            return "unable to determine"
        if choice == "unable to judge":
            return "unable to judge"
        if choice == "price not satisfied":
            return "price not satisfied"

    # Generic normalize (punctuation/case)
    s2 = re.sub(r"[\s\.\,\;\:\!\?\(\)\[\]\{\}\"'`]+", " ", s).strip().lower()
    return s2


def _is_satisfied(judge_answer: Any, *, ignore_price: bool = False) -> bool:
    a = _normalize_judge_answer(judge_answer)
    if not a:
        return False

    # zh
    if a == "满足":
        return True
    if a in ("不满足", "无法判断"):
        return False

    # en
    if a == "satisfied":
        return True
    if a == "price not satisfied":
        return bool(ignore_price)
    if a in ("not satisfied", "unable to determine", "unable to judge"):
        return False

    # legacy truthy tokens
    if a in ("yes", "true", "1", "y", "t"):
        return True
    if a in ("no", "false", "0", "n", "f"):
        return False

    # very defensive fallback: handle "not satisfied" substring
    if "not satisfied" in a:
        return False
    if "unable to determine" in a:
        return False
    if "unable to judge" in a:
        return False
    if "price not satisfied" in a:
        return bool(ignore_price)
    if "satisfied" in a:
        return True

    return False


def _product_satisfaction(eval_product_list, *, ignore_price: bool = False):
    if not eval_product_list:
        return 0.0

    total_satisfaction = 0.0
    valid_products = 0

    for product_eval in eval_product_list:
        for _product_name, product_evaluations in product_eval.items():
            if not product_evaluations:
                continue
            satisfied = 0
            total_req = len(product_evaluations)
            for evaluation in product_evaluations:
                # New judge file may store judge_tag; keep backward compatibility.
                judge_answer = evaluation.get("judge_tag") or evaluation.get("judge_answer")
                if _is_satisfied(judge_answer, ignore_price=ignore_price):
                    satisfied += 1
            if total_req > 0:
                total_satisfaction += satisfied / total_req
                valid_products += 1

    return total_satisfaction / valid_products if valid_products > 0 else 0.0


class SoPEval(BaseEval):
    eval_name = "sop"

    def __init__(self, judge_file: str) -> None:
        super().__init__(judge_file=judge_file)
        self.judge_file = judge_file

    def run(self) -> EvalResult:
        satisfactions = []
        satisfactions_ignore_price = []
        price_not_satisfied_cnt = 0
        total_evaluations_cnt = 0
        for row in _iter_jsonl(self.judge_file):
            eval_product_list = row.get("eval_product_list", []) or []
            sat = _product_satisfaction(eval_product_list, ignore_price=False)
            sat_ignore_price = _product_satisfaction(eval_product_list, ignore_price=True)
            satisfactions.append(sat)
            satisfactions_ignore_price.append(sat_ignore_price)

            # tag distribution (best-effort)
            for product_eval in eval_product_list:
                for _product_name, product_evaluations in (product_eval or {}).items():
                    for evaluation in (product_evaluations or []):
                        total_evaluations_cnt += 1
                        tag = _normalize_judge_answer(evaluation.get("judge_tag") or evaluation.get("judge_answer"))
                        if tag == "price not satisfied":
                            price_not_satisfied_cnt += 1

        if not satisfactions:
            metric = MetricScore(
                name="sop",
                value=0.0,
                extra={
                    "mean_satisfaction": 0.0,
                    "mean_satisfaction_ignore_price": 0.0,
                    "min_satisfaction": 0.0,
                    "max_satisfaction": 0.0,
                    "total_items": 0,
                    "price_not_satisfied_count": 0,
                    "total_evaluations": 0,
                },
            )
            return EvalResult(metrics={"sop": metric}, score=0.0)

        mean_satisfaction = sum(satisfactions) / len(satisfactions)
        mean_satisfaction_ignore_price = (
            sum(satisfactions_ignore_price) / len(satisfactions_ignore_price)
            if satisfactions_ignore_price
            else mean_satisfaction
        )
        metric = MetricScore(
            name="sop",
            value=mean_satisfaction,
            extra={
                "mean_satisfaction": mean_satisfaction,
                "mean_satisfaction_ignore_price": mean_satisfaction_ignore_price,
                "min_satisfaction": min(satisfactions),
                "max_satisfaction": max(satisfactions),
                "total_items": len(satisfactions),
                "price_not_satisfied_count": price_not_satisfied_cnt,
                "total_evaluations": total_evaluations_cnt,
            },
        )
        return EvalResult(metrics={"sop": metric}, score=metric.value)



