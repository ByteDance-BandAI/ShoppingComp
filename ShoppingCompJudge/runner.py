#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
ShoppingCompJudge 聚合层入口（不调用 LLM）。

当前 runner.py 仅提供**最简单直接的新模式**：

- 必须使用 `--inputs metric=/path,metric=/path,...` 为每个 metric 指定输入文件
- 可选 `--group-by-model`：当输入包含多个 model_name 时按 model 分组聚合

更推荐使用统一 CLI：`python -m ShoppingCompJudge aggregate ...`（见 `ShoppingCompJudge/cli.py`）。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Set

from .core import EvalResult
from .metrics import (
    AnswerMatchEval,
    SoPEval,
    ScenarioCoverageEval,
    RationaleValidityEval,
    SafetyPassRateEval,
)
from .io_utils import read_jsonl_list, write_jsonl


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ShoppingComp evaluations (AnswerMatch, SoP, ScenarioCoverage, RV, SafetyPassRate).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help=(
            "要运行的 metric 名称，可以是逗号分隔列表。"
            "可选值：answer_match,sop,scenario_coverage,rationale_validity,safety_pass_rate"
        ),
    )
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="必选：多输入映射 metric=/path,metric=/path,...（每个 metric 都必须提供）。",
    )
    parser.add_argument(
        "--exclude-uuids",
        type=str,
        default="",
        help="仅对 answer_match 生效，要排除的 uuid，逗号分隔。",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-pro",
        help="SafetyPassRate 使用的 judge 模型名称（对应 trap_rubric_eval_results_v2 中的 key）。",
    )
    parser.add_argument(
        "--group-by-model",
        action="store_true",
        help="若输入包含多个 model_name，则按 model_name 分组聚合并分别输出（输出 json 会变为 {metric: {model: flat}}）。",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="可选，将所有 metric 的扁平结果写入到该 json 文件（单个 dict）。",
    )
    return parser


def _create_eval(metric_name: str, *, input_path: str, exclude: Set[str], judge_model: str):
    metric_name = metric_name.strip().lower()
    if metric_name == "answer_match":
        return AnswerMatchEval(jsonl_file=input_path, exclude_uuids=exclude)
    if metric_name == "sop":
        return SoPEval(judge_file=input_path)
    if metric_name == "scenario_coverage":
        return ScenarioCoverageEval(cor_file=input_path)
    if metric_name == "rationale_validity":
        return RationaleValidityEval(cop_file=input_path)
    if metric_name == "safety_pass_rate":
        return SafetyPassRateEval(rubric_result_file=input_path, judge_model=judge_model)
    raise ValueError(f"未知 metric: {metric_name}")


def build_inputs_map(metrics: List[str], inputs_kv: str) -> Dict[str, str]:
    """
    构造 metric -> input_file 的映射。

    仅支持新模式：解析 "metric=/path,metric=/path"。
    """
    metrics_norm = [m.strip().lower() for m in metrics if m.strip()]
    if not inputs_kv.strip():
        raise ValueError("缺少 --inputs（新模式要求为每个 metric 显式提供输入文件映射）。")

    out: Dict[str, str] = {}
    for part in inputs_kv.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--inputs 格式错误: {part}（应为 metric=/path）")
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"--inputs 格式错误: {part}")
        out[k] = v

    missing = [m for m in metrics_norm if m not in out]
    if missing:
        raise ValueError(f"--inputs 未提供以下 metric 的输入文件: {missing}")
    return out


def _group_rows_by_model(rows: List[Dict[str, Any]], model_field: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        model = str(r.get(model_field) or "").strip() or "unknown"
        # 若该 metric 输出包含 run_id，则按 model + run_id 分组，确保每次 run 单独聚合
        if "run_id" in r and r.get("run_id") is not None:
            try:
                run_id = int(r.get("run_id") or 0)
            except Exception:
                run_id = 0
            model = f"{model}__run{run_id}"
        grouped.setdefault(model, []).append(r)
    return grouped


def _safe_filename(s: str) -> str:
    # 仅用于 /tmp 临时文件名：把奇怪字符替换掉，避免路径穿越或不可用文件名
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:120] or "unknown"


def aggregate_from_inputs(
    *,
    metrics: List[str],
    inputs_map: Dict[str, str],
    exclude_uuids: str = "",
    judge_model: str = "gemini-2.5-pro",
    group_by_model: bool = False,
) -> Dict[str, Any]:
    """
    聚合多个 metric 的结果，返回可 json 序列化的 dict。

    - group_by_model=False：返回 {metric: flat_metrics}
    - group_by_model=True ：返回 {metric: {model_name: flat_metrics}}
    """
    metric_names = [m.strip().lower() for m in metrics if m.strip()]
    exclude = {u.strip() for u in exclude_uuids.split(",") if u.strip()}

    out: Dict[str, Any] = {}
    for name in metric_names:
        input_path = inputs_map[name]

        # Safety 特殊约束：若没有任何 uuid 能提取到 trap rubric，上游会跳过生成结果文件；
        # 此时该指标不应参与聚合计算（也避免 FileNotFoundError）。
        if name == "safety_pass_rate" and not os.path.exists(input_path):
            print(f"[aggregate] safety_pass_rate 输入文件不存在，跳过该指标: {input_path}")
            continue

        # 针对“同一文件包含多 model”的情况：按 model_name（或 safety 的 source_model_name）分组
        if group_by_model:
            rows = read_jsonl_list(input_path)
            if name == "safety_pass_rate" and not rows:
                # 空文件/无有效样本：不输出该指标
                continue
            if name == "safety_pass_rate":
                grouped = _group_rows_by_model(rows, "source_model_name")
            else:
                grouped = _group_rows_by_model(rows, "model_name")

            metric_out: Dict[str, Any] = {}
            for model_name, model_rows in grouped.items():
                # 复用 Eval 类：将 rows 写入临时 jsonl（避免在各 eval 内重复实现 from_rows）
                # 这里以“简单可靠”为优先：临时文件写在 /tmp，文件名可读且避免冲突。
                tmp_path = f"/tmp/shoppingcompjudge_{name}_{_safe_filename(model_name)}.jsonl"
                write_jsonl(model_rows, tmp_path)
                eval_obj = _create_eval(name, input_path=tmp_path, exclude=exclude, judge_model=judge_model)
                res = eval_obj()
                flat = res.to_flat_metrics()
                if name == "safety_pass_rate" and not flat:
                    # 该 model 下无可评估样本：跳过
                    continue
                metric_out[model_name] = flat
            if name == "safety_pass_rate" and not metric_out:
                continue
            out[name] = metric_out
            continue

        # 单文件聚合
        eval_obj = _create_eval(name, input_path=input_path, exclude=exclude, judge_model=judge_model)
        res = eval_obj()
        flat = res.to_flat_metrics()
        if name == "safety_pass_rate" and not flat:
            continue
        out[name] = flat

    return out


def run_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    metric_names = [m.strip().lower() for m in args.metric.split(",") if m.strip()]
    if len(metric_names) == 1 and metric_names[0] == "all":
        metric_names = ["answer_match", "sop", "scenario_coverage", "rationale_validity", "safety_pass_rate"]

    inputs_map = build_inputs_map(metric_names, args.inputs)
    out = aggregate_from_inputs(
        metrics=metric_names,
        inputs_map=inputs_map,
        exclude_uuids=args.exclude_uuids,
        judge_model=args.judge_model,
        group_by_model=args.group_by_model,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n所有结果已写入 {args.output_json}")

    return out


def main() -> Dict[str, Any]:  # 便于被 python -m 调用
    parser = _build_parser()
    args = parser.parse_args()
    return run_from_args(args)


if __name__ == "__main__":
    main()


