#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
ShoppingCompJudge 统一命令行入口。

设计目标：
- 一个入口覆盖：judgers(LLM-as-a-Judge 生成中间 jsonl) + metrics(聚合统计)
- 参数命名一致、子命令清晰（judge / aggregate / run）
- 保持历史脚本入口仍可用（judgers/*.py 与 runner.py 不强制删除）
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from .runner import aggregate_from_inputs, build_inputs_map
from .hf_data import DEFAULT_HF_DATASET_REPO, resolve_local_or_hf_file


def _add_common_judge_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-model", "-m", required=True, help="用于 LLM-as-a-Judge 的模型名（需在 api_config.yaml 中配置）")
    p.add_argument("--max-workers", type=int, default=64, help="并发线程数")
    p.add_argument("--max-retries", type=int, default=10, help="API 最大重试次数（针对限流/网络等失败）")
    p.add_argument("--lang", choices=["zh", "en"], default="en", help="judge prompt 语言（zh/en）")
    # checkpoint / resume
    p.set_defaults(resume=True)
    p.add_argument("--resume", dest="resume", action="store_true", help="断点续跑：跳过已成功的请求（默认开启）")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="关闭断点续跑（忽略已有 checkpoint）")
    p.add_argument(
        "--resume-mode",
        choices=["success_only", "all"],
        default="success_only",
        help="断点策略：success_only=仅跳过已成功请求（失败会重跑）；all=跳过所有已存在记录",
    )
    p.add_argument("--checkpoint-batch-size", type=int, default=10, help="checkpoint 落盘批大小（越大 I/O 越少，越小越实时）")
    p.add_argument(
        "--api-error-log",
        default="",
        help="API 异常/重试日志 jsonl 路径（默认写到输出目录下 api_errors.log.jsonl）",
    )


def _set_api_error_log_path(path: str) -> None:
    path = (path or "").strip()
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    os.environ["SHOPPINGCOMPJUDGE_API_ERROR_LOG_PATH"] = path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ShoppingCompJudge", description="ShoppingCompJudge: judge + aggregate for ShoppingComp.")
    parser.add_argument(
        "--hf-dataset-repo",
        default=DEFAULT_HF_DATASET_REPO,
        help=(
            "Hugging Face dataset repo id used for auto-downloading GT/trap files when a local path does not exist. "
            f'Default: "{DEFAULT_HF_DATASET_REPO}". You can also set env SHOPPINGCOMP_HF_DATASET_REPO.'
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # judge/*
    judge = sub.add_parser("judge", help="调用 LLM-as-a-Judge 生成各指标所需的中间 jsonl")
    judge_sub = judge.add_subparsers(dest="judge_metric", required=True)

    p = judge_sub.add_parser("answer_match", help="生成 AnswerMatch 输入（gt+step3 -> answer_match_inputs.jsonl）")
    p.add_argument("--gt", "-g", required=True, help="ground-truth jsonl")
    p.add_argument("--pred", "-p", required=True, help="step3 结果 jsonl（products 列表）")
    p.add_argument("--out", "-o", required=True, help="输出 jsonl")
    _add_common_judge_args(p)

    p = judge_sub.add_parser("sop", help="生成 SoP 输入（gt+step3 -> sop_inputs.jsonl）")
    p.add_argument("--gt", "-g", required=True, help="ground-truth jsonl")
    p.add_argument("--pred", "-p", required=True, help="step3 结果 jsonl（products 列表）")
    p.add_argument("--out", "-o", required=True, help="输出 jsonl")
    _add_common_judge_args(p)

    p = judge_sub.add_parser("scenario_coverage", help="生成 Scenario Coverage 结果（gt+prediction_with_answer -> cor_results.jsonl）")
    p.add_argument("--gt", "-g", required=True, help="ground-truth jsonl")
    p.add_argument("--pred", "-p", required=True, help="prediction jsonl（含 response/<answer>）")
    p.add_argument("--out", "-o", required=True, help="输出 jsonl（带 cor_evaluation）")
    _add_common_judge_args(p)

    p = judge_sub.add_parser("rationale_validity", help="生成 Rationale Validity 结果（answer_match_inputs + prediction_dir -> cop_results.jsonl）")
    p.add_argument("--judge-file", "-j", required=True, help="answer_match_inputs.jsonl（含 product_set/source_file/model_name）")
    p.add_argument("--pred", required=False, default="", help="（推荐）单个 prediction jsonl（response 含 <candidate_product_list>）")
    p.add_argument("--prediction-dir", required=False, default="", help="（兼容旧）prediction jsonl 目录（response 含 <candidate_product_list>）")
    p.add_argument("--out", "-o", required=True, help="输出 jsonl（带 cop_evaluation）")
    _add_common_judge_args(p)
    p.set_defaults(max_workers=64)

    p = judge_sub.add_parser("safety_pass_rate", help="生成 Safety Rubric judge 结果（trap_input -> trap_rubric_eval_results.jsonl）")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", "-i", help="（旧）trap evaluator 输入 jsonl")
    g.add_argument("--gt", help="（新）ground-truth jsonl（含 trap rubric）")
    p.add_argument("--pred", default="", help="（新）prediction jsonl（含 response）")
    p.add_argument("--out", "-o", required=True, help="输出 jsonl")
    _add_common_judge_args(p)
    p.set_defaults(max_workers=64)

    # aggregate/*
    agg = sub.add_parser("aggregate", help="不调用 LLM，仅对中间 jsonl 做统计聚合")
    agg.add_argument(
        "--metric",
        required=True,
        help="要聚合的 metric，逗号分隔：answer_match,sop,scenario_coverage,rationale_validity,safety_pass_rate 或 all",
    )
    agg.add_argument(
        "--inputs",
        required=True,
        help="必选：多输入映射：metric=/path,metric=/path,...（新模式要求对每个 metric 显式提供）。",
    )
    agg.add_argument("--exclude-uuids", default="", help="仅 answer_match 生效，要排除的 uuid，逗号分隔。")
    agg.add_argument(
        "--judge-model",
        default="gemini-2.5-pro",
        help="仅 safety_pass_rate 生效：使用哪个 judge_model key 读取 model_results。",
    )
    agg.add_argument(
        "--group-by-model",
        action="store_true",
        help="若输入包含多个 model_name，则按 model_name 分组聚合并分别输出。",
    )
    agg.add_argument("--output-json", default="", help="可选：将所有结果写入 json 文件。")
    agg.add_argument("--quiet", action="store_true", help="不在 stdout 打印详细结果（只写 output-json）。")

    # run（便捷 pipeline）
    run = sub.add_parser("run", help="一键：生成所有中间文件并聚合 all 指标（需要提供对应输入）")
    run.add_argument("--gt", required=True, help="ground-truth jsonl")
    run.add_argument("--pred", required=True, help="prediction jsonl（同一份文件同时提供 products + response）")
    run.add_argument("--out-dir", required=True, help="输出目录（将生成各指标中间文件 + 汇总 json）")
    run.add_argument("--judge-model", "-m", required=True, help="用于 LLM-as-a-Judge 的模型名")
    run.add_argument(
        "--metric",
        default="all",
        help="仅 run 子命令生效：要跑哪些指标，逗号分隔：answer_match,sop,scenario_coverage,rationale_validity,safety_pass_rate 或 all（默认 all）。",
    )
    run.add_argument("--max-workers", type=int, default=64, help="并发线程数（answer_match/sop/scenario_coverage 使用）")
    run.add_argument("--max-workers-rv", type=int, default=64, help="RV 并发线程数")
    run.add_argument("--max-workers-safety", type=int, default=64, help="Safety 并发线程数")
    run.add_argument("--max-retries", type=int, default=3, help="API 最大重试次数（针对限流/网络等失败）")
    run.add_argument("--lang", choices=["zh", "en"], default="en", help="judge prompt 语言（zh/en）")
    run.add_argument("--output-json", default="", help="可选：汇总写入该文件；默认写到 out-dir/metrics.json")
    run.set_defaults(resume=True)
    run.add_argument("--resume", dest="resume", action="store_true", help="断点续跑：跳过已成功的请求（默认开启）")
    run.add_argument("--no-resume", dest="resume", action="store_false", help="关闭断点续跑（忽略已有 checkpoint）")
    run.add_argument(
        "--resume-mode",
        choices=["success_only", "all"],
        default="success_only",
        help="断点策略：success_only=仅跳过已成功请求（失败会重跑）；all=跳过所有已存在记录",
    )
    run.add_argument("--checkpoint-batch-size", type=int, default=10, help="checkpoint 落盘批大小（越大 I/O 越少，越小越实时）")
    run.add_argument(
        "--api-error-log",
        default="",
        help="API 异常/重试日志 jsonl 路径（默认写到 out-dir/api_errors.log.jsonl）",
    )

    return parser


def _cmd_judge(args: argparse.Namespace) -> int:
    # 设置 API error log 默认路径
    out_dir = os.path.dirname(args.out) or "."
    error_log = args.api_error_log.strip() if getattr(args, "api_error_log", "") else os.path.join(out_dir, "api_errors.log.jsonl")
    _set_api_error_log_path(error_log)

    # Auto-download GT / trap inputs from Hugging Face if needed (keeps downstream APIs file-based).
    hf_token = os.environ.get("HF_TOKEN")
    if getattr(args, "gt", ""):
        args.gt, _ = resolve_local_or_hf_file(args.gt, default_repo_id=args.hf_dataset_repo, hf_token=hf_token)
    if getattr(args, "input", ""):
        args.input, _ = resolve_local_or_hf_file(args.input, default_repo_id=args.hf_dataset_repo, hf_token=hf_token)

    # 延迟 import，避免 aggregate-only 场景加载 openai 等依赖导致用户困惑
    if args.judge_metric == "answer_match":
        from .judgers.answer_match import generate_answer_match_inputs

        generate_answer_match_inputs(
            args.gt,
            args.pred,
            args.out,
            args.judge_model,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            lang=args.lang,
            resume=args.resume,
            resume_mode=args.resume_mode,
            checkpoint_batch_size=args.checkpoint_batch_size,
        )
        return 0
    if args.judge_metric == "sop":
        from .judgers.sop import generate_sop_inputs

        generate_sop_inputs(
            args.gt,
            args.pred,
            args.out,
            args.judge_model,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            lang=args.lang,
            resume=args.resume,
            resume_mode=args.resume_mode,
            checkpoint_batch_size=args.checkpoint_batch_size,
        )
        return 0
    if args.judge_metric == "scenario_coverage":
        from .judgers.scenario_coverage import generate_scenario_coverage_inputs

        generate_scenario_coverage_inputs(
            args.gt,
            args.pred,
            args.out,
            args.judge_model,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            lang=args.lang,
            resume=args.resume,
            resume_mode=args.resume_mode,
            checkpoint_batch_size=args.checkpoint_batch_size,
        )
        return 0
    if args.judge_metric == "rationale_validity":
        from .judgers.rationale_validity import generate_rationale_validity_inputs

        if not args.pred and not args.prediction_dir:
            raise ValueError("rationale_validity 需要提供 --pred 或 --prediction-dir")
        generate_rationale_validity_inputs(
            args.judge_file,
            args.prediction_dir,
            args.out,
            args.judge_model,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            pred_file=args.pred,
            lang=args.lang,
            resume=args.resume,
            resume_mode=args.resume_mode,
            checkpoint_batch_size=args.checkpoint_batch_size,
        )
        return 0
    if args.judge_metric == "safety_pass_rate":
        from .judgers.safety_pass_rate import (
            generate_safety_rubric_judge_from_gt_pred,
            generate_safety_rubric_judge_inputs,
        )

        if args.input:
            generate_safety_rubric_judge_inputs(
                input_file=args.input,
                output_file=args.out,
                judge_model=args.judge_model,
                max_workers=args.max_workers,
                max_retries=args.max_retries,
                lang=args.lang,
                resume=args.resume,
                resume_mode=args.resume_mode,
                checkpoint_batch_size=args.checkpoint_batch_size,
            )
        else:
            if not args.pred:
                raise ValueError("safety_pass_rate 使用 --gt 时必须同时提供 --pred")
            generate_safety_rubric_judge_from_gt_pred(
                gt_file=args.gt,
                pred_file=args.pred,
                output_file=args.out,
                judge_model=args.judge_model,
                max_workers=args.max_workers,
                max_retries=args.max_retries,
                lang=args.lang,
                resume=args.resume,
                resume_mode=args.resume_mode,
                checkpoint_batch_size=args.checkpoint_batch_size,
            )
        return 0
    raise ValueError(f"未知 judge_metric: {args.judge_metric}")


def _cmd_aggregate(args: argparse.Namespace) -> int:
    metrics = [m.strip() for m in args.metric.split(",") if m.strip()]
    if len(metrics) == 1 and metrics[0].lower() == "all":
        metrics = ["answer_match", "sop", "scenario_coverage", "rationale_validity", "safety_pass_rate"]

    inputs_map = build_inputs_map(metrics=metrics, inputs_kv=args.inputs)
    results = aggregate_from_inputs(
        metrics=metrics,
        inputs_map=inputs_map,
        exclude_uuids=args.exclude_uuids,
        judge_model=args.judge_model,
        group_by_model=args.group_by_model,
    )

    # stdout 展示
    if not args.quiet:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    # 文件输出
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # 设置 API error log 默认路径
    error_log = args.api_error_log.strip() if getattr(args, "api_error_log", "") else os.path.join(out_dir, "api_errors.log.jsonl")
    _set_api_error_log_path(error_log)

    # Auto-download GT from Hugging Face if needed (prediction is usually local).
    hf_token = os.environ.get("HF_TOKEN")
    args.gt, _ = resolve_local_or_hf_file(args.gt, default_repo_id=args.hf_dataset_repo, hf_token=hf_token)

    answer_match_inputs = os.path.join(out_dir, "answer_match_inputs.jsonl")
    sop_inputs = os.path.join(out_dir, "sop_inputs.jsonl")
    cor_results = os.path.join(out_dir, "cor_results.jsonl")
    cop_results = os.path.join(out_dir, "cop_results.jsonl")
    safety_results = os.path.join(out_dir, "trap_rubric_eval_results.jsonl")

    # run 指标选择
    metric_names = [m.strip().lower() for m in (args.metric or "").split(",") if m.strip()]
    if len(metric_names) == 1 and metric_names[0] == "all":
        metric_names = ["answer_match", "sop", "scenario_coverage", "rationale_validity", "safety_pass_rate"]
    if not metric_names:
        raise ValueError("--metric 不能为空")

    inputs_map: Dict[str, str] = {}

    # judge 生成（按需）
    if "answer_match" in metric_names or "rationale_validity" in metric_names:
        from .judgers.answer_match import generate_answer_match_inputs

        generate_answer_match_inputs(
            args.gt,
            args.pred,
            answer_match_inputs,
            args.judge_model,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            lang=args.lang,
            resume=args.resume,
            resume_mode=args.resume_mode,
            checkpoint_batch_size=args.checkpoint_batch_size,
        )
        inputs_map["answer_match"] = answer_match_inputs

    if "sop" in metric_names:
        from .judgers.sop import generate_sop_inputs

        generate_sop_inputs(
            args.gt,
            args.pred,
            sop_inputs,
            args.judge_model,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            lang=args.lang,
            resume=args.resume,
            resume_mode=args.resume_mode,
            checkpoint_batch_size=args.checkpoint_batch_size,
        )
        inputs_map["sop"] = sop_inputs

    if "scenario_coverage" in metric_names:
        from .judgers.scenario_coverage import generate_scenario_coverage_inputs

        generate_scenario_coverage_inputs(
            args.gt,
            args.pred,
            cor_results,
            args.judge_model,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            lang=args.lang,
            resume=args.resume,
            resume_mode=args.resume_mode,
            checkpoint_batch_size=args.checkpoint_batch_size,
        )
        inputs_map["scenario_coverage"] = cor_results

    if "rationale_validity" in metric_names:
        if "answer_match" not in inputs_map:
            raise ValueError("rationale_validity 依赖 answer_match：请在 --metric 中同时包含 answer_match")
        from .judgers.rationale_validity import generate_rationale_validity_inputs

        generate_rationale_validity_inputs(
            answer_match_inputs,
            prediction_dir="",
            output_file=cop_results,
            judge_model=args.judge_model,
            max_workers=args.max_workers_rv,
            max_retries=args.max_retries,
            pred_file=args.pred,
            lang=args.lang,
            resume=args.resume,
            resume_mode=args.resume_mode,
            checkpoint_batch_size=args.checkpoint_batch_size,
        )
        inputs_map["rationale_validity"] = cop_results

    if "safety_pass_rate" in metric_names:
        from .judgers.safety_pass_rate import generate_safety_rubric_judge_from_gt_pred

        generate_safety_rubric_judge_from_gt_pred(
            gt_file=args.gt,
            pred_file=args.pred,
            output_file=safety_results,
            judge_model=args.judge_model,
            max_workers=args.max_workers_safety,
            max_retries=args.max_retries,
            lang=args.lang,
            resume=args.resume,
            resume_mode=args.resume_mode,
            checkpoint_batch_size=args.checkpoint_batch_size,
        )
        inputs_map["safety_pass_rate"] = safety_results

    # aggregate（只聚合所选 metric）
    metrics = metric_names
    results = aggregate_from_inputs(
        metrics=metrics,
        inputs_map=inputs_map,
        exclude_uuids="",
        judge_model=args.judge_model,
        group_by_model=True,
    )

    out_json = args.output_json or os.path.join(out_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[run] 汇总指标已写入: {out_json}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "judge":
        return _cmd_judge(args)
    if args.command == "aggregate":
        return _cmd_aggregate(args)
    if args.command == "run":
        return _cmd_run(args)
    raise ValueError(f"未知 command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
