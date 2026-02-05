#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
Safety Rubric Pass Rate 评估输入生成（judger 层）。

输入（推荐）：
- gt_file + pred_file：trap rubric 来自 GT（通常在 scene_list 或 trap_scene 字段中），response 来自预测文件

输入（兼容旧流程）：
- --input: trap evaluator 输入 jsonl，每行含 question / response / trap_scene.rubric 等字段
  - 也兼容 workflow_rebuild 产物：每行含顶层 trap（rubric 文本）+ model_results[*].model/response

输出：
- --output: 每行形如：
  {
    "sample_uuid": ...,
    "source_model_name": ...,        # 被评估的回答来自哪个模型（可选字段）
    "trap_rubric": ...,
    "model_raw_outputs": { judge_model: "<raw>" },
    "model_results": { judge_model: "是" / "否" }   # 是=回答考虑到了陷阱；否=未考虑到
  }

可直接喂给 SafetyPassRateEval。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

from ..api_client import batch_infer
from ..io_utils import read_jsonl_index, read_jsonl_latest_by_key
from ..prompt_loader import format_prompt
from ..tools import TOOL_DEFINITIONS


def iter_jsonl_list(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError as e:
                print(f"[judge_safety] {path} 第{line_num}行 JSON 解析失败: {e}")
                continue
    return items


def _extract_trap_rubric_from_row(row: Dict[str, Any]) -> str:
    """
    优先使用顶层 trap 字段（workflow_rebuild/product_pipeline 产物）。
    兼容旧结构：trap_scene.rubric / trap_scene_list[*].rubric / scene_list[*].rubric。
    """
    trap = row.get("trap_rubric")
    if isinstance(trap, str) and trap.strip():
        return trap.strip()

    trap_scene = row.get("trap_scene")
    if isinstance(trap_scene, dict):
        rubric = trap_scene.get("rubric")
        if isinstance(rubric, str) and rubric.strip():
            return rubric.strip()

    trap_scene_list = row.get("trap_scene_list")
    if isinstance(trap_scene_list, list):
        for ts in trap_scene_list:
            if isinstance(ts, dict):
                rubric = ts.get("rubric")
                if isinstance(rubric, str) and rubric.strip():
                    return rubric.strip()

    scene_list = row.get("scene_list")
    if isinstance(scene_list, list):
        for s in scene_list:
            if isinstance(s, dict):
                rubric = s.get("trap_rubric")
                if isinstance(rubric, str) and rubric.strip():
                    return rubric.strip()

    return ""


def get_model_entries(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    返回 [{model_name, run_id, response}] 列表，仅使用新格式 model_results。
    run_id 为同一 uuid 下同一 model 的出现顺序编号：0,1,2...
    sample_uuid 会在主流程中处理。
    """
    entries: List[Dict[str, Any]] = []
    mr = row.get("model_results")
    if isinstance(mr, list):
        run_counter: Dict[str, int] = {}
        for item in mr:
            model_name = str(item.get("model") or "")
            resp = item.get("response")
            if isinstance(resp, str) and resp.strip():
                run_id = run_counter.get(model_name, 0)
                run_counter[model_name] = run_id + 1
                entries.append({"model_name": model_name, "run_id": run_id, "response": resp})
    else:
        resp = row.get("response")
        if isinstance(resp, str) and resp.strip():
            entries.append({"model_name": "", "run_id": 0, "response": resp})
    return entries


def parse_safety_result(text: str) -> str:
    """
    从 LLM 返回中解析 <result> 是/否 </result>（或 Yes/No），若无则基于关键字判断。
    """
    if not text:
        return "否"
    m = re.search(r"<result>(.*?)</result>", text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        content_lower = content.lower()
        # 中文
        if ("是" in content and "不是" not in content) or content_lower == "yes":
            return "是"
        if ("否" in content) or ("不是" in content) or content_lower == "no":
            return "否"
    # 英文兜底
    text_lower = text.lower()
    if re.search(r"\byes\b", text_lower) and not re.search(r"\bno\b", text_lower):
        return "是"
    if re.search(r"\bno\b", text_lower):
        return "否"
    # 中文兜底
    if "正确" in text and "不正确" not in text:
        return "是"
    return "否"


def generate_safety_rubric_judge_inputs(
    input_file: str,
    output_file: str,
    judge_model: str,
    max_workers: int = 64,
    max_retries: int = 3,
    lang: str = "en",
    resume: bool = True,
    resume_mode: str = "success_only",
    checkpoint_batch_size: int = 20,
) -> None:
    rows = iter_jsonl_list(input_file)
    if not rows:
        print("[judge_safety] 输入文件为空，退出。")
        return

    infer_data: List[Dict[str, Any]] = []
    for row in rows:
        rubric = _extract_trap_rubric_from_row(row)
        if not rubric:
            continue

        question = str(row.get("question") or "").strip()
        model_entries = get_model_entries(row)
        if not model_entries:
            continue

        for entry in model_entries:
            model_name = entry["model_name"]
            run_id = int(entry.get("run_id") or 0)
            response = entry["response"]
            if not question or not response:
                continue

            # 判断是否需要添加工具使用指南（非 gemini-2.5-pro 时需要）
            is_gemini = judge_model and "gemini-2.5-pro" in judge_model.lower()
            include_tool_guide = not is_gemini

            prompt = format_prompt(
                "safety_rubric",
                lang=lang,
                question=question,
                trap_rubric=rubric,
                response=response,
                include_tool_guide=include_tool_guide,
            )
            sample_uuid = row.get("uuid") or row.get("sample_uuid")
            if model_name:
                sample_uuid = f"{sample_uuid}__{model_name}__run{run_id}"
            infer_data.append(
                {
                    "sample_uuid": sample_uuid,
                    "source_model_name": model_name,
                    "run_id": run_id,
                    "trap_rubric": rubric,
                    "question": question,
                    "response": response,
                    "prompt": prompt,
                }
            )

    if not infer_data:
        # 关键约束：只有能提取到 trap rubric 的 uuid 才评测 safety。
        # 若没有任何可评测样本，则不生成（或清理）输出文件，避免下游误用旧结果。
        if output_file and os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception:
                pass
        print("[judge_safety] 没有可评估的数据（无法从输入行中提取 trap rubric），跳过 safety judge。")
        return

    print(f"[judge_safety] 共构造 {len(infer_data)} 条安全 rubric 评估请求")
    
    # 根据 judge_model 选择不同的 tools
    is_gemini = "gemini-2.5-pro" in judge_model.lower()
    tools_to_use = [{"type": "google_search"}] if is_gemini else TOOL_DEFINITIONS

    raw_file = output_file + ".judge_raw.jsonl"
    results = batch_infer(
        infer_data,
        model_name=judge_model,
        tools=tools_to_use,
        thinking_mode=True,
        max_workers=max_workers,
        max_retries=max_retries,
        output_file=raw_file,
        resume=resume,
        resume_key_fields=["sample_uuid"],
        resume_mode=resume_mode,
        checkpoint_batch_size=checkpoint_batch_size,
    )

    try:
        latest = read_jsonl_latest_by_key(raw_file, key_fields=["sample_uuid"], prefer_success=True)
        results_for_write = list(latest.values())
    except Exception:
        results_for_write = results

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results_for_write:
            sample_uuid = r.get("sample_uuid")
            rubric = r.get("trap_rubric")
            source_model_name = r.get("source_model_name")
            run_id = int(r.get("run_id") or 0)
            raw = str(r.get("judge_response") or "")
            label = parse_safety_result(raw)
            out = {
                "sample_uuid": sample_uuid,
                "source_model_name": source_model_name,
                "run_id": run_id,
                "trap_rubric": rubric,
                "model_raw_outputs": {judge_model: raw},
                "model_results": {judge_model: label},
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[judge_safety] 已生成 Safety Rubric judge 结果: {output_file} （{len(results)} 条）")


def _extract_trap_scenes_from_gt(gt_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 GT 行中抽取“trap 场景”列表。

    兼容几种常见结构：
    - gt_row["trap_scene"] = {uuid, scene, rubric}
    - gt_row["trap_scene_list"] = [{...}, ...]
    - 否则回退使用 gt_row["scene_list"]（将每个 scene 当作一个 rubric 评估项）
    """
    ts = gt_row.get("trap_scene")
    if isinstance(ts, dict):
        return [ts]

    tsl = gt_row.get("trap_scene_list")
    if isinstance(tsl, list):
        out = [x for x in tsl if isinstance(x, dict)]
        if out:
            return out

    sl = gt_row.get("scene_list")
    if isinstance(sl, list):
        return [x for x in sl if isinstance(x, dict)]
    return []


def generate_safety_rubric_judge_from_gt_pred(
    gt_file: str,
    pred_file: str,
    output_file: str,
    judge_model: str,
    max_workers: int = 64,
    max_retries: int = 3,
    lang: str = "en",
    resume: bool = True,
    resume_mode: str = "success_only",
    checkpoint_batch_size: int = 20,
) -> None:
    """
    从 GT + 预测文件直接生成 Safety Rubric judge 结果。

    - GT：提供 trap_rubric（来自 trap_scene/trap_scene_list/scene_list）
    - pred：提供 response（来自 model_results[*].response 或顶层 response）
    """
    gt_index = read_jsonl_index(gt_file)
    pred_index = read_jsonl_index(pred_file)
    if not gt_index or not pred_index:
        print("[judge_safety] GT 或 pred 为空，退出。")
        return

    infer_data: List[Dict[str, Any]] = []
    for uuid, gt_row in gt_index.items():
        if uuid not in pred_index:
            continue
        pred_row = pred_index[uuid]
        question = str(gt_row.get("question") or pred_row.get("question") or "").strip()
        if not question:
            continue
        # 关键约束：只有当 _extract_trap_rubric_from_row(gt_row) 非空时，该 uuid 才参与 safety judge。
        # 不再回退使用 trap_scene_list/scene_list 等结构，避免无 trap rubric 的 uuid 被误计入。
        rubric = _extract_trap_rubric_from_row(gt_row)
        if not rubric:
            continue

        model_entries = get_model_entries(pred_row)
        if not model_entries:
            continue

        for entry in model_entries:
            model_name = entry["model_name"]
            run_id = int(entry.get("run_id") or 0)
            response = entry["response"]
            if not response:
                continue

            # 判断是否需要添加工具使用指南（非 gemini-2.5-pro 时需要）
            is_gemini = judge_model and "gemini-2.5-pro" in judge_model.lower()
            include_tool_guide = not is_gemini

            prompt = format_prompt(
                "safety_rubric",
                lang=lang,
                question=question,
                trap_rubric=rubric,
                response=response,
                include_tool_guide=include_tool_guide,
            )

            sample_uuid = uuid
            if model_name:
                sample_uuid = f"{sample_uuid}__{model_name}__run{run_id}"

            infer_data.append(
                {
                    "sample_uuid": sample_uuid,
                    "question_uuid": uuid,
                    "source_model_name": model_name,
                    "run_id": run_id,
                    "trap_rubric": rubric,
                    "question": question,
                    "response": response,
                    "prompt": prompt,
                }
            )

    if not infer_data:
        # 关键约束：若没有任何 uuid 能提取到 trap rubric，则跳过该指标，并清理旧输出，避免下游误用。
        if output_file and os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception:
                pass
        print("[judge_safety] 没有可评估的数据（无 uuid 可提取 trap rubric 或 pred 缺 response），跳过 safety judge。")
        return

    print(f"[judge_safety] 共构造 {len(infer_data)} 条安全 rubric 评估请求（GT+pred）")
    
    # 根据 judge_model 选择不同的 tools
    is_gemini = "gemini-2.5-pro" in judge_model.lower()
    tools_to_use = [{"type": "google_search"}] if is_gemini else TOOL_DEFINITIONS

    raw_file = output_file + ".judge_raw.jsonl"
    results = batch_infer(
        infer_data,
        model_name=judge_model,
        tools=tools_to_use,
        thinking_mode=True,
        max_workers=max_workers,
        max_retries=max_retries,
        output_file=raw_file,
        resume=resume,
        resume_key_fields=["sample_uuid"],
        resume_mode=resume_mode,
        checkpoint_batch_size=checkpoint_batch_size,
    )

    try:
        latest = read_jsonl_latest_by_key(raw_file, key_fields=["sample_uuid"], prefer_success=True)
        results_for_write = list(latest.values())
    except Exception:
        results_for_write = results

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results_for_write:
            sample_uuid = r.get("sample_uuid")
            rubric = r.get("trap_rubric")
            source_model_name = r.get("source_model_name")
            run_id = int(r.get("run_id") or 0)
            raw = str(r.get("judge_response") or "")
            label = parse_safety_result(raw)
            out = {
                "sample_uuid": sample_uuid,
                "question_uuid": r.get("question_uuid"),
                "source_model_name": source_model_name,
                "run_id": run_id,
                "trap_rubric": rubric,
                "model_raw_outputs": {judge_model: raw},
                "model_results": {judge_model: label},
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[judge_safety] 已生成 Safety Rubric judge 结果: {output_file} （{len(results)} 条，GT+pred）")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Safety Rubric judge results with LLM-as-a-Judge")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="（旧）trap evaluator 输入 jsonl")
    group.add_argument("--gt_file", "-g", help="（新）ground-truth jsonl（含 trap rubric）")
    parser.add_argument("--pred_file", "-p", default="", help="（新）prediction jsonl（含 response）")
    parser.add_argument("--output", "-o", required=True, help="输出 JSONL 路径")
    parser.add_argument("--judge_model", "-m", required=True, help="用于 judge 的 LLM 模型名")
    parser.add_argument("--max_workers", type=int, default=64, help="并发线程数")
    parser.add_argument("--lang", choices=["zh", "en"], default="en", help="prompt 语言")

    args = parser.parse_args()
    if args.input:
        generate_safety_rubric_judge_inputs(
            input_file=args.input,
            output_file=args.output,
            judge_model=args.judge_model,
            max_workers=args.max_workers,
            lang=args.lang,
        )
        return

    if not args.pred_file:
        raise SystemExit("当使用 --gt_file 时，必须同时提供 --pred_file")
    generate_safety_rubric_judge_from_gt_pred(
        gt_file=args.gt_file,
        pred_file=args.pred_file,
        output_file=args.output,
        judge_model=args.judge_model,
        max_workers=args.max_workers,
        lang=args.lang,
    )


if __name__ == "__main__":
    main()


