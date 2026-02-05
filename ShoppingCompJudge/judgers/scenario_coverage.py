#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
Scenario Coverage 评估输入生成（judger 层）。

从 ground-truth jsonl + prediction jsonl 出发，调用 LLM 判断：
- demand 是否被 scene_set 覆盖（用于 Precision）
- scene 是否被 demand_set 覆盖（用于 Recall）

生成的 cor_evaluation 结构可直接喂给 ScenarioCoverageEval。
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List, Set, Tuple

from ..api_client import batch_infer
from ..io_utils import read_jsonl_index, read_jsonl_latest_by_key
from ..prompt_loader import format_prompt
from ..tools import TOOL_DEFINITIONS


import json


def extract_scenes_from_gt(scene_list: List[Dict[str, Any]]) -> Set[str]:
    scenes: Set[str] = set()
    for item in scene_list or []:
        scene = (item.get("scene") or "").strip()
        if scene:
            scenes.add(scene)
    return scenes


def is_valid_demand(demand: str) -> bool:
    demand_clean = demand.replace(" ", "")
    pattern = r"\d+年\d+月"
    return re.search(pattern, demand_clean) is None


def extract_demands_from_response(response: str) -> Set[str]:
    demands: Set[str] = set()
    try:
        m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not m:
            return demands
        content = m.group(1)
        matches = re.findall(r"<demand>(.*?)</demand>", content, re.DOTALL)
        for d in matches:
            text = d.strip()
            if text and is_valid_demand(text):
                demands.add(text)
    except Exception as e:
        print(f"[judge_scenario_coverage] 提取 demand 出错: {e}")
    return demands


def get_model_entries(row: Dict[str, Any]) -> List[Tuple[str, int, str]]:
    """
    返回 [(model_name, run_id, response)] 列表，仅使用新格式 model_results。
    run_id 为同一 uuid 下同一 model 的出现顺序编号：0,1,2...
    """
    entries: List[Tuple[str, int, str]] = []
    mr = row.get("model_results")
    if isinstance(mr, list):
        run_counter: Dict[str, int] = {}
        for item in mr:
            model_name = str(item.get("model") or "")
            resp = item.get("response")
            if isinstance(resp, str) and resp.strip():
                run_id = run_counter.get(model_name, 0)
                run_counter[model_name] = run_id + 1
                entries.append((model_name, run_id, resp))
    else:
        resp = row.get("response")
        if isinstance(resp, str) and resp.strip():
            entries.append(("", 0, resp))
    return entries


def parse_result_tag(text: str) -> int:
    if not text:
        return 0
    m = re.search(r"<result>(.*?)</result>", text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        if "1" in content:
            return 1
        if "0" in content:
            return 0
    return 0


def build_infer_data_for_demands(
    gt_index: Dict[str, Dict[str, Any]],
    model_entries: List[Tuple[str, str, int, str]],
    lang: str = "en",
    judge_model: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, int, str], Dict[str, Any]]]:
    infer_data: List[Dict[str, Any]] = []
    meta_index: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}

    for uuid, model_name, run_id, response in model_entries:
        gt_row = gt_index.get(uuid, {})
        question = gt_row.get("question") or ""
        scene_set = extract_scenes_from_gt(gt_row.get("scene_list", []) or [])
        demand_set = extract_demands_from_response(response)
        if not scene_set or not demand_set:
            continue

        scene_block = "\n".join([f"- {s}" for s in scene_set])
        # 判断是否需要添加工具使用指南（非 gemini-2.5-pro 时需要）
        is_gemini = judge_model and "gemini-2.5-pro" in judge_model.lower()
        include_tool_guide = not is_gemini
        
        for demand in demand_set:
            prompt = format_prompt(
                "scenario_coverage_demand2scene",
                lang=lang,
                question=question,
                scene_list_block=scene_block,
                demand=demand,
                include_tool_guide=include_tool_guide,
            )
            item = {
                "uuid": uuid,
                "model_name": model_name,
                "run_id": run_id,
                "demand": demand,
                "prompt": prompt,
            }
            infer_data.append(item)
            meta_index[(uuid, model_name, run_id, demand)] = {
                "uuid": uuid,
                "model_name": model_name,
                "run_id": run_id,
                "demand": demand,
            }

    return infer_data, meta_index


def build_infer_data_for_scenes(
    gt_index: Dict[str, Dict[str, Any]],
    model_entries: List[Tuple[str, str, int, str]],
    lang: str = "en",
    judge_model: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, int, str], Dict[str, Any]]]:
    infer_data: List[Dict[str, Any]] = []
    meta_index: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}

    for uuid, model_name, run_id, response in model_entries:
        gt_row = gt_index.get(uuid, {})
        question = gt_row.get("question") or ""
        scene_set = extract_scenes_from_gt(gt_row.get("scene_list", []) or [])
        demand_set = extract_demands_from_response(response)
        if not scene_set or not demand_set:
            continue

        demand_block = "\n".join([f"- {d}" for d in demand_set])
        # 判断是否需要添加工具使用指南（非 gemini-2.5-pro 时需要）
        is_gemini = judge_model and "gemini-2.5-pro" in judge_model.lower()
        include_tool_guide = not is_gemini
        
        for scene in scene_set:
            prompt = format_prompt(
                "scenario_coverage_scene2demand",
                lang=lang,
                question=question,
                scene=scene,
                demand_list_block=demand_block,
                include_tool_guide=include_tool_guide,
            )
            item = {
                "uuid": uuid,
                "model_name": model_name,
                "run_id": run_id,
                "scene": scene,
                "prompt": prompt,
            }
            infer_data.append(item)
            meta_index[(uuid, model_name, run_id, scene)] = {
                "uuid": uuid,
                "model_name": model_name,
                "run_id": run_id,
                "scene": scene,
            }

    return infer_data, meta_index


def generate_scenario_coverage_inputs(
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
    gt_index = read_jsonl_index(gt_file)
    pred_index = read_jsonl_index(pred_file)

    # 有 GT + 有 prediction 的 uuid
    uuids = [u for u in gt_index.keys() if u in pred_index]
    if not uuids:
        print("[judge_scenario_coverage] 没有可评估的 uuid，退出。")
        return

    model_entries: List[Tuple[str, str, int, str]] = []
    for uuid in uuids:
        pred_row = pred_index[uuid]
        entries = get_model_entries(pred_row)
        for model_name, run_id, resp in entries:
            model_entries.append((uuid, model_name, run_id, resp))

    if not model_entries:
        print("[judge_scenario_coverage] 没有可评估的 (uuid, model) 数据，退出。")
        return

    # 根据 judge_model 选择不同的 tools
    is_gemini = "gemini-2.5-pro" in judge_model.lower()
    tools_to_use = [{"type": "google_search"}] if is_gemini else TOOL_DEFINITIONS
    
    # 构造 demand->scene 判定
    infer_demand, meta_demand = build_infer_data_for_demands(gt_index, model_entries, lang=lang, judge_model=judge_model)
    # 构造 scene->demand 判定
    infer_scene, meta_scene = build_infer_data_for_scenes(gt_index, model_entries, lang=lang, judge_model=judge_model)

    demand_raw = output_file + ".demand_judge_raw.jsonl"
    demand_results = batch_infer(
        infer_demand,
        model_name=judge_model,
        tools=tools_to_use,
        thinking_mode=True,
        max_workers=max_workers,
        max_retries=max_retries,
        output_file=demand_raw,
        resume=resume,
        resume_key_fields=["uuid", "model_name", "run_id", "demand"],
        resume_mode=resume_mode,
        checkpoint_batch_size=checkpoint_batch_size,
    )
    scene_raw = output_file + ".scene_judge_raw.jsonl"
    scene_results = batch_infer(
        infer_scene,
        model_name=judge_model,
        tools=tools_to_use,
        thinking_mode=True,
        max_workers=max_workers,
        max_retries=max_retries,
        output_file=scene_raw,
        resume=resume,
        resume_key_fields=["uuid", "model_name", "run_id", "scene"],
        resume_mode=resume_mode,
        checkpoint_batch_size=checkpoint_batch_size,
    )

    try:
        latest_d = read_jsonl_latest_by_key(
            demand_raw, key_fields=["uuid", "model_name", "run_id", "demand"], prefer_success=True
        )
        demand_results_for_agg = list(latest_d.values())
    except Exception:
        demand_results_for_agg = demand_results
    try:
        latest_s = read_jsonl_latest_by_key(
            scene_raw, key_fields=["uuid", "model_name", "run_id", "scene"], prefer_success=True
        )
        scene_results_for_agg = list(latest_s.values())
    except Exception:
        scene_results_for_agg = scene_results

    # 按 uuid 聚合 demand_matches / scene_matches
    demand_matches_per_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in demand_results_for_agg:
        uuid = r.get("uuid")
        model_name = r.get("model_name") or ""
        run_id = int(r.get("run_id") or 0)
        demand = r.get("demand")
        if not uuid or not demand:
            continue
        judge_raw = str(r.get("judge_response") or "")
        result = parse_result_tag(judge_raw)
        demand_matches_per_key.setdefault((uuid, model_name, run_id), []).append(
            {"demand": demand, "result": result, "analysis": judge_raw}
        )

    scene_matches_per_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in scene_results_for_agg:
        uuid = r.get("uuid")
        model_name = r.get("model_name") or ""
        run_id = int(r.get("run_id") or 0)
        scene = r.get("scene")
        if not uuid or not scene:
            continue
        judge_raw = str(r.get("judge_response") or "")
        result = parse_result_tag(judge_raw)
        scene_matches_per_key.setdefault((uuid, model_name, run_id), []).append(
            {"scene": scene, "result": result, "analysis": judge_raw}
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for uuid, model_name, run_id, response in model_entries:
            gt_row = gt_index.get(uuid, {})
            pred_row = pred_index.get(uuid, {})

            scene_set = extract_scenes_from_gt(gt_row.get("scene_list", []) or [])
            demand_set = extract_demands_from_response(response)

            if not demand_set:
                cor_eval = {
                    "demand_set": [],
                    "scene_set": list(scene_set),
                    "demand_matches": [],
                    "scene_matches": [],
                    "metrics": {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "matched_demands": 0,
                        "matched_scenes": 0,
                        "total_demands": 0,
                        "total_scenes": len(scene_set),
                    },
                }
            else:
                dm = demand_matches_per_key.get((uuid, model_name, run_id), [])
                sm = scene_matches_per_key.get((uuid, model_name, run_id), [])

                matched_demands = sum(1 for x in dm if x["result"] == 1)
                matched_scenes = sum(1 for x in sm if x["result"] == 1)
                total_demands = len(demand_set)
                total_scenes = len(scene_set)

                precision = matched_demands / total_demands if total_demands > 0 else 0.0
                recall = matched_scenes / total_scenes if total_scenes > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                cor_eval = {
                    "demand_set": list(demand_set),
                    "scene_set": list(scene_set),
                    "demand_matches": dm,
                    "scene_matches": sm,
                    "metrics": {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "matched_demands": matched_demands,
                        "matched_scenes": matched_scenes,
                        "total_demands": total_demands,
                        "total_scenes": total_scenes,
                    },
                }

            row = {
                "uuid": uuid,
                "question": gt_row.get("question") or pred_row.get("question") or "",
                "model_name": model_name,
                "run_id": run_id,
                "response": response,
                "cor_evaluation": cor_eval,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[judge_scenario_coverage] 已生成 CoR 输入+结果文件: {output_file} （{len(model_entries)} 行，按 uuid+model+run 粒度）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Scenario Coverage eval inputs with LLM-as-a-Judge")
    parser.add_argument("--gt_file", "-g", required=True, help="ground-truth jsonl 路径")
    parser.add_argument("--pred_file", "-p", required=True, help="prediction jsonl 路径（含 response/<answer>）")
    parser.add_argument("--output_file", "-o", required=True, help="输出 JSONL 路径（带 cor_evaluation）")
    parser.add_argument("--judge_model", "-m", required=True, help="用于 judge 的 LLM 模型名（需在 api_config.yaml 中配置）")
    parser.add_argument("--max_workers", type=int, default=64, help="并发线程数")

    args = parser.parse_args()
    generate_scenario_coverage_inputs(
        gt_file=args.gt_file,
        pred_file=args.pred_file,
        output_file=args.output_file,
        judge_model=args.judge_model,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()


