#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
Rationale Validity (CoP) 评估输入生成（judger 层）。

输入：
- --judge_file: 包含 uuid / product_set / source_file 的判断结果文件
- --prediction_dir: 含有模型原始 response 的 jsonl 目录（response 中有 <candidate_product_list>）

输出：
- --output_file: 每行在原 judge 行基础上新增 cop_evaluation，可直接喂给 RationaleValidityEval。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..api_client import batch_infer
from ..io_utils import read_jsonl_latest_by_key
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
                print(f"[judge_rationale_validity] {path} 第{line_num}行 JSON 解析失败: {e}")
                continue
    return items


def iter_jsonl_map(path: str) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for obj in iter_jsonl_list(path):
        uuid = obj.get("uuid")
        if uuid:
            m[uuid] = obj
    return m


def extract_candidate_products(response: str) -> List[Dict[str, Any]]:
    products: List[Dict[str, Any]] = []
    try:
        m = re.search(r"<candidate_product_list>(.*?)</candidate_product_list>", response, re.DOTALL)
        if not m:
            return products
        content = m.group(1)
        product_blocks = re.findall(r"<product>(.*?)</product>", content, re.DOTALL)
        for pb in product_blocks:
            name_m = re.search(r"<product_name>(.*?)</product_name>", pb, re.DOTALL)
            if not name_m:
                continue
            product_name = name_m.group(1).strip()
            check_list: List[Dict[str, Any]] = []
            for cb in re.findall(r"<check_item>(.*?)</check_item>", pb, re.DOTALL):
                d_m = re.search(r"<demand>(.*?)</demand>", cb, re.DOTALL)
                r_m = re.search(r"<reason>(.*?)</reason>", cb, re.DOTALL)
                s_m = re.search(r"<is_satisfied>(.*?)</is_satisfied>", cb, re.DOTALL)
                if d_m and r_m and s_m:
                    check_list.append(
                        {
                            "demand": d_m.group(1).strip(),
                            "reason": r_m.group(1).strip(),
                            "is_satisfied": s_m.group(1).strip(),
                        }
                    )
            if check_list:
                products.append({"product_name": product_name, "check_list": check_list})
    except Exception as e:
        print(f"[judge_rationale_validity] 提取候选商品时出错: {e}")
    return products


def get_response_from_row(row: Dict[str, Any]) -> str:
    """
    新格式：response 位于 model_results[*].response；若顶层有则优先顶层。
    """
    resp = row.get("response")
    if isinstance(resp, str) and resp.strip():
        return resp
    mr = row.get("model_results")
    if isinstance(mr, list):
        for item in mr:
            r = item.get("response")
            if isinstance(r, str) and r.strip():
                return r
    return ""


def get_response_for_model(row: Dict[str, Any], model_name: str) -> str:
    """
    按 model_name 取对应 response。
    注意：如果显式传入了 model_name 但找不到对应条目，返回空串（避免跨模型误回退）。
    """
    mr = row.get("model_results")
    if isinstance(mr, list):
        if model_name:
            for item in mr:
                if str(item.get("model") or "") == model_name:
                    r = item.get("response")
                    if isinstance(r, str) and r.strip():
                        return r
            # 显式指定了 model_name，但未找到：不要回退到其它模型
            return ""
        # 未指定 model_name：回退首个可用 response
        for item in mr:
            r = item.get("response")
            if isinstance(r, str) and r.strip():
                return r
    resp = row.get("response")
    return resp if isinstance(resp, str) and resp.strip() else ""


def get_response_for_model_run(row: Dict[str, Any], model_name: str, run_id: int) -> str:
    """
    按 (model_name, run_id) 精确取对应 response。
    约定：run_id 是同一 uuid 下同一 model 在 model_results 中的出现顺序编号：0,1,2...
    若找不到该 run，则回退到 get_response_for_model。
    """
    mr = row.get("model_results")
    if isinstance(mr, list) and model_name:
        idx = 0
        for item in mr:
            if str(item.get("model") or "") != model_name:
                continue
            if idx == run_id:
                r = item.get("response")
                if isinstance(r, str) and r.strip():
                    return r
            idx += 1
        # 显式指定 model_name 但 run_id 超界：不要跨模型回退
        return ""
    return get_response_for_model(row, model_name)

def parse_cop_result(text: str) -> Tuple[int, str]:
    if not text:
        return 0, ""
    analysis_m = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL | re.IGNORECASE)
    analysis = analysis_m.group(1).strip() if analysis_m else text.strip()

    result_m = re.search(r"<result>(.*?)</result>", text, re.DOTALL | re.IGNORECASE)
    if result_m:
        content = (result_m.group(1) or "").strip()
        # 只接受明确的 0/1，避免把 "10"、"01"、"true" 等误判
        if content == "1":
            return 1, analysis
        if content == "0":
            return 0, analysis
    return 0, analysis


def build_infer_data(
    judge_items: List[Dict[str, Any]],
    prediction_files: Dict[str, Dict[str, Dict[str, Any]]],
    lang: str = "en",
    judge_model: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, int, str, int], Dict[str, Any]]]:
    """
    构造 batch_infer 所需的 infer_data 列表，以及
    (uuid, model_name, run_id, product_name, check_idx) -> meta 的映射。

    注意：同一 uuid 下会有多个 model_name/run_id（不同模型/多次采样）。
    若 meta_index 不包含 model/run 维度，会发生覆盖，导致不同输出行共享同一份 reason/demand。
    """
    infer_data: List[Dict[str, Any]] = []
    meta_index: Dict[Tuple[str, str, int, str, int], Dict[str, Any]] = {}

    for item in judge_items:
        uuid = item.get("uuid")
        source_file = item.get("source_file") or ""
        model_name = item.get("model_name") or ""
        run_id = int(item.get("run_id") or 0)
        # 新逻辑：不再依赖 product_set，直接评估 response 里解析出的 model_product_list
        # （即 <candidate_product_list> 里的每个 <product> 的每条 <check_item>）
        if not uuid:
            continue
        # prediction_files 的 key 是文件名（basename）。judge_file 里 source_file 可能是 basename 或路径。
        source_key = source_file
        if source_key not in prediction_files and source_file:
            source_key = Path(source_file).name
        # pred_file 单文件模式：若 source_file 不匹配，兜底使用唯一 prediction
        if source_key not in prediction_files:
            if len(prediction_files) == 1:
                source_key = next(iter(prediction_files.keys()))
            else:
                print(f"[judge_rationale_validity] 找不到预测文件 {source_file}")
                continue
        pred_map = prediction_files[source_key]
        if uuid not in pred_map:
            print(f"[judge_rationale_validity] 在 {source_key} 中找不到 uuid={uuid}")
            continue

        pred_row = pred_map[uuid]
        response = get_response_for_model_run(pred_row, model_name, run_id)
        if model_name and not response:
            print(
                f"[judge_rationale_validity] uuid={uuid} 找不到 model={model_name} run_id={run_id} 的 response，跳过"
            )
            continue
        model_product_list = extract_candidate_products(response)

        # 针对 response 中解析出的每个商品，展开其 check_list 中的每条 demand/reason 做判断
        for prod in model_product_list:
            product_name = prod.get("product_name") or ""
            check_list = prod.get("check_list") or []
            if not product_name or not isinstance(check_list, list) or not check_list:
                continue
            for idx, check in enumerate(check_list):
                demand = (check or {}).get("demand")
                reason = (check or {}).get("reason")
                if not demand or not reason:
                    continue
                # 判断是否需要添加工具使用指南（非 gemini-2.5-pro 时需要）
                is_gemini = judge_model and "gemini-2.5-pro" in judge_model.lower()
                include_tool_guide = not is_gemini
                
                prompt = format_prompt(
                    "rationale_validity",
                    lang=lang,
                    product_name=product_name,
                    demand=demand,
                    reason=reason,
                    include_tool_guide=include_tool_guide,
                )
                infer_item = {
                    "uuid": uuid,
                    "model_name": model_name,
                    "run_id": run_id,
                    "product_name": product_name,
                    "check_index": idx,
                    "demand": demand,
                    "reason": reason,
                    "is_satisfied": (check or {}).get("is_satisfied"),
                    "prompt": prompt,
                }
                infer_data.append(infer_item)
                meta_index[(uuid, model_name, run_id, product_name, idx)] = {
                    "uuid": uuid,
                    "model_name": model_name,
                    "run_id": run_id,
                    "product_name": product_name,
                    "check_index": idx,
                    "demand": demand,
                    "reason": reason,
                    "is_satisfied": (check or {}).get("is_satisfied"),
                }

    return infer_data, meta_index


def generate_rationale_validity_inputs(
    judge_file: str,
    prediction_dir: str,
    output_file: str,
    judge_model: str,
    max_workers: int = 64,
    max_retries: int = 3,
    pred_file: str = "",
    lang: str = "en",
    resume: bool = True,
    resume_mode: str = "success_only",
    checkpoint_batch_size: int = 20,
) -> None:
    judge_items = iter_jsonl_list(judge_file)

    # 加载 prediction：
    # 1) 若提供 pred_file：只加载该单文件（用于“仅两份输入：GT+pred”的推荐流程）
    # 2) 否则：加载 prediction_dir 下的所有 jsonl（旧流程兼容）
    prediction_files: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if pred_file:
        name = Path(pred_file).name
        prediction_files[name] = iter_jsonl_map(str(Path(pred_file)))
        print(f"[judge_rationale_validity] 加载预测文件 {name} ({len(prediction_files[name])} 条)")
    else:
        pred_dir = Path(prediction_dir)
        for pred_path in pred_dir.glob("*.jsonl"):
            name = pred_path.name
            prediction_files[name] = iter_jsonl_map(str(pred_path))
            print(f"[judge_rationale_validity] 加载预测文件 {name} ({len(prediction_files[name])} 条)")

    infer_data, meta_index = build_infer_data(judge_items, prediction_files, lang=lang, judge_model=judge_model)
    if not infer_data:
        print("[judge_rationale_validity] 没有可评估的数据，退出。")
        return

    print(f"[judge_rationale_validity] 共构造 {len(infer_data)} 条 (uuid, product, check_item) 评估请求")
    
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
        resume_key_fields=["uuid", "model_name", "run_id", "product_name", "check_index"],
        resume_mode=resume_mode,
        checkpoint_batch_size=checkpoint_batch_size,
    )

    try:
        latest = read_jsonl_latest_by_key(
            raw_file,
            key_fields=["uuid", "model_name", "run_id", "product_name", "check_index"],
            prefer_success=True,
        )
        results_for_agg = list(latest.values())
    except Exception:
        results_for_agg = results

    # 按 (uuid, model_name, run_id) 聚合（避免不同模型/不同采样互相混写）
    per_key_products: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    for r in results_for_agg:
        uuid = r.get("uuid")
        src_model_name = r.get("model_name") or ""
        src_run_id = int(r.get("run_id") or 0)
        p_name = r.get("product_name")
        check_index = r.get("check_index")
        if uuid is None or p_name is None or check_index is None:
            continue
        meta = meta_index.get((uuid, str(src_model_name), int(src_run_id), p_name, int(check_index)))
        if not meta:
            continue

        judge_raw = str(r.get("judge_response") or "")
        acc_result, acc_analysis = parse_cop_result(judge_raw)

        entry = per_key_products.setdefault((uuid, str(src_model_name), int(src_run_id)), {})
        prod_eval = entry.setdefault(
            p_name,
            {
                "product_name": p_name,
                "check_evaluations": [],
                "product_accuracy": 0.0,
            },
        )
        prod_eval["check_evaluations"].append(
            {
                "demand": meta["demand"],
                "reason": meta["reason"],
                "is_satisfied": meta["is_satisfied"],
                "accuracy_result": acc_result,
                "accuracy_analysis": acc_analysis,
            }
        )

    # 计算 per product / per uuid 的准确率，并写出
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in judge_items:
            uuid = item.get("uuid")
            if not uuid:
                continue
            model_name = item.get("model_name") or ""
            run_id = int(item.get("run_id") or 0)
            prods = per_key_products.get((uuid, str(model_name), int(run_id)), {})
            total_checks = 0
            accurate = 0
            product_evaluations: List[Dict[str, Any]] = []

            for p_name, peval in prods.items():
                checks = peval["check_evaluations"]
                p_total = len(checks)
                p_acc = sum(1 for c in checks if c.get("accuracy_result") == 1)
                peval["product_accuracy"] = p_acc / p_total if p_total > 0 else 0.0

                total_checks += p_total
                accurate += p_acc
                product_evaluations.append(peval)

            overall_accuracy = accurate / total_checks if total_checks > 0 else 0.0
            out = item.copy()
            out["cop_evaluation"] = {
                "product_evaluations": product_evaluations,
                "total_checks": total_checks,
                "accurate_checks": accurate,
                "overall_accuracy": overall_accuracy,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[judge_rationale_validity] 已生成 CoP 输入+结果文件: {output_file} （{len(judge_items)} 行）")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Rationale Validity (CoP) eval inputs with LLM-as-a-Judge")
    parser.add_argument("--judge_file", "-j", required=True, help="包含 uuid/product_set/source_file 的判断结果文件")
    parser.add_argument("--prediction_dir", "-p", default="", help="（旧）预测结果 jsonl 目录（含 response/<candidate_product_list>）")
    parser.add_argument("--pred_file", default="", help="（新）单个 prediction jsonl（含 response/<candidate_product_list>）")
    parser.add_argument("--output_file", "-o", required=True, help="输出 JSONL 路径（带 cop_evaluation）")
    parser.add_argument("--judge_model", "-m", required=True, help="用于 judge 的 LLM 模型名")
    parser.add_argument("--max_workers", type=int, default=64, help="并发线程数")

    args = parser.parse_args()
    if not args.pred_file and not args.prediction_dir:
        raise SystemExit("必须提供 --pred_file 或 --prediction_dir 之一")
    generate_rationale_validity_inputs(
        judge_file=args.judge_file,
        prediction_dir=args.prediction_dir,
        output_file=args.output_file,
        judge_model=args.judge_model,
        max_workers=args.max_workers,
        pred_file=args.pred_file,
    )


if __name__ == "__main__":
    main()


