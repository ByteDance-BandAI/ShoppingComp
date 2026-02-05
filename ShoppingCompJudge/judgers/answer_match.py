#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
Answer Match 评估输入生成（judger 层）。

从 ground-truth jsonl + 模型结果 jsonl 出发，调用 LLM 判断
“候选商品是否匹配任一 GT 商品”，生成 AnswerMatchEval 所需的输入文件。
"""

from __future__ import annotations

import argparse
import os
import re
import math
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from ..api_client import batch_infer
from ..io_utils import extract_model_items, get_question, model_products, read_gt_index_light, read_pred_index_products_only, read_jsonl_latest_by_key, write_jsonl
from ..prompt_loader import format_prompt

from ..tools import TOOL_DEFINITIONS


RunKey = Tuple[str, str, int]


def extract_gt_products(gt_row: Dict[str, Any]) -> List[str]:
    products = []
    for item in gt_row.get("product_list", []) or []:
        name = item.get("product_name")
        if name:
            products.append(str(name).strip())
    return products


def parse_match_result(text: str) -> int:
    if not text:
        return 0

    result_block = re.search(r"<result>(.*?)</result>", text, re.DOTALL | re.IGNORECASE)
    if result_block:
        content = result_block.group(1).strip()
        if "1" in content:
            return 1
        if "0" in content:
            return 0

    lowered = text.lower()
    if "不匹配" in text or "no match" in lowered:
        return 0
    if "匹配" in text or "match" in lowered:
        return 1
    return 0


def build_infer_data(
    gt_index: Dict[str, Dict[str, Any]],
    pred_index: Dict[str, Dict[str, Any]],
    source_file: str,
    lang: str = "en",
    judge_model: str | None = None,
) -> Tuple[Iterator[Dict[str, Any]], Dict[RunKey, Dict[str, Any]], int]:
    """
    返回：
    - infer_items: 迭代器（避免一次性把所有 prompt/task 堆进内存）
    - run_meta: (uuid, model_name, run_id) -> 轻量元信息（每个 run 一份，避免为每个 candidate 复制大列表）
    - total_cnt: 估计/真实总任务数（用于日志）
    """

    def _iter() -> Iterator[Dict[str, Any]]:
        for uuid, gt_row in gt_index.items():
            if uuid not in pred_index:
                continue
            pred_row = pred_index[uuid]
            gt_products = extract_gt_products(gt_row)
            model_items = extract_model_items(pred_row)
            if not gt_products or not model_items:
                continue

            gt_block = "\n".join([f"- {name}" for name in gt_products])

            # 兼容“同一 uuid 下同一 model 多次运行”的情况：按出现顺序编号 run_id=0,1,2...
            run_counter: Dict[str, int] = {}
            for model_item in model_items:
                source_model_name, model_product_list = model_products(model_item)
                if not model_product_list:
                    continue
                run_id = run_counter.get(source_model_name, 0)
                run_counter[source_model_name] = run_id + 1

                rk: RunKey = (uuid, source_model_name, run_id)
                if rk not in run_meta:
                    run_meta[rk] = {
                        "uuid": uuid,
                        "run_id": run_id,
                        "model_name": source_model_name,
                        "source_file": source_file,
                        "gt_product_list": gt_products,
                        "model_product_list": model_product_list,
                        "question": get_question(uuid, gt_index, pred_index),
                    }

                # 判断是否需要添加工具使用指南（非 gemini-2.5-pro 时需要）
                is_gemini = judge_model and "gemini-2.5-pro" in judge_model.lower()
                include_tool_guide = not is_gemini

                for candidate in model_product_list:
                    prompt = format_prompt(
                        "answer_match",
                        lang=lang,
                        gt_products_block=gt_block,
                        candidate_product=candidate,
                        include_tool_guide=include_tool_guide,
                    )
                    yield {
                        "uuid": uuid,
                        "model_name": source_model_name,
                        "run_id": run_id,
                        "candidate_product": candidate,
                        "prompt": prompt,
                    }

    # 预估总数：不生成 prompt，快速计数（节省内存）
    total_cnt = 0
    run_meta: Dict[RunKey, Dict[str, Any]] = {}
    for uuid, gt_row in gt_index.items():
        if uuid not in pred_index:
            continue
        pred_row = pred_index[uuid]
        gt_products = extract_gt_products(gt_row)
        model_items = extract_model_items(pred_row)
        if not gt_products or not model_items:
            continue
        for model_item in model_items:
            _, model_product_list = model_products(model_item)
            total_cnt += len(model_product_list)

    return _iter(), run_meta, total_cnt


def aggregate_results(
    results: List[Dict[str, Any]],
    run_meta: Dict[RunKey, Dict[str, Any]],
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    per_uuid_model_run: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    for result in results:
        uuid = str(result.get("uuid") or "")
        # 这里的 model_name 表示“被评估/来源模型”，不能被 judge 模型覆盖（见 api_client.py）
        model_name = str(result.get("model_name") or "")
        run_id = int(result.get("run_id") or 0)
        candidate = result.get("candidate_product")
        if not uuid or not candidate:
            continue

        meta = run_meta.get((uuid, model_name, run_id))
        if not meta:
            continue

        entry = per_uuid_model_run.setdefault(
            (uuid, model_name, run_id),
            {
                "gt_product_list": meta.get("gt_product_list") or [],
                # 注意：不要跨 run 合并 products；每行对应 (uuid, model, run_id) 的一次输出
                "model_product_list": meta.get("model_product_list") or [],
                "product_set": [],
                "question": meta.get("question") or "",
                "source_file": meta.get("source_file") or "",
                "model_name": model_name,
                "run_id": run_id,
            },
        )

        if not result.get("judge_success"):
            continue

        match_flag = parse_match_result(str(result.get("judge_response") or ""))
        if match_flag == 1 and candidate not in entry["product_set"]:
            entry["product_set"].append(candidate)

    return per_uuid_model_run


def write_answer_match_file(
    per_uuid_model_run: Dict[Tuple[str, str, int], Dict[str, Any]], output_file: str
) -> None:
    rows: List[Dict[str, Any]] = []
    for (uuid, model_name, run_id), data in per_uuid_model_run.items():
        rows.append(
            {
                "uuid": uuid,
                "question": data["question"],
                "gt_product_list": data["gt_product_list"],
                "model_product_list": data["model_product_list"],
                "product_set": data["product_set"],
                "source_file": data["source_file"],
                "model_name": model_name,
                "run_id": run_id,
            }
        )
    write_jsonl(rows, output_file)


def generate_answer_match_inputs(
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
    # 只做 products 相关评估：不需要把超长 response 读进内存（容易触发 memcg OOM）
    gt_index = read_gt_index_light(gt_file)
    pred_index = read_pred_index_products_only(pred_file)

    source_file = os.path.basename(pred_file)
    infer_items, run_meta, total_cnt = build_infer_data(
        gt_index, pred_index, source_file, lang=lang, judge_model=judge_model
    )

    if total_cnt <= 0:
        print("[judge_answer_match] 没有可评估的数据，退出。")
        return

    print(f"[judge_answer_match] 共构造 {total_cnt} 条 (uuid, candidate_product) 评估请求")
    
    # 根据 judge_model 选择不同的 tools
    is_gemini = "gemini-2.5-pro" in judge_model.lower()
    tools_to_use = [{"type": "google_search"}] if is_gemini else TOOL_DEFINITIONS

    # raw checkpoint：用于断点续跑（每个 candidate 一条）
    raw_file = output_file + ".judge_raw.jsonl"
    # 分块推理：避免一次性把 1w+ prompt/task 堆进内存
    chunk_size = int(os.environ.get("SHOPPINGCOMPJUDGE_CHUNK_SIZE", "2000") or "2000")
    chunk_size = max(50, chunk_size)
    chunks = int(math.ceil(total_cnt / chunk_size))
    buf: List[Dict[str, Any]] = []
    seen = 0
    chunk_idx = 0
    for item in infer_items:
        buf.append(item)
        if len(buf) >= chunk_size:
            chunk_idx += 1
            print(f"[judge_answer_match] chunk {chunk_idx}/{chunks}: items={len(buf)}")
            batch_infer(
                buf,
                model_name=judge_model,
                tools=tools_to_use,
                thinking_mode=True,
                max_workers=max_workers,
                max_retries=max_retries,
                output_file=raw_file,
                resume=resume,
                resume_key_fields=["uuid", "model_name", "run_id", "candidate_product"],
                resume_mode=resume_mode,
                checkpoint_batch_size=checkpoint_batch_size,
            )
            seen += len(buf)
            buf = []
    if buf:
        chunk_idx += 1
        print(f"[judge_answer_match] chunk {chunk_idx}/{chunks}: items={len(buf)}")
        batch_infer(
            buf,
            model_name=judge_model,
            tools=tools_to_use,
            thinking_mode=True,
            max_workers=max_workers,
            max_retries=max_retries,
            output_file=raw_file,
            resume=resume,
            resume_key_fields=["uuid", "model_name", "run_id", "candidate_product"],
            resume_mode=resume_mode,
            checkpoint_batch_size=checkpoint_batch_size,
        )
        seen += len(buf)
    print(f"[judge_answer_match] chunks done: {seen}/{total_cnt}")

    # 断点续跑场景：需要用 raw_file 的“最新去重结果”做聚合（避免只聚合本次 results）
    try:
        latest = read_jsonl_latest_by_key(
            raw_file, key_fields=["uuid", "model_name", "run_id", "candidate_product"], prefer_success=True
        )
        results_for_agg = list(latest.values())
    except Exception:
        results_for_agg = []

    per_uuid_model_run = aggregate_results(results_for_agg, run_meta)
    write_answer_match_file(per_uuid_model_run, output_file)

    print(
        f"[judge_answer_match] 已生成 AnswerMatch 输入文件: {output_file} （{len(per_uuid_model_run)} 行，按 uuid+model+run 粒度）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AnswerMatch eval inputs with LLM-as-a-Judge")
    parser.add_argument("--gt_file", "-g", required=True, help="ground-truth jsonl 路径")
    parser.add_argument("--pred_file", "-p", required=True, help="step3 模型结果 jsonl 路径")
    parser.add_argument("--output_file", "-o", required=True, help="输出 JSONL 路径")
    parser.add_argument("--judge_model", "-m", required=True, help="用于 judge 的 LLM 模型名（需在 api_config.yaml 中配置）")
    parser.add_argument("--max_workers", type=int, default=64, help="并发线程数")

    args = parser.parse_args()
    generate_answer_match_inputs(
        gt_file=args.gt_file,
        pred_file=args.pred_file,
        output_file=args.output_file,
        judge_model=args.judge_model,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()


