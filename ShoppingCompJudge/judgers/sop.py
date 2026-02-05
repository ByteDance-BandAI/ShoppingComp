#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
SoP（Selection Accuracy）评估输入生成（judger 层）。

从 ground-truth jsonl + 结果 jsonl 出发，调用 LLM 判断
“(scene, product) 是否满足 rubric”，生成 SoPEval 所需的 eval_product_list。
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple

from ..api_client import batch_infer
from ..io_utils import (
    extract_model_items,
    get_question,
    model_products,
    read_gt_index_light,
    read_pred_index_products_only,
    read_jsonl_latest_by_key,
    write_jsonl,
)
from ..prompt_loader import format_prompt
from ..tools import TOOL_DEFINITIONS


_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<tag>\s*(.*?)\s*</tag>", re.DOTALL | re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"<evidence>\s*(.*?)\s*</evidence>", re.DOTALL | re.IGNORECASE)


def parse_judge_answer(text: str) -> Dict[str, str]:
    """
    Parse judge response for SoP.

    New prompt (sop.en) expects:
      <answer>
        <tag>Satisfied/Not satisfied/Price not satisfied/Unable to judge</tag>
        <evidence>...</evidence>
      </answer>

    Legacy prompt may return:
      <answer>满足/不满足/无法判断</answer>
    """
    if not text:
        return {"answer_block": "", "tag": "", "evidence": "", "raw_inner": ""}

    s = str(text).strip()
    if not s:
        return {"answer_block": "", "tag": "", "evidence": "", "raw_inner": ""}

    m = _ANSWER_BLOCK_RE.search(s)
    answer_block = (m.group(1) or "").strip() if m else s

    tag = ""
    evidence = ""
    mt = _TAG_RE.search(answer_block)
    if mt:
        tag = (mt.group(1) or "").strip()
    me = _EVIDENCE_RE.search(answer_block)
    if me:
        evidence = (me.group(1) or "").strip()

    # Legacy: if no <tag>, treat the whole <answer> inner as tag-like answer.
    raw_inner = answer_block.strip()
    if not tag and raw_inner and not _TAG_RE.search(s):
        tag = raw_inner

    return {"answer_block": answer_block, "tag": tag, "evidence": evidence, "raw_inner": raw_inner}


def build_infer_data(
    gt_index: Dict[str, Dict[str, Any]],
    pred_index: Dict[str, Dict[str, Any]],
    source_file: str,
    lang: str = "en",
    judge_model: str | None = None,
) -> Tuple[Iterator[Dict[str, Any]], int]:
    """
    返回：
    - infer_items: 迭代器（避免一次性把所有 prompt/task 堆进内存）
    - total_cnt: 总任务数（用于日志）
    """

    # 先快速计数（不生成 prompt）
    total_cnt = 0
    for uuid, gt_row in gt_index.items():
        if uuid not in pred_index:
            continue
        pred_row = pred_index[uuid]
        scenes = gt_row.get("scene_list", []) or []
        model_items = extract_model_items(pred_row)
        if not scenes or not model_items:
            continue
        for model_item in model_items:
            _, model_product_list = model_products(model_item)
            if not model_product_list:
                continue
            total_cnt += len(model_product_list) * len([s for s in scenes if (s.get("uuid") and s.get("scene") and s.get("rubric"))])

    def _iter() -> Iterator[Dict[str, Any]]:
        for uuid, gt_row in gt_index.items():
            if uuid not in pred_index:
                continue
            pred_row = pred_index[uuid]

            question = gt_row.get("question") or pred_row.get("question") or ""
            scenes = gt_row.get("scene_list", []) or []
            model_items = extract_model_items(pred_row)
            if not scenes or not model_items:
                continue

            # 兼容“同一 uuid 下同一 model 多次运行”的情况：
            # - 若 prediction 的 model_results 条目里已提供 run_id（或 runId），则优先使用该值；
            # - 否则回退为“按出现顺序编号” run_id=0,1,2...
            run_counter: Dict[str, int] = {}
            for model_item in model_items:
                source_model_name, model_product_list = model_products(model_item)
                if not model_product_list:
                    continue

                explicit_run_id = None
                if "run_id" in model_item and model_item.get("run_id") is not None:
                    explicit_run_id = model_item.get("run_id")
                elif "runId" in model_item and model_item.get("runId") is not None:
                    explicit_run_id = model_item.get("runId")

                if explicit_run_id is not None:
                    try:
                        run_id = int(explicit_run_id)
                    except Exception:
                        run_id = run_counter.get(source_model_name, 0)
                    run_counter[source_model_name] = max(run_counter.get(source_model_name, 0), run_id + 1)
                else:
                    run_id = run_counter.get(source_model_name, 0)
                    run_counter[source_model_name] = run_id + 1

                # 判断是否需要添加工具使用指南（非 gemini-2.5-pro 时需要）
                is_gemini = judge_model and "gemini-2.5-pro" in judge_model.lower()
                include_tool_guide = not is_gemini

                for p_name in model_product_list:
                    for scene_item in scenes:
                        scene_uuid = scene_item.get("uuid") or ""
                        scene = scene_item.get("scene") or ""
                        rubric = scene_item.get("rubric") or ""
                        if not scene_uuid or not scene or not rubric:
                            continue

                        prompt = format_prompt(
                            "sop",
                            lang=lang,
                            question=question,
                            scene=scene,
                            rubric=rubric,
                            product_name=p_name,
                            include_tool_guide=include_tool_guide,
                        )
                        yield {
                            "uuid": uuid,
                            "model_name": source_model_name,
                            "run_id": run_id,
                            "product_name": p_name,
                            "scene_uuid": scene_uuid,
                            "scene": scene,
                            "rubric": rubric,
                            "prompt": prompt,
                            # 保留 source_file 供下游输出
                            "source_file": source_file,
                        }

    return _iter(), total_cnt


def generate_sop_inputs(
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
    # SoP 只依赖 products + GT rubrics，不需要加载 prediction 的超长 response（避免 OOM）
    gt_index = read_gt_index_light(gt_file)
    pred_index = read_pred_index_products_only(pred_file)

    source_file = os.path.basename(pred_file)
    infer_items, total_cnt = build_infer_data(gt_index, pred_index, source_file, lang=lang, judge_model=judge_model)

    if total_cnt <= 0:
        print("[judge_sop] 没有可评估的数据，退出。")
        return

    print(f"[judge_sop] 共构造 {total_cnt} 条 (uuid, product, scene) 评估请求")
    
    # 根据 judge_model 选择不同的 tools
    is_gemini = "gemini-2.5-pro" in judge_model.lower()
    tools_to_use = [{"type": "google_search"}] if is_gemini else TOOL_DEFINITIONS

    raw_file = output_file + ".judge_raw.jsonl"
    # 分块推理：避免一次性把 6w+ prompt/task 堆进内存
    chunk_size = int(os.environ.get("SHOPPINGCOMPJUDGE_CHUNK_SIZE", "1000") or "1000")
    chunk_size = max(50, chunk_size)
    chunks = int(math.ceil(total_cnt / chunk_size))
    buf: List[Dict[str, Any]] = []
    seen = 0
    chunk_idx = 0
    for item in infer_items:
        buf.append(item)
        if len(buf) >= chunk_size:
            chunk_idx += 1
            print(f"[judge_sop] chunk {chunk_idx}/{chunks}: items={len(buf)}")
            batch_infer(
                buf,
                model_name=judge_model,
                tools=tools_to_use,
                thinking_mode=True,
                max_workers=max_workers,
                max_retries=max_retries,
                output_file=raw_file,
                resume=resume,
                resume_key_fields=["uuid", "model_name", "run_id", "product_name", "scene_uuid"],
                resume_mode=resume_mode,
                checkpoint_batch_size=checkpoint_batch_size,
            )
            seen += len(buf)
            buf = []
    if buf:
        chunk_idx += 1
        print(f"[judge_sop] chunk {chunk_idx}/{chunks}: items={len(buf)}")
        batch_infer(
            buf,
            model_name=judge_model,
            tools=tools_to_use,
            thinking_mode=True,
            max_workers=max_workers,
            max_retries=max_retries,
            output_file=raw_file,
            resume=resume,
            resume_key_fields=["uuid", "model_name", "run_id", "product_name", "scene_uuid"],
            resume_mode=resume_mode,
            checkpoint_batch_size=checkpoint_batch_size,
        )
        seen += len(buf)
    print(f"[judge_sop] chunks done: {seen}/{total_cnt}")

    try:
        latest = read_jsonl_latest_by_key(
            raw_file,
            key_fields=["uuid", "model_name", "run_id", "product_name", "scene_uuid"],
            prefer_success=True,
        )
        results_for_agg = list(latest.values())
    except Exception:
        results_for_agg = []

    per_uuid_model_products: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in results_for_agg:
        uuid = r.get("uuid")
        model_name = r.get("model_name") or ""
        run_id = int(r.get("run_id") or 0)
        p_name = r.get("product_name")
        scene_uuid = r.get("scene_uuid")
        if not uuid or not p_name or not scene_uuid:
            continue
        scene = r.get("scene") or ""
        rubric = r.get("rubric") or ""

        judge_raw = str(r.get("judge_response") or "")
        parsed = parse_judge_answer(judge_raw)
        judge_tag = parsed.get("tag") or ""
        judge_evidence = parsed.get("evidence") or ""
        # Keep legacy field name for downstream metrics.
        judge_answer = judge_tag

        per_uuid_model_products[(uuid, f"{model_name}__run{run_id}")][p_name].append(
            {
                "scene_uuid": scene_uuid,
                "scene": scene,
                "rubric": rubric,
                "judge_answer": judge_answer,
                "judge_tag": judge_tag,
                "judge_evidence": judge_evidence,
                "judge_raw": judge_raw,
            }
        )

    rows: List[Dict[str, Any]] = []
    for (uuid, model_key), prod_dict in per_uuid_model_products.items():
        question = get_question(uuid, gt_index, pred_index)
        scene_list = (gt_index.get(uuid, {}) or {}).get("scene_list", []) or []

        eval_product_list: List[Dict[str, Any]] = []
        for p_name, evals in prod_dict.items():
            eval_product_list.append({p_name: evals})

        # model_key = "{model_name}__run{run_id}"
        if "__run" in model_key:
            model_name, run_id_str = model_key.split("__run", 1)
            run_id = int(run_id_str or 0)
        else:
            model_name, run_id = model_key, 0

        rows.append(
            {
                "uuid": uuid,
                "question": question,
                "scene_list": scene_list,
                "eval_product_list": eval_product_list,
                "source_file": source_file,
                "model_name": model_name,
                "run_id": run_id,
            }
        )

    write_jsonl(rows, output_file)

    print(
        f"[judge_sop] 已生成 SoP 输入文件: {output_file} （{len(per_uuid_model_products)} 行，按 uuid+model+run 粒度）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SoP eval inputs with LLM-as-a-Judge")
    parser.add_argument("--gt_file", "-g", required=True, help="ground-truth jsonl 路径")
    parser.add_argument("--pred_file", "-p", required=True, help="step3 模型结果 jsonl 路径")
    parser.add_argument("--output_file", "-o", required=True, help="输出 JSONL 路径")
    parser.add_argument("--judge_model", "-m", required=True, help="用于 judge 的 LLM 模型名（需在 api_config.yaml 中配置）")
    parser.add_argument("--max_workers", type=int, default=64, help="并发线程数")

    args = parser.parse_args()
    generate_sop_inputs(
        gt_file=args.gt_file,
        pred_file=args.pred_file,
        output_file=args.output_file,
        judge_model=args.judge_model,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()


