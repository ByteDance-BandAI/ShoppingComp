#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
I/O 与解析工具（judgers + metrics 共用）。

目标：
- 统一 jsonl 读写与错误处理
- 统一简单的 tag 解析（<answer>/<result>/<analysis>）
- 提供少量“预测文件结构”辅助函数（例如 model_results/products/question）
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """逐行读取 jsonl；坏行会被跳过并打印警告。"""
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[io_utils] {path} 第{line_num}行 JSON 解析失败: {e}")
                continue
            if isinstance(obj, dict):
                yield obj


def read_jsonl_index(path: str, key: str = "uuid") -> Dict[str, Dict[str, Any]]:
    """读取 jsonl 为 dict 索引，默认使用 uuid 作为 key。"""
    out: Dict[str, Dict[str, Any]] = {}
    for obj in iter_jsonl(path):
        k = obj.get(key)
        if k is None:
            continue
        ks = str(k).strip()
        if not ks:
            continue
        out[ks] = obj
    return out


def read_pred_index_products_only(path: str, key: str = "uuid") -> Dict[str, Dict[str, Any]]:
    """
    读取 prediction jsonl 为 index，但**只保留 products 相关字段**（显著降低内存）。

    适用场景：
    - answer_match / sop 只依赖 model_results[*].products，不需要超长 response / tool history。

    保留字段：
    - uuid, question（若存在）
    - model_results[*].model / products / success（若存在）
    """
    out: Dict[str, Dict[str, Any]] = {}
    for obj in iter_jsonl(path):
        k = obj.get(key)
        if k is None:
            continue
        ks = str(k).strip()
        if not ks:
            continue

        slim: Dict[str, Any] = {"uuid": ks}
        if "question" in obj:
            slim["question"] = obj.get("question") or ""

        mr = obj.get("model_results")
        if isinstance(mr, list):
            slim_mr: List[Dict[str, Any]] = []
            for item in mr:
                if not isinstance(item, dict):
                    continue
                # 仅保留轻量字段，避免把 response（可能很大）读入内存
                slim_item: Dict[str, Any] = {}
                if "model" in item:
                    slim_item["model"] = item.get("model")
                if "products" in item:
                    slim_item["products"] = item.get("products")
                if "success" in item:
                    slim_item["success"] = item.get("success")
                # 保底：没有任何字段则跳过
                if slim_item:
                    slim_mr.append(slim_item)
            slim["model_results"] = slim_mr

        out[ks] = slim
    return out


def read_gt_index_light(path: str, key: str = "uuid") -> Dict[str, Dict[str, Any]]:
    """
    读取 GT jsonl 为 index，但只保留 answer_match/sop 需要的轻量字段：
    - uuid, question, product_list, scene_list
    """
    out: Dict[str, Dict[str, Any]] = {}
    for obj in iter_jsonl(path):
        k = obj.get(key)
        if k is None:
            continue
        ks = str(k).strip()
        if not ks:
            continue
        slim: Dict[str, Any] = {"uuid": ks}
        if "question" in obj:
            slim["question"] = obj.get("question") or ""
        if "product_list" in obj:
            slim["product_list"] = obj.get("product_list") or []
        if "scene_list" in obj:
            slim["scene_list"] = obj.get("scene_list") or []
        out[ks] = slim
    return out


def read_jsonl_list(path: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_row_key(row: Dict[str, Any], key_fields: Sequence[str]) -> Optional[Tuple[Any, ...]]:
    key: List[Any] = []
    for f in key_fields:
        if f not in row:
            return None
        key.append(row.get(f))
    return tuple(key)


def read_jsonl_latest_by_key(
    path: str,
    *,
    key_fields: Sequence[str],
    prefer_success: bool = True,
    success_field: str = "judge_success",
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    """
    读取 jsonl，并按 key_fields 去重，返回 {key_tuple: row}。

    - 若同一 key 出现多次，默认“优先保留 judge_success=True 的结果”；否则保留最后出现的行。
    - 常用于断点续跑：同一请求可能先失败、重跑后成功，最终聚合应取成功版本。
    """
    out: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        k = _build_row_key(row, key_fields)
        if k is None:
            continue
        if not prefer_success:
            out[k] = row
            continue
        prev = out.get(k)
        if prev is None:
            out[k] = row
            continue
        prev_ok = bool(prev.get(success_field))
        cur_ok = bool(row.get(success_field))
        if cur_ok and not prev_ok:
            out[k] = row
        elif cur_ok == prev_ok:
            # 都成功或都失败：保留最后出现的
            out[k] = row
    return out


_TAG_RE_CACHE: Dict[str, re.Pattern[str]] = {}


def extract_tag(text: str, tag: str) -> Optional[str]:
    """提取形如 <tag>...</tag> 的内容；无则返回 None。"""
    if not text:
        return None
    pat = _TAG_RE_CACHE.get(tag)
    if pat is None:
        pat = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
        _TAG_RE_CACHE[tag] = pat
    m = pat.search(text)
    if not m:
        return None
    return m.group(1).strip()


def parse_result_01(text: str, *, default: int = 0) -> int:
    """解析 <result>0/1</result>，或包含 0/1 的内容。"""
    content = extract_tag(text or "", "result")
    if content is None:
        return default
    if "1" in content:
        return 1
    if "0" in content:
        return 0
    return default


def parse_answer_text(text: str) -> str:
    """优先解析 <answer>...</answer>，否则返回原文本 strip。"""
    ans = extract_tag(text or "", "answer")
    return ans if ans is not None else (text or "").strip()


def parse_analysis_text(text: str) -> str:
    """优先解析 <analysis>...</analysis>，否则返回原文本 strip。"""
    ana = extract_tag(text or "", "analysis")
    return ana if ana is not None else (text or "").strip()


# -----------------------------
# Prediction structure helpers
# -----------------------------


def extract_model_items(pred_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 pred_row 中提取有效的 model_results 条目（至少包含非空 products）。

    说明：该函数属于“数据结构约定”层面的辅助，不依赖具体某个 metric。
    """
    items: List[Dict[str, Any]] = []
    mr = pred_row.get("model_results")
    if not isinstance(mr, list):
        return items
    for item in mr:
        products = item.get("products")
        if isinstance(products, list) and products:
            items.append(item)
    return items


def model_products(model_item: Dict[str, Any]) -> tuple[str, List[str]]:
    """返回 (model_name, 非空产品列表)。"""
    model_name = str(model_item.get("model") or "")
    products = [str(p).strip() for p in model_item.get("products", []) if str(p).strip()]
    return model_name, products


def get_question(uuid: str, gt_index: Dict[str, Dict[str, Any]], pred_index: Dict[str, Dict[str, Any]]) -> str:
    """从 GT/pred 两边回退取 question。"""
    gt_row = gt_index.get(uuid, {})
    pred_row = pred_index.get(uuid, {})
    return gt_row.get("question") or pred_row.get("question") or ""


