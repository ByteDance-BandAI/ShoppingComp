#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
简单的 prompt 加载与渲染工具。

从本目录下的 prompts.yaml 中读取模板：
- 一级 key: metric 名称（如 answer_match, sop）
- 二级 key: 语言代码（zh/en）
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml


_PROMPTS_CACHE: Dict[str, Any] | None = None


def _load_all_prompts() -> Dict[str, Any]:
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE

    path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
    with open(path, "r", encoding="utf-8") as f:
        _PROMPTS_CACHE = yaml.safe_load(f) or {}
    return _PROMPTS_CACHE


def get_default_language() -> str:
    """
    获取默认 prompt 语言。

    优先级：
    1) 环境变量 SHOPPINGCOMPJUDGE_PROMPT_LANG（zh/en）
    2) prompts.yaml: defaults.language（zh/en）
    3) 回退：en
    """
    env_lang = (os.environ.get("SHOPPINGCOMPJUDGE_PROMPT_LANG") or "").strip().lower()
    if env_lang in ("zh", "en"):
        return env_lang

    data = _load_all_prompts()
    defaults = data.get("defaults") or {}
    if isinstance(defaults, dict):
        yaml_lang = str(defaults.get("language") or "").strip().lower()
        if yaml_lang in ("zh", "en"):
            return yaml_lang

    return "en"


def get_prompt_template(metric: str, lang: str | None = None) -> str:
    data = _load_all_prompts()
    metric_cfg = data.get(metric)
    if not metric_cfg:
        raise KeyError(f"prompts.yaml 中未找到 metric '{metric}'")
    if not lang:
        lang = get_default_language()
    if lang not in metric_cfg:
        raise KeyError(f"prompts.yaml 中 metric '{metric}' 未配置语言 '{lang}'")
    return metric_cfg[lang]


def format_prompt(metric: str, lang: str | None = None, **kwargs: Any) -> str:
    """
    按 metric + 语言加载模板，并用 kwargs.format(...) 渲染。
    
    特殊处理：如果 kwargs 中包含 include_tool_guide=True，且 metric 为 answer_match，
    则在 prompt 中添加工具使用指南。
    """
    tmpl = get_prompt_template(metric, lang=lang)
    
    # 特殊处理：answer_match 且需要工具指南时，在 prompt 中添加工具使用说明
    if metric == "answer_match" and kwargs.get("include_tool_guide", False):
        tool_guide = _get_tool_usage_guide(lang or get_default_language())
        # 在 "If necessary, call..." 或 "必要时调用..." 之后插入工具指南
        if lang == "zh":
            # 中文版本：在 "**必要时调用 google_search..." 之后插入
            if "**必要时调用" in tmpl or "**If necessary" in tmpl:
                # 找到工具调用提示的位置，在其后插入完整工具指南
                insert_pos = tmpl.find("**必要时调用")
                if insert_pos == -1:
                    insert_pos = tmpl.find("**If necessary")
                if insert_pos != -1:
                    # 找到该行的结束位置
                    line_end = tmpl.find("\n", insert_pos)
                    if line_end != -1:
                        tmpl = tmpl[:line_end+1] + "\n" + tool_guide + "\n" + tmpl[line_end+1:]
                    else:
                        tmpl = tmpl + "\n\n" + tool_guide
                else:
                    # 如果找不到，在开头插入
                    tmpl = tool_guide + "\n\n" + tmpl
            else:
                tmpl = tool_guide + "\n\n" + tmpl
        else:
            # 英文版本：在 "**If necessary, call..." 之后插入
            if "**If necessary" in tmpl or "**必要时调用" in tmpl:
                insert_pos = tmpl.find("**If necessary")
                if insert_pos == -1:
                    insert_pos = tmpl.find("**必要时调用")
                if insert_pos != -1:
                    line_end = tmpl.find("\n", insert_pos)
                    if line_end != -1:
                        tmpl = tmpl[:line_end+1] + "\n" + tool_guide + "\n" + tmpl[line_end+1:]
                    else:
                        tmpl = tmpl + "\n\n" + tool_guide
                else:
                    tmpl = tool_guide + "\n\n" + tmpl
            else:
                tmpl = tool_guide + "\n\n" + tmpl
    
    # 移除 include_tool_guide 参数，避免 format 时出错
    kwargs_clean = {k: v for k, v in kwargs.items() if k != "include_tool_guide"}
    return tmpl.format(**kwargs_clean)


def _get_tool_usage_guide(lang: str) -> str:
    """获取工具使用指南（从 prompts.yaml 的 shoppingcomp_predict 中提取）。"""
    data = _load_all_prompts()
    shoppingcomp = data.get("shoppingcomp_predict", {})
    if lang in shoppingcomp:
        prompt_text = shoppingcomp[lang]
        # 提取工具使用指南部分（从 "## Tool usage guide" 或 "## 工具使用指引" 到 "## Output format" 或 "## 输出格式" 之前）
        if lang == "zh":
            start_marker = "## 工具使用指引"
            end_marker = "## 输出格式"
        else:
            start_marker = "## Tool usage guide"
            end_marker = "## Output format"
        
        start_idx = prompt_text.find(start_marker)
        end_idx = prompt_text.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            return prompt_text[start_idx:end_idx].strip()
        elif start_idx != -1:
            # 如果找不到结束标记，取到文件末尾
            return prompt_text[start_idx:].strip()
    
    # 兜底：返回硬编码的工具指南
    if lang == "zh":
        return """## 工具使用指引
你可以使用以下工具来帮助搜索和分析商品信息：
### 1. 搜索工具 (search_api)
- **功能**: 搜索网络上的商品信息、评测、价格等
- **使用方法**:
  - `query`: 搜索关键词，如"iPhone 15 Pro 手机"、"华为Mate60 续航"
  - `return_n`: 返回结果数量，建议设置为10

### 2. 链接摘要工具 (link_summary_tool)
- **功能**: 对多个网页链接进行智能摘要，提取关键信息
- **使用方法**:
  - `links`: 网页URL列表，通常来自搜索工具的结果
  - `question`: 希望从 `links` 中提取哪些关键信息

### 3. 读取链接工具 (link_reader_tool)
- **功能**: 获取某个网页链接的原文内容，markdown格式
- **使用方法**:
  - `url`: 需要读取的网页URL，通常来自搜索工具的结果"""
    else:
        return """## Tool usage guide

You can use the following tools to help search and analyze product information:

### 1. Search tool (search_api)
- **Function**: search product information, reviews, prices, etc. on the web.
- **Usage**:
  - `query`: search keywords, such as "iPhone 15 Pro smartphone",
    "Huawei Mate60 battery life".
  - `return_n`: number of results to return, recommended 10.

### 2. Link summary tool (link_summary_tool)
- **Function**: summarize multiple web pages and extract key information.
- **Usage**:
  - `links`: list of URLs, usually from search results.
  - `question`: what key information you want to extract from `links`.

### 3. Link reading tool (link_reader_tool)
- **Function**: fetch the original content of a web page in Markdown format.
- **Usage**:
  - `url`: the URL of the web page, usually from search results."""



