#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
工具调用分发器（可选）。

当前 ShoppingCompJudge 的主流程并不依赖工具调用，但 `api_client.py` 为了兼容某些
带 tool-calls 的模型，会尝试 import `tool_utils`。

本仓库版本提供“可执行”的工具分发实现，覆盖 `TOOL_DEFINITIONS` 中定义的函数工具：
- search_api(query=[...])
- link_reader_tool(url=...)
- link_summary_tool(question=..., links=[...])

注意：
- 工具是否真正被调用，取决于 judge 模型是否产生 tool_calls；
- 该分发器只负责执行模型请求的工具调用并返回字符串结果给模型；
- 为避免卡死/超长输出，默认会做超时与长度截断。
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests  # type: ignore


_TOOL_LOG_LOCK = threading.Lock()


def _get_env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _jina_ai_wrap(url: str) -> str:
    # Example: https://r.jina.ai/https://example.com/path
    return "https://r.jina.ai/" + url


def _tool_log_path() -> str:
    """
    工具调用日志路径（JSONL，一行一个 tool-call）。
    - 通过环境变量 `SHOPPINGCOMPJUDGE_TOOL_LOG_PATH` 指定
    - 为空则不写
    """
    return (os.environ.get("SHOPPINGCOMPJUDGE_TOOL_LOG_PATH") or "").strip()


def _append_tool_log(row: Dict[str, Any]) -> None:
    path = _tool_log_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        line = json.dumps(row, ensure_ascii=False)
        with _TOOL_LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # 日志失败不影响主流程
        return


def _non_empty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, list):
        out: List[str] = []
        for it in x:
            if isinstance(it, str) and it.strip():
                out.append(it.strip())
        return out
    return []


def _truncate_text(text: str, limit: int) -> str:
    if not isinstance(text, str):
        return ""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[TRUNCATED]"


def _extract_urls(text: str, *, max_n: int = 10) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    # stop at whitespace / quote / bracket
    urls = re.findall(r"https?://[^\s\"\'\]\)<>]+", text)
    out: List[str] = []
    for u in urls:
        u = (u or "").strip().rstrip(".,;")
        if u and u not in out:
            out.append(u)
        if len(out) >= max_n:
            break
    return out


def _extract_quoted_phrases(text: str, *, max_n: int = 10) -> List[str]:
    """
    从 raw arguments 中抽取尽可能像 query 的片段。
    常见坏格式：JSON 解析失败/有未转义引号/混入未加引号的 token。
    我们优先抽取双引号/单引号包裹的片段作为 queries。
    """
    if not isinstance(text, str) or not text:
        return []
    out: List[str] = []
    for pat in (r"\"([^\"]{2,256})\"", r"'([^']{2,256})'"):
        for m in re.findall(pat, text):
            s = (m or "").strip()
            if s and s not in out:
                out.append(s)
            if len(out) >= max_n:
                return out
    return out


def _salvage_args_from_raw(tool_name: str, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    尝试从 args_dict['raw_arguments'] 中补齐关键字段，避免模型参数格式不规范导致工具空转。
    """
    raw = args_dict.get("raw_arguments")
    if not isinstance(raw, str) or not raw.strip():
        return args_dict
    raw_s = raw.strip()
    name = (tool_name or "").strip()

    # search_api / google_search: try to recover query list
    if name in ("search_api", "google_search"):
        qv = args_dict.get("query") or args_dict.get("q")
        if not _as_str_list(qv):
            qs = _extract_quoted_phrases(raw_s, max_n=8)
            if qs:
                args_dict["query"] = qs
            else:
                # fallback: use whole raw as a single query (bounded)
                args_dict["query"] = [_truncate_text(raw_s, 200)]
        return args_dict

    # link_reader_tool: recover url
    if name == "link_reader_tool":
        if not _non_empty_str(args_dict.get("url")):
            urls = _extract_urls(raw_s, max_n=1)
            if urls:
                args_dict["url"] = urls[0]
        return args_dict

    # link_summary_tool: recover links + question
    if name == "link_summary_tool":
        if not _as_str_list(args_dict.get("links")):
            urls = _extract_urls(raw_s, max_n=5)
            if urls:
                args_dict["links"] = urls
        if not _non_empty_str(args_dict.get("question")):
            qs = _extract_quoted_phrases(raw_s, max_n=1)
            if qs:
                args_dict["question"] = qs[0]
        return args_dict

    return args_dict


def search_api(
    query: Any,
    *,
    num: int = 10,
    start: int = 0,
    engine: str = "google",
    location: str = "cn",
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """
    搜索工具（与 TOOL_DEFINITIONS 对齐）。
    返回结构尽量保持稳定：{"succeed": bool, "data": [...], "error_message": "..."}。
    """
    query_list = _as_str_list(query)
    if not query_list:
        return {"succeed": False, "data": [], "error_message": "search_api: empty query"}

    username = (os.environ.get("SHOPPINGCOMPJUDGE_SEARCH_USERNAME") or "").strip()
    # 兼容历史：不配置时也尝试请求（某些环境 username 可能不是必需）
    search_url = os.environ.get("SHOPPINGCOMPJUDGE_SEARCH_URL", "https://bandai-relay.bytedance.net/mcp_relay/search").strip()

    # prune results to keep tool payload bounded (avoid tool loops / token limit)
    keep_top_n = _get_env_int("SHOPPINGCOMPJUDGE_SEARCH_KEEP_TOP_N", 5)

    def _prune_obj(obj: Any) -> Any:
        # We don't assume a fixed schema from the relay. Best-effort prune.
        if isinstance(obj, list):
            out2 = []
            for it in obj[:keep_top_n]:
                if isinstance(it, dict):
                    out2.append(
                        {
                            "title": it.get("title") or it.get("name"),
                            "url": it.get("url") or it.get("link"),
                            "snippet": it.get("snippet") or it.get("description"),
                            "source": it.get("source"),
                        }
                    )
                else:
                    out2.append(it)
            return out2
        if isinstance(obj, dict):
            # common patterns: {"data":[...]} or {"results":[...]}
            for k in ("data", "results", "items"):
                v = obj.get(k)
                if isinstance(v, list):
                    pruned = dict(obj)
                    pruned[k] = _prune_obj(v)
                    # drop potentially huge blobs
                    pruned.pop("raw", None)
                    pruned.pop("html", None)
                    return pruned
        return obj

    all_results: List[Dict[str, Any]] = []
    for q in query_list:
        params: Dict[str, Any] = {
            "query": q,
            "num": int(num),
            "start": int(start),
            "engine": engine,
            "location": location,
        }
        if username:
            params["username"] = username
        try:
            resp = requests.get(search_url, params=params, timeout=timeout_s)
            if resp.status_code == 200:
                all_results.append({"query": q, "results": _prune_obj(resp.json())})
            else:
                all_results.append({"query": q, "error": f"HTTP {resp.status_code}", "text": _truncate_text(resp.text or "", 2000)})
        except Exception as e:
            all_results.append({"query": q, "error": f"{type(e).__name__}: {e}"})

        # 多 query 时轻微打散，避免被限流
        if len(query_list) > 1:
            time.sleep(0.2 + random.random() * 0.3)

    return {"succeed": True, "data": all_results}


def link_reader_tool(url: Any, *, timeout_s: float = 30.0, max_chars: int = 12000) -> Dict[str, Any]:
    """
    读取网页内容工具（与 TOOL_DEFINITIONS 对齐）。
    优先尝试内部 bandai_mcp_host；不可用时回退到 requests.get。
    """
    if not _non_empty_str(url):
        return {"succeed": False, "data": "", "error_message": "link_reader_tool: empty url"}
    url_s = str(url).strip()

    # allow overriding max chars via env (keeps tool payload small)
    max_chars = _get_env_int("SHOPPINGCOMPJUDGE_TOOL_RESULT_MAX_CHARS", int(max_chars))

    try:
        from bytedance.bandai_mcp_host import read_link  # type: ignore

        link_result = read_link(url_s)
        if getattr(link_result, "status", None) and link_result.status.is_succeeded():
            return {"succeed": True, "data": _truncate_text(str(link_result.result or ""), max_chars)}
        # If internal reader fails (common for sites like reddit), fall back to HTTP methods.
        internal_err = f"read_link failed: {getattr(link_result, 'status', None)}"
    except Exception:
        internal_err = ""

    # fallback: requests direct
    try:
        resp = requests.get(
            url_s,
            timeout=timeout_s,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"},
        )
        if resp.status_code == 200 and (resp.text or ""):
            return {"succeed": True, "data": _truncate_text(resp.text or "", max_chars)}
    except Exception:
        pass

    # fallback: jina.ai wrapper (often bypasses bot blocks / JS-heavy pages)
    try:
        resp = requests.get(
            _jina_ai_wrap(url_s),
            timeout=max(timeout_s, 25.0),
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"},
        )
        if resp.status_code == 200 and (resp.text or ""):
            return {"succeed": True, "data": _truncate_text(resp.text or "", max_chars)}
        return {
            "succeed": False,
            "data": "",
            "error_message": f"{internal_err}; fallback HTTP status={resp.status_code}",
        }
    except Exception as e:
        msg = f"{internal_err}; {type(e).__name__}: {e}".strip("; ")
        if not msg:
            msg = f"{type(e).__name__}: {e}"
        # Hint the model to proceed without tools to avoid loops
        return {
            "succeed": False,
            "data": "",
            "error_message": msg + " (please proceed without this tool if needed)",
        }


def link_summary_tool(
    question: Any,
    links: Any,
    *,
    timeout_s: float = 30.0,
    max_chars_per_link: int = 6000,
    max_links: int = 3,
) -> Dict[str, Any]:
    """
    链接摘要工具（与 TOOL_DEFINITIONS 对齐）。
    优先使用内部 bandai_mcp_host.link_summary；不可用时用 link_reader_tool 简易拼接。
    """
    q = str(question).strip() if question is not None else ""
    link_list = _as_str_list(links)
    if not q:
        q = "Summarize key information relevant to the task."
    if not link_list:
        return {"succeed": False, "data": "", "error_message": "link_summary_tool: empty links"}

    try:
        from bytedance.bandai_mcp_host import link_summary  # type: ignore

        link_result = link_summary(question=q, url=link_list)
        if getattr(link_result, "status", None) and link_result.status.is_succeeded():
            return {"succeed": True, "data": str(link_result.result or "")}
        return {
            "succeed": False,
            "data": "",
            "error_message": f"link_summary failed: {getattr(link_result, 'status', None)}",
        }
    except Exception:
        # fallback: read a few pages and concat
        chunks: List[str] = []
        for u in link_list[: max_links]:
            r = link_reader_tool(u, timeout_s=timeout_s, max_chars=max_chars_per_link)
            if r.get("succeed"):
                chunks.append(f"Source: {u}\n{r.get('data','')}")
            else:
                chunks.append(f"Source: {u}\n[READ FAILED] {r.get('error_message','')}")
        return {"succeed": True, "data": f"Question: {q}\n\n" + "\n\n".join(chunks)}


def tool_dispatcher(tool_call: Dict[str, Any]) -> str:
    """
    执行一次工具调用，返回字符串（用于 OpenAI tool message content）。
    输入约定：
      {\"name\": \"search_api\", \"arguments\": \"{...json...}\"} 或 arguments 已是 dict
    """
    name = str(tool_call.get("name") or "").strip()
    args = tool_call.get("arguments")

    # normalize args to dict
    args_dict: Dict[str, Any] = {}
    if isinstance(args, dict):
        args_dict = args
    elif isinstance(args, str) and args.strip():
        try:
            args_dict = json.loads(args)
        except Exception:
            args_dict = {"raw_arguments": args}
    # salvage when json is invalid or required fields are missing
    args_dict = _salvage_args_from_raw(name, args_dict)

    start_ts = time.time()
    try:
        if name == "search_api":
            out = search_api(args_dict.get("query"))
        elif name == "link_reader_tool":
            out = link_reader_tool(args_dict.get("url"))
        elif name == "link_summary_tool":
            out = link_summary_tool(args_dict.get("question"), args_dict.get("links"))
        elif name == "google_search":
            # 兼容极少数模型：把 google_search 当作 search_api
            out = search_api(args_dict.get("query") or args_dict.get("q"))
        else:
            out = {"succeed": False, "error_message": f"unknown tool: {name}", "data": None}

        elapsed_ms = int((time.time() - start_ts) * 1000)
        # 记录独立 tool log（不要写入超大内容）
        _append_tool_log(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "pid": os.getpid(),
                "tool": name,
                "arguments": args_dict,
                "succeed": bool(isinstance(out, dict) and out.get("succeed") is True),
                "elapsed_ms": elapsed_ms,
                "error_message": (out.get("error_message") if isinstance(out, dict) else None),
            }
        )

        # 统一序列化为 string 返回给模型；同时确保 payload 有界（尤其是 data 为 str 时）
        try:
            max_chars2 = _get_env_int("SHOPPINGCOMPJUDGE_TOOL_RESULT_MAX_CHARS", 12000)
            if isinstance(out, dict) and out.get("succeed") and isinstance(out.get("data"), str):
                out["data"] = _truncate_text(out.get("data") or "", max_chars2)
        except Exception:
            pass
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        elapsed_ms = int((time.time() - start_ts) * 1000)
        _append_tool_log(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "pid": os.getpid(),
                "tool": name,
                "arguments": args_dict,
                "succeed": False,
                "elapsed_ms": elapsed_ms,
                "error_message": f"{type(e).__name__}: {e}",
            }
        )
        return json.dumps({"succeed": False, "error_message": f"{type(e).__name__}: {e}", "tool": name, "arguments": args_dict}, ensure_ascii=False)


def tool_dispatcher_with_retry(tool_call: Dict[str, Any], max_retries: int = 3) -> str:
    """
    分发一次工具调用并带重试。

    约定输入形如：
    {
      "name": "google_search",
      "arguments": "{...json...}"  # 也可能已经是 dict
    }
    """
    for attempt in range(max_retries + 1):
        if attempt > 0:
            time.sleep(min(0.5 * (2**attempt), 6.0))
        result = tool_dispatcher(tool_call)
        try:
            obj = json.loads(result)
            if isinstance(obj, dict) and obj.get("succeed") is True:
                return result
        except Exception:
            # 非法 JSON 也直接返回，避免循环卡死
            return result

    # 达到最大重试次数，返回最后一次结果
    return result


