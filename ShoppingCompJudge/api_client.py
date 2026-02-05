#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
API 客户端（完整功能版），与 workflow_rebuild/product_pipeline/common/api_client.py 对齐，
支持工具调用、thinking / reasoning、多轮对话。
"""
from __future__ import annotations

import json
import random
import time
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

import os

from .config import API_CONFIG_EXAMPLE_PATH, API_CONFIG_PATH, API_MAX_TOKENS, API_TEMPERATURE


def _tool_calls_to_jsonable(tool_calls: Any) -> Any:
    """
    将 openai SDK 的 tool_call 对象转换为 JSON 可序列化的结构。

    某些模型/SDK 会返回 ChatCompletionMessageFunctionToolCall 等对象，
    直接 json.dumps 会报：Object of type ... is not JSON serializable。
    """
    if tool_calls is None:
        return None
    # 已经是 list[dict] 的情况
    if isinstance(tool_calls, list) and all(isinstance(x, dict) for x in tool_calls):
        return tool_calls
    out: List[Dict[str, Any]] = []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            try:
                fn = getattr(tc, "function", None)
                out.append(
                    {
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", None),
                        "function": {
                            "name": getattr(fn, "name", None) if fn is not None else None,
                            "arguments": getattr(fn, "arguments", None) if fn is not None else None,
                        },
                    }
                )
            except Exception:
                # 兜底：转成字符串，至少不阻塞日志
                out.append({"raw": str(tc)})
        return out
    # 非 list：兜底转字符串
    return str(tool_calls)


_API_ERROR_LOG_LOCK = threading.Lock()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _get_api_error_log_path() -> str:
    return os.environ.get("SHOPPINGCOMPJUDGE_API_ERROR_LOG_PATH", "").strip()


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        _ensure_parent_dir(path)
        line = json.dumps(obj, ensure_ascii=False)
        with _API_ERROR_LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # logging 本身不能影响主流程
        return


def _extract_request_id(err: Exception) -> Optional[str]:
    """
    尽量从 SDK 异常中提取 request_id（不同 openai 版本/代理实现字段可能不同）。
    """
    try:
        rid = getattr(err, "request_id", None)
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    except Exception:
        pass
    try:
        resp = getattr(err, "response", None)
        # 某些错误会携带 response.headers
        headers = getattr(resp, "headers", None)
        if headers:
            for k in ("x-request-id", "request-id", "x-ms-request-id"):
                v = headers.get(k) if hasattr(headers, "get") else None
                if isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        pass
    return None


def _messages_preview(messages: List[Dict[str, Any]], max_chars: int = 400) -> Dict[str, Any]:
    """
    仅用于错误日志：避免把完整 prompt/历史写爆日志，也避免泄露过多内容。
    """
    try:
        last_user = ""
        for m in reversed(messages or []):
            if (m or {}).get("role") == "user":
                last_user = str((m or {}).get("content") or "")
                break
        last_user = last_user.strip()
        if max_chars > 0 and len(last_user) > max_chars:
            prev = last_user[:max_chars] + f"...(truncated,len={len(last_user)})"
        else:
            prev = last_user
        return {"last_user_len": len(last_user), "last_user_preview": prev, "num_messages": len(messages or [])}
    except Exception:
        return {"last_user_len": None, "last_user_preview": "", "num_messages": None}


def load_api_config(model_name: str) -> Dict[str, Any]:
    """加载模型配置。"""
    try:
        if not API_CONFIG_PATH or not os.path.exists(API_CONFIG_PATH):
            example_hint = ""
            if API_CONFIG_EXAMPLE_PATH and os.path.exists(API_CONFIG_EXAMPLE_PATH):
                example_hint = f"\n- 你可以复制示例配置：cp {API_CONFIG_EXAMPLE_PATH} {os.path.join(os.path.dirname(API_CONFIG_EXAMPLE_PATH), 'api_config.yaml')}"
            raise FileNotFoundError(
                "未找到 LLM API 配置文件。\n"
                f"- 当前 API_CONFIG_PATH: {API_CONFIG_PATH}\n"
                "- 解决方式：\n"
                "  1) 在包目录下创建 `api_config.yaml`（该文件已在 .gitignore 中忽略，不会被提交）；或\n"
                "  2) 设置环境变量 `SHOPPINGCOMPJUDGE_API_CONFIG=/path/to/api_config.yaml`。\n"
                f"{example_hint}"
            )
        with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        defaults = config.get("defaults", {}) or {}
        models = config.get("models", {}) or {}

        if model_name not in models:
            print(f"[api_client] 警告: 模型 '{model_name}' 未找到，使用默认配置")
            return defaults

        model_configs = models[model_name]
        if isinstance(model_configs, dict):
            selected_config = model_configs
        elif isinstance(model_configs, list):
            selected_config = random.choice(model_configs)
        else:
            print(f"[api_client] 警告: 模型 '{model_name}' 配置格式错误，使用默认配置")
            return defaults

        merged = defaults.copy()
        merged.update(selected_config)
        return merged
    except Exception as e:
        print(f"[api_client] 加载配置失败: {e}")
        return {}


def _build_timeout_from_config(model_config: Dict[str, Any]) -> Any:
    """
    为 openai-python(httpx) 构建 timeout。

    为什么要做这个：
    - 线上偶发网络/服务端卡住时，若没有 timeout，单个线程会永远阻塞在请求上；
      batch_infer 使用 as_completed 等待所有 future，于是“任务都跑完了但进程不退出”的现象就会出现。
    """
    # 优先允许通过环境变量临时覆盖（便于线上排障时快速收敛“单请求卡死”问题）
    env_total = os.environ.get("SHOPPINGCOMPJUDGE_REQUEST_TIMEOUT_S", "").strip()
    if env_total:
        try:
            total = float(env_total)
        except Exception:
            total = None
    else:
        total = None

    # 兼容多种字段名：便于在 api_config.yaml 里按需配置
    if total is None:
        total = model_config.get("timeout_s", None)
    if total is None:
        total = model_config.get("request_timeout_s", 600)
    connect = model_config.get("connect_timeout_s", 30)
    read = model_config.get("read_timeout_s", total)
    write = model_config.get("write_timeout_s", total)
    pool = model_config.get("pool_timeout_s", 30)

    try:
        import httpx  # openai-python 依赖 httpx，这里按需 import

        return httpx.Timeout(timeout=float(total), connect=float(connect), read=float(read), write=float(write), pool=float(pool))
    except Exception:
        # 兜底：如果 httpx 不可用，返回一个数字，让上层尝试传入（不同 openai 版本可能接受 seconds）
        try:
            return float(total)
        except Exception:
            return None


def build_api_client(
    base_url: str,
    api_version: str,
    api_key: str,
    *,
    model_config: Optional[Dict[str, Any]] = None,
) -> AzureOpenAI | OpenAI:
    """构建 API 客户端（带超时配置，防止请求无期限挂起导致进程不退出）。"""
    model_config = model_config or {}
    timeout = _build_timeout_from_config(model_config)

    kwargs: Dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = timeout

    if "ark" in base_url:
        return OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return AzureOpenAI(azure_endpoint=base_url, api_version=api_version, api_key=api_key, **kwargs)


def is_rate_limit_error(error_message: str) -> bool:
    indicators = ["429", "rate limit", "too many requests", "quota exceeded", "throttled", "限流", "频率限制"]
    return any(indicator in error_message.lower() for indicator in indicators)


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    delay = min(base_delay * (2**attempt), max_delay)
    jitter = delay * 0.1 * random.random()
    return delay + jitter


def build_thinking_config(model_name: str, model_config: Dict[str, Any], thinking_mode: bool) -> Dict[str, Any]:
    """构建 thinking 配置。"""
    if not thinking_mode or not model_config.get("thinking_enabled", False):
        return {}

    budget_tokens = model_config.get("thinking_budget_tokens", 2000)
    lower = model_name.lower()

    if "claude" in lower:
        return {
            "extra_body": {
                "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
                "anthropic_beta": ["interleaved-thinking-2025-05-14"],
            }
        }
    if "gemini" in lower:
        return {"extra_body": {"thinking": {"include_thoughts": True, "budget_tokens": budget_tokens}}}
    if "gpt" in lower:
        return {
            "reasoning_effort": "medium",
            "extra_body": {"reasoning": {"effort": "medium"}},
        }
    if "ep-" in lower:
        return {"extra_body": {"thinking": {"type": "enabled"}}}
    return {}


def extract_thinking_info(completion: Any, model_name: str) -> str:
    """提取 thinking 信息。"""
    try:
        if not hasattr(completion, "choices") or not completion.choices:
            return ""
        msg = completion.choices[0].message
        for attr in ["reasoning_content", "thinking", "reasoning"]:
            if hasattr(msg, attr):
                val = getattr(msg, attr)
                if val:
                    return val
    except Exception:
        return ""
    return ""


def chat_completion(
    client: Any,
    model_name: str,
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    thinking_mode: bool = False,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    单次聊天 API 调用。
    返回:
        {
            "success": bool,
            "content": str,
            "thinking": str,
            "tool_calls": list,
            "error": str,
            "api_params": dict
        }
    """
    model_config = load_api_config(model_name)

    # 自动注入 google_search
    if tools is None and model_config.get("websearch_enabled", False):
        tools = [{"type": "google_search"}]

    request_model = model_config.get("model_name", model_name)
    thinking_cfg = build_thinking_config(model_name, model_config, thinking_mode)
    api_params = {
        "model": request_model,
        "messages": messages,
        "max_tokens": model_config.get("max_tokens", API_MAX_TOKENS),
        "temperature": model_config.get("temperature", API_TEMPERATURE),
        **thinking_cfg,
    }
    if tools and not model_config.get("is_ark_bot", False):
        api_params["tools"] = tools

    safe_api_params = {
        "model": request_model,
        "max_tokens": api_params.get("max_tokens"),
        "temperature": api_params.get("temperature"),
    }
    if "tools" in api_params:
        safe_api_params["tools"] = api_params["tools"]
    if thinking_cfg:
        safe_api_params.update(thinking_cfg)

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(**api_params)
            # 基本结构校验：某些 SDK/代理异常时会返回非预期结构
            if not hasattr(completion, "choices") or not completion.choices:
                raise RuntimeError("completion.choices 为空或缺失")
            message = completion.choices[0].message
            if message is None:
                raise RuntimeError("completion.choices[0].message 为空")
            thinking = extract_thinking_info(completion, model_name)
            return {
                "success": True,
                "content": message.content or "",
                "thinking": thinking,
                "tool_calls": message.tool_calls if hasattr(message, "tool_calls") else None,
                "error": None,
                "api_params": safe_api_params,
            }
        except Exception as e:
            err = str(e)
            req_id = _extract_request_id(e)
            if attempt < max_retries:
                if is_rate_limit_error(err):
                    delay = exponential_backoff(attempt, 10.0, 300.0)
                else:
                    delay = exponential_backoff(attempt, 2.0, 60.0)
                _append_jsonl(
                    _get_api_error_log_path(),
                    {
                        "ts": datetime.now().isoformat(),
                        "event": "api_retry",
                        "request_id": req_id,
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "delay_s": delay,
                        "is_rate_limit": is_rate_limit_error(err),
                        "error": err,
                        "api_params": safe_api_params,
                        **_messages_preview(messages),
                    },
                )
                time.sleep(delay)
            else:
                print(f"[api_client] API 调用失败（已达最大重试次数 {max_retries}）: {err}")
                _append_jsonl(
                    _get_api_error_log_path(),
                    {
                        "ts": datetime.now().isoformat(),
                        "event": "api_failed",
                        "request_id": req_id,
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "is_rate_limit": is_rate_limit_error(err),
                        "error": err,
                        "api_params": safe_api_params,
                        **_messages_preview(messages),
                    },
                )
                return {
                    "success": False,
                    "content": None,
                    "thinking": None,
                    "tool_calls": None,
                    "error": err,
                    "api_params": safe_api_params,
                }


def process_single_inference(task_data: Tuple) -> Dict[str, Any]:
    """多轮对话（含工具）。"""
    idx, model_name, prompt, base_data, tools, max_retries, thinking_mode, verbose = task_data

    config = load_api_config(model_name)
    client = build_api_client(
        config.get("azure_endpoint", config.get("base_url", "")),
        config.get("api_version", "2024-03-01-preview"),
        config.get("api_key", ""),
        model_config=config,
    )

    messages = [{"role": "user", "content": prompt}]
    conversation_history = []
    start_time = time.time()
    result: Dict[str, Any] = {}

    try:
        # 单条样本允许的“模型-工具”多轮次数；过大可能导致卡在某条样本很久（尤其工具不可用/频繁失败时）
        try:
            max_rounds = int(os.environ.get("SHOPPINGCOMPJUDGE_MAX_ROUNDS", "10").strip() or "10")
        except Exception:
            max_rounds = 10
        if max_rounds <= 0:
            max_rounds = 1
        for round_idx in range(max_rounds):
            if verbose:
                print(f"round_idx: {round_idx}")
            result = chat_completion(
                client, model_name, messages, tools, thinking_mode, max_retries
            )
            if not result["success"]:
                break

            conversation_history.append(
                {"role": "assistant", "content": result["content"], "thinking": result["thinking"]}
            )

            if result["tool_calls"]:
                from .tool_utils import tool_dispatcher_with_retry

                assistant_message = {
                    "role": "assistant",
                    "content": result["content"],
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in result["tool_calls"]
                    ],
                }
                messages.append(assistant_message)
                if verbose:
                    print(f"assistant_message: {assistant_message}")

                # 工具重试次数不要跟 API max_retries 绑死：API 可能设到 1000，但工具通常 2-5 次就够了，
                # 否则会在单条样本里对同一个 URL/tool 卡上几十分钟（看起来进程“停了”）。
                try:
                    tool_max_retries = int(os.environ.get("SHOPPINGCOMPJUDGE_TOOL_MAX_RETRIES", "3").strip() or "3")
                except Exception:
                    tool_max_retries = 3
                if tool_max_retries < 0:
                    tool_max_retries = 0
                if tool_max_retries > 10:
                    tool_max_retries = 10

                for tc in result["tool_calls"]:
                    tool_call_dict = {"name": tc.function.name, "arguments": tc.function.arguments}
                    tool_result = tool_dispatcher_with_retry(tool_call_dict, max_retries=tool_max_retries)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result})

                    if "gpt-5" not in model_name.lower():
                        conversation_history.append(
                            {"role": "tool", "name": tc.function.name, "content": tool_result}
                        )
            else:
                break
    finally:
        # 关闭底层 http client，避免连接/资源长期占用（并减少“看起来跑完了但进程还挂着”的尾巴）
        try:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass

    response_time = time.time() - start_time
    return {
        **base_data,
        "judge_success": result.get("success", False),
        "judge_response": result.get("content"),
        "judge_thinking": result.get("thinking"),
        "judge_error": result.get("error"),
        "judge_api_params": result.get("api_params"),
        "judge_conversation_history": conversation_history,
        "judge_response_time": response_time,
        "judge_tool_calls": _tool_calls_to_jsonable(result.get("tool_calls")),
        # 注意：不要覆盖 base_data 里的 model_name（它通常表示“被评估/来源模型”）
        # 这里单独记录用于 LLM-as-a-Judge 的模型名
        "judge_model_name": model_name,
        "judge_timestamp": datetime.now().isoformat(),
        "batch_index": idx,
    }


def batch_infer(
    infer_data: List[Dict[str, Any]],
    model_name: str,
    tools: Optional[List[Dict]] = None,
    max_retries: int = 3,
    max_workers: int = 64,
    delay_between_requests: float = 2.0,
    thinking_mode: bool = False,
    output_file: Optional[str] = None,
    verbose: bool = True,
    resume: bool = False,
    resume_key_fields: Optional[List[str]] = None,
    resume_mode: str = "success_only",
    checkpoint_batch_size: int = 20,
    checkpoint_prune: bool = True,
) -> List[Dict[str, Any]]:
    """批量推理，支持工具和 thinking。"""
    # ---- Debug / logging knobs (env) ----
    # 说明：
    # - 默认 judgers 调用 batch_infer 时 output_file=None（不落盘）。
    # - 若你希望把每条 judge 的返回（success/error/content/tool_calls/params...）都记录下来，
    #   可以设置环境变量 SHOPPINGCOMPJUDGE_JUDGE_RESULT_LOG_PATH=/path/to/judge_results.jsonl
    # - 若你希望把每条返回也直接打印到 stdout，设置 SHOPPINGCOMPJUDGE_PRINT_JUDGE_RESULTS=1
    env_result_log = os.environ.get("SHOPPINGCOMPJUDGE_JUDGE_RESULT_LOG_PATH", "").strip()
    if env_result_log and not output_file:
        output_file = env_result_log
    env_print_results = os.environ.get("SHOPPINGCOMPJUDGE_PRINT_JUDGE_RESULTS", "").strip().lower()
    print_results = env_print_results in ("1", "true", "yes", "y", "on")
    # 打印时截断，避免 stdout 被超长内容淹没（可通过 env 调大/设为0关闭截断）
    try:
        max_chars = int(os.environ.get("SHOPPINGCOMPJUDGE_PRINT_JUDGE_RESULTS_MAX_CHARS", "2000").strip() or "2000")
    except Exception:
        max_chars = 2000
    # 允许通过环境变量覆盖节流（默认保守为 2s；在本地/受控环境可调小以提速）
    env_delay = os.environ.get("SHOPPINGCOMPJUDGE_DELAY_BETWEEN_REQUESTS_S", "").strip()
    if env_delay:
        try:
            delay_between_requests = float(env_delay)
        except Exception:
            pass

    tasks: List[Tuple] = []
    completed_keys: set = set()

    def _make_key(row: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
        if not resume_key_fields:
            return None
        key: List[Any] = []
        for f in resume_key_fields:
            if f not in row:
                return None
            key.append(row.get(f))
        return tuple(key)

    # 断点续跑：从 output_file 中读取已完成 key（默认只跳过 judge_success=True 的）
    if resume and output_file and resume_key_fields and os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    k = _make_key(obj)
                    if k is None:
                        continue
                    if resume_mode == "all":
                        completed_keys.add(k)
                    else:
                        # success_only
                        if obj.get("judge_success") is True:
                            completed_keys.add(k)
        except Exception as e:
            _append_jsonl(
                _get_api_error_log_path(),
                {
                    "ts": datetime.now().isoformat(),
                    "event": "resume_read_failed",
                    "error": str(e),
                    "output_file": output_file,
                },
            )

    for i, item in enumerate(infer_data):
        # 重要：不要在 base_data 里重复拷贝 prompt（prompt 往往很长，且 infer_data 里已包含一份）。
        base_data = item.copy()
        base_data.pop("prompt", None)
        if resume and completed_keys and resume_key_fields:
            k = _make_key(base_data)
            if k is not None and k in completed_keys:
                continue
        tasks.append(
            (i, model_name, item["prompt"], base_data, tools, max_retries, thinking_mode, verbose)
        )

    # 重要：不要在内存里保留“未裁剪的完整结果”。
    # 在开启 tools/thinking 时，judge_conversation_history 可能非常大，
    # 若把 1w+ 条结果完整堆在 results 里，进程容易被 OOM killer 杀死。
    # 因此：results 里仅保留裁剪后的轻量副本（与 checkpoint_prune 保持一致）。
    results: List[Dict[str, Any]] = []
    checkpoint_buf: List[Dict[str, Any]] = []

    # ---- Progress heartbeat (more readable than tqdm in logs) ----
    # tqdm uses '\r' which often looks "stuck" in redirected logs. This heartbeat prints
    # stable lines like: [progress] 推理-xxx done=123/999 elapsed=...
    progress_every_n = 0
    progress_every_s = 0.0
    try:
        progress_every_n = int(os.environ.get("SHOPPINGCOMPJUDGE_PROGRESS_EVERY_N", "0").strip() or "0")
    except Exception:
        progress_every_n = 0
    try:
        progress_every_s = float(os.environ.get("SHOPPINGCOMPJUDGE_PROGRESS_EVERY_S", "30").strip() or "30")
    except Exception:
        progress_every_s = 30.0
    if progress_every_s <= 0:
        progress_every_s = 30.0
    done_cnt = 0
    start_ts = time.time()
    last_progress_ts = start_ts

    def _flush_checkpoint() -> None:
        nonlocal checkpoint_buf
        if not output_file or not checkpoint_buf:
            checkpoint_buf = []
            return
        try:
            _ensure_parent_dir(output_file)
            with open(output_file, "a", encoding="utf-8") as f:
                for rr in checkpoint_buf:
                    f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        except Exception as e:
            _append_jsonl(
                _get_api_error_log_path(),
                {
                    "ts": datetime.now().isoformat(),
                    "event": "checkpoint_write_failed",
                    "error": str(e),
                    "output_file": output_file,
                },
            )
        checkpoint_buf = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 重要：不要把完整 task tuple（包含长 prompt）塞进 future->task 映射，会显著放大内存占用。
        future_to_meta: Dict[Any, Dict[str, Any]] = {}
        for task in tasks:
            fut = executor.submit(process_single_inference, task)
            # task: (idx, model_name, prompt, base_data, ...)
            base_data = task[3] if len(task) > 3 and isinstance(task[3], dict) else {}
            future_to_meta[fut] = {"batch_index": task[0], "base_data": base_data}

        # NOTE: using wait(timeout=...) so we can print heartbeats even when 0 future completes.
        pending = set(future_to_meta.keys())
        with tqdm(total=len(tasks), desc=f"推理-{model_name}") as pbar:
            while pending:
                done_set, pending = wait(pending, timeout=progress_every_s, return_when=FIRST_COMPLETED)
                now = time.time()
                # heartbeat even when no completion
                if (progress_every_n <= 0) and (now - last_progress_ts >= progress_every_s) and (not done_set):
                    elapsed = now - start_ts
                    print(f"[progress] 推理-{model_name} done={done_cnt}/{len(tasks)} elapsed_s={elapsed:.1f}")
                    last_progress_ts = now

                for future in done_set:
                    try:
                        r = future.result()
                        done_cnt += 1
                        pbar.update(1)
                        now2 = time.time()
                        if (progress_every_n > 0 and done_cnt % progress_every_n == 0) or (now2 - last_progress_ts >= progress_every_s):
                            elapsed = now2 - start_ts
                            print(f"[progress] 推理-{model_name} done={done_cnt}/{len(tasks)} elapsed_s={elapsed:.1f}")
                            last_progress_ts = now2
                        mem_row = dict(r)
                        if checkpoint_prune:
                            mem_row.pop("prompt", None)
                            mem_row.pop("judge_conversation_history", None)
                        results.append(mem_row)
                        if output_file:
                            rr = dict(r)
                            # 额外兜底：确保 JSON 可序列化（避免 tool_calls/其他对象导致落盘失败）
                            if "judge_tool_calls" in rr:
                                rr["judge_tool_calls"] = _tool_calls_to_jsonable(rr.get("judge_tool_calls"))
                            # 默认裁剪超大字段：checkpoint 主要用于断点续跑/聚合，不需要完整历史
                            if checkpoint_prune:
                                rr.pop("prompt", None)
                                rr.pop("judge_conversation_history", None)
                            checkpoint_buf.append(rr)
                            if checkpoint_batch_size <= 1 or len(checkpoint_buf) >= checkpoint_batch_size:
                                _flush_checkpoint()
                        if print_results:
                            rr = dict(r)
                            if "judge_tool_calls" in rr:
                                rr["judge_tool_calls"] = _tool_calls_to_jsonable(rr.get("judge_tool_calls"))
                            if max_chars > 0:
                                for k in ("judge_response", "judge_thinking"):
                                    v = rr.get(k)
                                    if isinstance(v, str) and len(v) > max_chars:
                                        rr[k] = v[:max_chars] + f"...(truncated,len={len(v)})"
                            print(json.dumps(rr, ensure_ascii=False))
                        if delay_between_requests > 0:
                            time.sleep(delay_between_requests)
                    except Exception as e:
                        print(f"[api_client] 任务执行失败: {e}")
                        done_cnt += 1
                        pbar.update(1)
                        now2 = time.time()
                        if (progress_every_n > 0 and done_cnt % progress_every_n == 0) or (now2 - last_progress_ts >= progress_every_s):
                            elapsed = now2 - start_ts
                            print(f"[progress] 推理-{model_name} done={done_cnt}/{len(tasks)} elapsed_s={elapsed:.1f}")
                            last_progress_ts = now2
                        meta = future_to_meta.get(future) or {}
                        base_data = meta.get("base_data") if isinstance(meta.get("base_data"), dict) else {}
                        fail_row = {
                            **base_data,
                            "judge_success": False,
                            "judge_error": str(e),
                            "judge_model_name": model_name,
                            "batch_index": meta.get("batch_index", 0),
                            "judge_timestamp": datetime.now().isoformat(),
                        }
                        results.append(fail_row)
                        _append_jsonl(
                            _get_api_error_log_path(),
                            {
                                "ts": datetime.now().isoformat(),
                                "event": "task_exception",
                                "error": str(e),
                                "judge_model_name": model_name,
                                "batch_index": meta.get("batch_index", 0),
                                "resume_key_fields": resume_key_fields,
                                "base_fields": {k: base_data.get(k) for k in (resume_key_fields or [])},
                            },
                        )
                        if output_file:
                            rr = dict(fail_row)
                            if checkpoint_prune:
                                rr.pop("prompt", None)
                                rr.pop("judge_conversation_history", None)
                            checkpoint_buf.append(rr)
                            if checkpoint_batch_size <= 1 or len(checkpoint_buf) >= checkpoint_batch_size:
                                _flush_checkpoint()

    _flush_checkpoint()
    results.sort(key=lambda x: x.get("batch_index", 0))
    return results




