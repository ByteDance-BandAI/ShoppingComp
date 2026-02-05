#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
工具定义模块（OpenAI function calling 格式）。

用于非 gemini-2.5-pro 的 judge model，提供完整的工具定义。
gemini-2.5-pro 使用简化的 {"type": "google_search"} 格式。
"""

# 工具定义列表（OpenAI function calling格式）
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_api",
            "description": "Search the web for information. Uses a search engine to retrieve relevant content and supports running multiple queries at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of search queries. Each query will be executed as a separate search. Example: ['Python tutorial', 'machine learning introduction']"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "link_reader_tool",
            "description": "Read the content of a web page. Given a URL, returns the page content as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page to read. Must be a full URL (including http:// or https://)."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "link_summary_tool",
            "description": "Extract key information from web pages based on a question. Use this when search result snippets are insufficient. This tool reads the given pages and extracts information relevant to the question, and is recommended for obtaining detailed evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific question to answer; describes what information you want to extract from the links."
                    },
                    "links": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of web page URLs, typically from search_api results."
                    }
                },
                "required": ["question", "links"]
            }
        }
    }
]
