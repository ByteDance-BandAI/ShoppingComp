#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

"""
ShoppingCompJudge 基础配置。

注意：这个包设计为可以单独拎出来作为一个 repo 使用，
因此这里不要依赖上级工程的 config。
"""

from __future__ import annotations

import os

# 默认的 API 配置文件路径（相对当前包目录）
_LOCAL_API_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "api_config.yaml")
_LOCAL_API_CONFIG_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "api_config.example.yaml")
_REPO_ROOT_API_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api_config.yaml"))

# 允许通过环境变量覆盖（便于在大仓里复用统一配置）
_ENV_API_CONFIG_PATH = os.environ.get("SHOPPINGCOMPJUDGE_API_CONFIG", "").strip()

# 最终使用的 API 配置路径：
# 1) 若设置了环境变量，则优先使用；
# 2) 否则优先使用包目录下的 api_config.yaml；
# 3) 若包目录不存在（或你在仓库根目录维护了 api_config.yaml/软链），则回退到仓库根目录的 api_config.yaml。
if _ENV_API_CONFIG_PATH:
    API_CONFIG_PATH = _ENV_API_CONFIG_PATH
else:
    API_CONFIG_PATH = _LOCAL_API_CONFIG_PATH if os.path.exists(_LOCAL_API_CONFIG_PATH) else _REPO_ROOT_API_CONFIG_PATH

# 仅用于报错提示：仓库自带的示例配置路径
API_CONFIG_EXAMPLE_PATH = _LOCAL_API_CONFIG_EXAMPLE_PATH

# 一些保守的默认推理参数
API_MAX_TOKENS = 1024
API_TEMPERATURE = 0.0



