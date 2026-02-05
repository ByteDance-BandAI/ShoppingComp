#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for resolving dataset files from Hugging Face.

Goal: allow CLI users to pass either:
- a local path (existing file), or
- a HF reference (e.g., filename in the HF dataset repo, or hf://...),
and transparently obtain a local file path for downstream code that expects files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


DEFAULT_HF_DATASET_REPO = os.environ.get("SHOPPINGCOMP_HF_DATASET_REPO", "huaixiao/ShoppingComp")


@dataclass(frozen=True)
class HfFileRef:
    repo_id: str
    filename: str
    repo_type: str = "dataset"


def _parse_hf_ref(s: str, *, default_repo_id: str) -> Optional[HfFileRef]:
    """
    Supported forms:
    - hf://<repo_id>/<filename>
    - hf://datasets/<repo_id>/<filename>
    - <repo_id>::<filename>
    - <filename>  (interpreted as file inside default_repo_id)
    """
    s = (s or "").strip()
    if not s:
        return None

    if s.startswith("hf://"):
        rest = s[len("hf://") :].lstrip("/")
        # allow optional "datasets/" prefix
        if rest.startswith("datasets/"):
            rest = rest[len("datasets/") :]
        if "/" not in rest:
            return None
        repo_id, filename = rest.split("/", 1)
        if repo_id and filename:
            return HfFileRef(repo_id=repo_id, filename=filename)
        return None

    if "::" in s:
        repo_id, filename = s.split("::", 1)
        repo_id = repo_id.strip()
        filename = filename.strip()
        if repo_id and filename:
            return HfFileRef(repo_id=repo_id, filename=filename)
        return None

    # filename only (no path separators) -> default repo
    if "/" not in s and "\\" not in s:
        return HfFileRef(repo_id=default_repo_id, filename=s)

    # data/<filename> convenience
    base = os.path.basename(s)
    if base and base != s:
        # if user passed something like data/<file> but file doesn't exist locally,
        # we still interpret it as a HF filename.
        return HfFileRef(repo_id=default_repo_id, filename=base)

    return None


def resolve_local_or_hf_file(
    path_or_ref: str,
    *,
    default_repo_id: str = DEFAULT_HF_DATASET_REPO,
    hf_token: Optional[str] = None,
) -> Tuple[str, bool]:
    """
    Returns (local_path, downloaded).

    - If path exists locally, return it with downloaded=False.
    - Otherwise, try to interpret it as a HF reference and download it to local cache.
    """
    path_or_ref = (path_or_ref or "").strip()
    if not path_or_ref:
        raise ValueError("Empty path/ref.")

    if os.path.exists(path_or_ref):
        return path_or_ref, False

    ref = _parse_hf_ref(path_or_ref, default_repo_id=default_repo_id)
    if ref is None:
        raise FileNotFoundError(
            f"File not found: {path_or_ref}\n"
            f"Hint: pass a local path, a filename in HF repo ({default_repo_id}), "
            f"or an explicit HF ref like hf://{default_repo_id}/<filename> or {default_repo_id}::<filename>."
        )

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to auto-download dataset files. "
            "Install with: pip install -U huggingface_hub"
        ) from e

    local_path = hf_hub_download(
        repo_id=ref.repo_id,
        repo_type=ref.repo_type,
        filename=ref.filename,
        token=hf_token,
    )
    return local_path, True

