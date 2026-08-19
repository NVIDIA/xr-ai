# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actionable classification for common vLLM startup failures."""
from __future__ import annotations

import re
from pathlib import Path

_KV_AVAILABLE_RE = re.compile(r"Available KV cache memory:\s*([-+]?[0-9.]+)\s*GiB")


def _flag(argv: list[str], name: str) -> str | None:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def classify_vllm_failure(log_path: Path | None, argv: list[str]) -> str | None:
    """Return a concise diagnosis when the captured log has a known signature."""
    if log_path is None:
        return None
    try:
        body = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if "No available memory for the cache blocks" in body:
        available = _KV_AVAILABLE_RE.findall(body)
        cache = (
            f" vLLM calculated {available[-1]} GiB available for KV cache."
            if available else ""
        )
        utilization = _flag(argv, "--gpu-memory-utilization")
        budget = f" The configured utilization was {utilization}." if utilization else ""
        return (
            "INSUFFICIENT VRAM: model weights and runtime allocations exhausted "
            "the vLLM GPU budget before any KV-cache blocks could be created."
            f"{cache}{budget} Re-measure this service and regenerate its absolute-GiB "
            "reservation, or reduce context/concurrency/media limits."
        )
    if "CUDA out of memory" in body or "torch.OutOfMemoryError" in body:
        return (
            "INSUFFICIENT VRAM: CUDA reported an out-of-memory allocation during "
            "vLLM startup. Check the XR-AI preflight table and existing GPU processes, "
            "then re-measure the affected service."
        )
    if "Free memory on device" in body and "desired GPU memory utilization" in body:
        return (
            "INSUFFICIENT FREE VRAM: another process is using memory reserved by this "
            "vLLM service. Stop the listed GPU consumer or choose a profile that fits."
        )
    return None
