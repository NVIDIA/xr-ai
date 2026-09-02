# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actionable classifications for common vLLM startup failures."""
from __future__ import annotations

import re
from pathlib import Path


def _argument_value(argv: list[str], option: str) -> str | None:
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError):
        return None


def is_cuda_memory_allocation_failure(log_path: str | Path) -> bool:
    """Return whether vLLM failed at the CUDA driver-allocation boundary.

    This deliberately excludes ``torch.OutOfMemoryError`` and KV-cache
    admission failures. On DGX Spark, ``torch.AcceleratorError`` paired with
    ``cudaErrorMemoryAllocation`` identifies the transient UMA cold-start
    failure for which one clean container restart is useful.
    """
    try:
        body = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    lowered = body.lower()
    return "torch.acceleratorerror" in lowered and (
        "cudaerrormemoryallocation" in lowered
        or "cuda error: out of memory" in lowered
    )


def classify_vllm_failure(
    log_path: str | Path,
    argv: list[str],
    *,
    spark_uma: bool = False,
) -> str | None:
    """Return a clear diagnosis when *log_path* contains a GPU-memory failure."""
    try:
        body = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lowered = body.lower()
    utilization = _argument_value(argv, "--gpu-memory-utilization")
    budget = (
        f" (configured gpu_memory_utilization={utilization})"
        if utilization else ""
    )

    if spark_uma and is_cuda_memory_allocation_failure(log_path):
        return (
            "DGX SPARK UMA ALLOCATION FAILURE: the CUDA driver could not allocate "
            "startup memory even though Linux may still report reclaimable memory. "
            "The automatic cold-start retry was exhausted. Stop unrelated "
            "memory-intensive workloads and verify the supported DGX OS and driver."
        )

    if (
        "no available memory for the cache blocks" in lowered
        or (
            "available kv cache memory" in lowered
            and re.search(r"available kv cache memory[^\n]*-\d", lowered)
        )
    ):
        available = re.search(
            r"available kv cache memory[^\n]*?(-?\d+(?:\.\d+)?)\s*gib",
            lowered,
        )
        detail = f"; vLLM reported {available.group(1)} GiB available" if available else ""
        return (
            "INSUFFICIENT GPU MEMORY: model weights and runtime allocations left "
            f"no capacity for the KV cache{detail}{budget}. Reduce the configured "
            "model/context/concurrency, free GPU memory, or use a larger device."
        )

    if "cuda out of memory" in lowered or "torch.outofmemoryerror" in lowered:
        return (
            "INSUFFICIENT GPU MEMORY: CUDA allocation failed during vLLM startup"
            f"{budget}. Free GPU memory, reduce the configured model/context/"
            "concurrency, or use a larger device."
        )

    if "free memory on device" in lowered and "desired gpu memory utilization" in lowered:
        return (
            "INSUFFICIENT FREE GPU MEMORY: another process is using memory required "
            f"by this vLLM service{budget}. Stop the conflicting process or lower "
            "the service memory budget."
        )
    return None
