# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU utilization snapshots so a sample can be tagged with how busy the box was."""

from __future__ import annotations

import asyncio
import subprocess
import urllib.request
from collections.abc import Iterable


def gpu_utilization() -> list[int]:
    """Return per-GPU utilization percent, or an empty list when ``nvidia-smi`` is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in out.split() if line.strip().isdigit()]


async def wait_until_quiet(max_util: int = 15, *, quiet_samples: int = 3, timeout_s: float = 300.0) -> bool:
    """Wait until every GPU stays under ``max_util`` percent for ``quiet_samples`` consecutive checks."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    streak = 0
    while asyncio.get_running_loop().time() < deadline:
        utils = gpu_utilization()
        streak = streak + 1 if utils and max(utils) <= max_util else 0
        if streak >= quiet_samples:
            return True
        await asyncio.sleep(1.0)
    return False


_metrics_urls: tuple[str, ...] = ()


def configure_metrics(base_urls: Iterable[str]) -> None:
    """Set the vLLM servers whose ``/metrics`` ``load_tags`` watches for requests from other clients."""
    global _metrics_urls  # noqa: PLW0603 - one profiler process watches one stack
    _metrics_urls = tuple(dict.fromkeys(f"{u.rstrip('/')}/metrics" for u in base_urls))


def foreign_requests(urls: tuple[str, ...] | None = None) -> int:
    """Requests running or queued on the vLLM servers right now.

    Call it while this profiler has nothing in flight: anything counted then belongs to another client.
    Unreachable metrics count as zero.
    """
    total = 0
    for url in _metrics_urls if urls is None else urls:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:  # noqa: S310 - fixed local http URLs
                text = response.read().decode()
        except (OSError, ValueError):
            continue
        for line in text.splitlines():
            if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
                total += int(float(line.rsplit(" ", 1)[1]))
    return total


def load_tags() -> dict[str, object]:
    """Context tags stored with every sample: GPU utilization and foreign vLLM load."""
    return {"gpu_util": gpu_utilization(), "foreign": foreign_requests()}
