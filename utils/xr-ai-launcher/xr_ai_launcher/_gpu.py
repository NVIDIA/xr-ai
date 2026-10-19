# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled model-server profile matching over shared GPU facts."""
from __future__ import annotations

import logging
import re

from xr_ai_gpu import GPUDevice as _GPUDevice
from xr_ai_gpu import GPUInventoryError, query_gpu_inventory, query_gpu_processes

log = logging.getLogger(__name__)
_SPARK_NAME = re.compile(r"(?i)\b(?:GB10|B10)\b")
_BLACKWELL_96_MIN_GIB = 90.0
_SPARK_MIN_GIB = 120.0


def _format_gib(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f} GiB"


def _format_compute_processes(devices: tuple[_GPUDevice, ...]) -> str:
    """Return best-effort process context for an unsupported topology."""
    try:
        processes = query_gpu_processes()
    except GPUInventoryError as exc:
        return f"compute-process inventory unavailable ({exc})"

    by_uuid = {gpu.uuid: gpu for gpu in devices}
    details = []
    for uuid, entries in processes.items():
        gpu = by_uuid.get(uuid)
        if gpu is None:
            continue
        for process in entries:
            used = _format_gib(process.used_memory_gib)
            details.append(
                f"GPU {gpu.index} PID {process.pid} {process.name} ({used})"
            )
    return ", ".join(details) if details else "none"


def detect_gpu_config() -> str:
    """Select an existing config only when the complete topology is supported.

    This compatibility bridge remains until capability-based service planning
    replaces named config selection. It deliberately has no fallback.

    Raises:
        GPUInventoryError: Inventory fails or no profile matches the topology.
    """
    devices = query_gpu_inventory()
    if len(devices) == 1:
        gpu = devices[0]
        if gpu.compute_capability >= 10.0:
            if _SPARK_NAME.search(gpu.name) or (
                gpu.total_memory_gib is not None
                and gpu.total_memory_gib >= _SPARK_MIN_GIB
            ):
                config = "spark"
            elif (
                gpu.total_memory_gib is not None
                and gpu.total_memory_gib >= _BLACKWELL_96_MIN_GIB
            ):
                config = "96G_blackwell"
            else:
                config = ""
        else:
            config = ""
    elif (
        len(devices) == 2
        and all(8.9 <= gpu.compute_capability < 10.0 for gpu in devices)
        and all(
            gpu.total_memory_gib is not None and gpu.total_memory_gib >= 44.0
            for gpu in devices
        )
    ):
        config = "dual_48G_ada"
    else:
        config = ""

    topology = "; ".join(
        f"GPU {gpu.index}: {gpu.name}, SM{gpu.compute_capability:.1f}, "
        f"{_format_gib(gpu.total_memory_gib)} total, "
        f"{_format_gib(gpu.free_memory_gib)} free"
        for gpu in devices
    )
    if not config:
        process_context = _format_compute_processes(tuple(devices))
        raise GPUInventoryError(
            "no existing XR-AI model-server configuration matches the visible "
            f"GPU topology: {topology}; active compute processes: {process_context}"
        )
    log.info("GPU config: %s", config)
    log.info("GPU inventory: %s", topology)
    return config
