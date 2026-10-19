# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict per-device GPU inventory and existing config selection."""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)

_MIB_PER_GIB = 1024.0
_SPARK_NAME = re.compile(r"(?i)\b(?:GB10|B10)\b")
_BLACKWELL_96_MIN_GIB = 90.0
_SPARK_MIN_GIB = 120.0


class GPUInventoryError(RuntimeError):
    """Raised when visible GPUs cannot be inventoried or safely classified."""


@dataclass(frozen=True)
class _GPUProcess:
    """One compute process reported by ``nvidia-smi``."""

    gpu_uuid: str
    pid: int
    name: str
    used_memory_gib: float | None


@dataclass(frozen=True)
class _GPUDevice:
    """Physical GPU capacity and current availability."""

    index: int
    uuid: str
    pci_bus_id: str
    name: str
    compute_capability: float
    total_memory_gib: float | None
    free_memory_gib: float | None
    used_memory_gib: float | None


def _parse_mib(value: str) -> float | None:
    text = value.strip()
    if text in {"N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    if not text:
        raise ValueError("empty memory field")
    return float(text.split()[0]) / _MIB_PER_GIB


def _format_gib(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f} GiB"


def _query_compute_processes() -> dict[str, list[_GPUProcess]]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except FileNotFoundError as exc:
        raise GPUInventoryError(
            "nvidia-smi is unavailable while inspecting GPU compute processes"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise GPUInventoryError(
            f"nvidia-smi compute-process inventory failed{suffix}"
        ) from exc

    result: dict[str, list[_GPUProcess]] = {}
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) != 4:
            raise GPUInventoryError(f"unparseable nvidia-smi process row: {line!r}")
        uuid, pid_text, name, memory_text = parts
        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise GPUInventoryError(
                f"unparseable nvidia-smi process row: {line!r}"
            ) from exc
        try:
            used = _parse_mib(memory_text)
        except ValueError as exc:
            raise GPUInventoryError(
                f"unparseable nvidia-smi process row: {line!r}"
            ) from exc
        result.setdefault(uuid, []).append(_GPUProcess(uuid, pid, name, used))
    return result


def _format_compute_processes(devices: tuple[_GPUDevice, ...]) -> str:
    """Return best-effort process context for an unsupported topology."""
    try:
        processes = _query_compute_processes()
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


def _query_gpu_inventory() -> tuple[_GPUDevice, ...]:
    """Return exact per-device GPU facts or fail without an assumed fallback."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,"
                "memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except FileNotFoundError as exc:
        raise GPUInventoryError(
            "nvidia-smi is unavailable; XR-AI cannot inspect GPU capacity"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise GPUInventoryError(f"nvidia-smi GPU inventory failed{suffix}") from exc

    devices: list[_GPUDevice] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=7)]
        if len(parts) != 8:
            raise GPUInventoryError(f"unparseable nvidia-smi GPU row: {line!r}")
        try:
            device = _GPUDevice(
                index=int(parts[0]),
                uuid=parts[1],
                pci_bus_id=parts[2],
                name=parts[3],
                compute_capability=float(parts[4]),
                total_memory_gib=_parse_mib(parts[5]),
                free_memory_gib=_parse_mib(parts[6]),
                used_memory_gib=_parse_mib(parts[7]),
            )
        except ValueError as exc:
            raise GPUInventoryError(f"unparseable nvidia-smi GPU row: {line!r}") from exc
        devices.append(device)

    if not devices:
        raise GPUInventoryError("nvidia-smi returned no GPUs")
    return tuple(devices)


def detect_gpu_config() -> str:
    """Select an existing config only when the complete topology is supported.

    This compatibility bridge remains until capability-based service planning
    replaces named config selection. It deliberately has no fallback.
    """
    devices = _query_gpu_inventory()
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
