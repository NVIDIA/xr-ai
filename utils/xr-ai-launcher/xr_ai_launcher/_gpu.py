# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict per-device GPU inventory and hardware-profile matching."""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)

_MIB_PER_GIB = 1024.0


class GPUInventoryError(RuntimeError):
    """Raised when GPU hardware cannot be inventoried or matched safely."""


@dataclass(frozen=True)
class GPUProcess:
    """One compute process reported by ``nvidia-smi``."""

    gpu_uuid: str
    pid: int
    name: str
    used_memory_gib: float | None


@dataclass(frozen=True)
class GPUDevice:
    """Physical GPU capacity and current availability."""

    index: int
    uuid: str
    pci_bus_id: str
    name: str
    compute_capability: float
    total_memory_gib: float
    free_memory_gib: float
    used_memory_gib: float
    processes: tuple[GPUProcess, ...] = ()


def _parse_mib(value: str) -> float:
    token = value.strip().split()[0]
    return float(token) / _MIB_PER_GIB


def _query_compute_processes() -> dict[str, list[GPUProcess]]:
    """Best-effort process inventory; capacity detection must still work without it."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    result: dict[str, list[GPUProcess]] = {}
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) != 4:
            continue
        uuid, pid_text, name, memory_text = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        try:
            used = float(memory_text) / _MIB_PER_GIB
        except ValueError:
            used = None
        result.setdefault(uuid, []).append(GPUProcess(uuid, pid, name, used))
    return result


def query_gpu_inventory() -> tuple[GPUDevice, ...]:
    """Return exact per-GPU capacity or fail without selecting a fallback."""
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
            "nvidia-smi is unavailable; XR-AI cannot safely select a GPU profile"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise GPUInventoryError(f"nvidia-smi GPU inventory failed{suffix}") from exc

    processes = _query_compute_processes()
    devices: list[GPUDevice] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=7)]
        if len(parts) != 8:
            raise GPUInventoryError(f"unparseable nvidia-smi GPU row: {line!r}")
        try:
            index = int(parts[0])
            compute_capability = float(parts[4])
            total = _parse_mib(parts[5])
            free = _parse_mib(parts[6])
            used = _parse_mib(parts[7])
        except ValueError as exc:
            raise GPUInventoryError(f"unparseable nvidia-smi GPU row: {line!r}") from exc
        devices.append(GPUDevice(
            index=index,
            uuid=parts[1],
            pci_bus_id=parts[2],
            name=parts[3],
            compute_capability=compute_capability,
            total_memory_gib=total,
            free_memory_gib=free,
            used_memory_gib=used,
            processes=tuple(processes.get(parts[1], ())),
        ))

    if not devices:
        raise GPUInventoryError("nvidia-smi returned no GPUs")
    return tuple(devices)


def _all_match(
    devices: tuple[GPUDevice, ...], *, count: int, min_cap: float,
    max_cap: float | None, min_memory_gib: float,
) -> bool:
    return (
        len(devices) == count
        and all(device.compute_capability >= min_cap for device in devices)
        and (max_cap is None or all(
            device.compute_capability < max_cap for device in devices
        ))
        and all(device.total_memory_gib >= min_memory_gib for device in devices)
    )


def match_gpu_config(devices: tuple[GPUDevice, ...]) -> str:
    """Match only hardware topologies covered by a bundled profile."""
    names = " ".join(device.name.lower() for device in devices)
    if (
        len(devices) == 1
        and devices[0].compute_capability >= 10.0
        and ("gb10" in names or "b10" in names)
    ):
        return "spark"
    if _all_match(
        devices, count=1, min_cap=10.0, max_cap=None, min_memory_gib=80.0,
    ):
        return "96G_blackwell"
    if _all_match(
        devices, count=2, min_cap=8.9, max_cap=10.0, min_memory_gib=44.0,
    ):
        return "dual_48G_ada"

    topology = "; ".join(
        f"GPU {gpu.index}: {gpu.name}, SM{gpu.compute_capability:.1f}, "
        f"{gpu.total_memory_gib:.1f} GiB"
        for gpu in devices
    )
    raise GPUInventoryError(
        "no bundled XR-AI GPU profile safely matches this topology: " + topology
    )


def detect_gpu_config() -> str:
    """Inventory every GPU and return an exact bundled hardware profile."""
    devices = query_gpu_inventory()
    profile = match_gpu_config(devices)
    log.info("GPU config: %s", profile)
    for gpu in devices:
        log.info(
            "GPU %d: %s, SM%.1f, %.1f GiB total, %.1f GiB free",
            gpu.index, gpu.name, gpu.compute_capability,
            gpu.total_memory_gib, gpu.free_memory_gib,
        )
    return profile
