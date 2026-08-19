# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict per-device GPU inventory and YAML hardware-profile matching."""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._config import read_config_scalar

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


@dataclass(frozen=True)
class GPUHardwareProfile:
    """Declarative topology constraints shared by every deployment."""

    name: str
    required_gpu_count: int
    min_compute_capability: float
    max_compute_capability: float | None
    min_memory_gib_per_gpu: float
    max_memory_gib_per_gpu: float | None
    gpu_name_pattern: str | None
    device_safety_reserve_gib: float
    path: Path


def _parse_mib(value: str) -> float:
    token = value.strip().split()[0]
    return float(token) / _MIB_PER_GIB


def _query_compute_processes() -> dict[str, list[GPUProcess]]:
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


def _optional_float(path: Path, key: str) -> float | None:
    raw = read_config_scalar(path, key)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise GPUInventoryError(f"{path}: {key} must be numeric") from exc


def load_gpu_hardware_profile(path: str | Path) -> GPUHardwareProfile:
    """Load one dependency-free top-level YAML hardware profile."""
    profile_path = Path(path)
    try:
        name = read_config_scalar(profile_path, "profile")
        count = int(read_config_scalar(profile_path, "required_gpu_count"))
        min_cap = float(read_config_scalar(profile_path, "min_compute_capability"))
        min_memory = float(read_config_scalar(profile_path, "min_memory_gib_per_gpu"))
        safety = float(read_config_scalar(profile_path, "device_safety_reserve_gib"))
    except ValueError as exc:
        raise GPUInventoryError(f"{profile_path}: invalid hardware profile scalar") from exc
    if not name or count <= 0 or min_memory <= 0 or safety < 0:
        raise GPUInventoryError(f"{profile_path}: incomplete hardware profile")
    return GPUHardwareProfile(
        name=name,
        required_gpu_count=count,
        min_compute_capability=min_cap,
        max_compute_capability=_optional_float(
            profile_path, "max_compute_capability",
        ),
        min_memory_gib_per_gpu=min_memory,
        max_memory_gib_per_gpu=_optional_float(
            profile_path, "max_memory_gib_per_gpu",
        ),
        gpu_name_pattern=read_config_scalar(profile_path, "gpu_name_pattern") or None,
        device_safety_reserve_gib=safety,
        path=profile_path.resolve(),
    )


def _matches(
    devices: tuple[GPUDevice, ...], profile: GPUHardwareProfile,
) -> bool:
    if len(devices) != profile.required_gpu_count:
        return False
    for gpu in devices:
        if gpu.compute_capability < profile.min_compute_capability:
            return False
        if (
            profile.max_compute_capability is not None
            and gpu.compute_capability >= profile.max_compute_capability
        ):
            return False
        if gpu.total_memory_gib < profile.min_memory_gib_per_gpu:
            return False
        if (
            profile.max_memory_gib_per_gpu is not None
            and gpu.total_memory_gib >= profile.max_memory_gib_per_gpu
        ):
            return False
        if profile.gpu_name_pattern and not re.search(
            profile.gpu_name_pattern, gpu.name,
        ):
            return False
    return True


def match_gpu_config(
    devices: tuple[GPUDevice, ...], profiles: tuple[GPUHardwareProfile, ...],
) -> GPUHardwareProfile:
    """Return the single hardware profile matching every physical GPU."""
    matches = [profile for profile in profiles if _matches(devices, profile)]
    if len(matches) == 1:
        return matches[0]
    topology = "; ".join(
        f"GPU {gpu.index}: {gpu.name}, SM{gpu.compute_capability:.1f}, "
        f"{gpu.total_memory_gib:.1f} GiB"
        for gpu in devices
    )
    if matches:
        names = ", ".join(profile.name for profile in matches)
        raise GPUInventoryError(
            f"multiple GPU profiles match this topology ({names}): {topology}"
        )
    raise GPUInventoryError(
        "no bundled XR-AI GPU profile safely matches this topology: " + topology
    )


def detect_gpu_config(profiles_root: str | Path) -> GPUHardwareProfile:
    """Inventory GPUs and match the YAML profiles below *profiles_root*."""
    root = Path(profiles_root)
    paths = sorted(root.glob("*/gpu_profile.yaml"))
    if not paths:
        raise GPUInventoryError(f"no gpu_profile.yaml files found below {root}")
    devices = query_gpu_inventory()
    profile = match_gpu_config(
        devices, tuple(load_gpu_hardware_profile(path) for path in paths),
    )
    log.info("GPU config: %s", profile.name)
    for gpu in devices:
        log.info(
            "GPU %d: %s, SM%.1f, %.1f GiB total, %.1f GiB free",
            gpu.index, gpu.name, gpu.compute_capability,
            gpu.total_memory_gib, gpu.free_memory_gib,
        )
    return profile
