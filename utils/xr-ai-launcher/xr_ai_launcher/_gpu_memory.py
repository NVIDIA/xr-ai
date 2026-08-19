# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-YAML GPU-memory requirements, preflight, and vLLM budgets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ._config import read_config_scalar
from ._gpu import GPUDevice

GPU_MEMORY_UTILIZATION_ENV = "XR_AI_GPU_MEMORY_UTILIZATION"


class GPUMemoryError(ValueError):
    """Raised when service GPU requirements are invalid or cannot fit."""


def _require_memory_telemetry(
    gpu: GPUDevice, value: float | None, field: str,
) -> float:
    if value is None:
        raise GPUMemoryError(
            f"GPU {gpu.index} ({gpu.name}): {field} GPU memory telemetry is unavailable"
        )
    return value


@dataclass(frozen=True)
class ServiceGPURequirement:
    """GPU capacity required by one service configuration."""

    service: str
    gpu: int
    memory_gib: float
    vllm: bool
    config_path: Path


@dataclass(frozen=True)
class GPUMemoryPlan:
    """Resolved requirements for one selected service stack."""

    stack: str
    device_safety_reserve_gib: float
    services: tuple[ServiceGPURequirement, ...]


@dataclass(frozen=True)
class GPUPreflight:
    """Capacity result for one GPU used by a plan."""

    gpu: GPUDevice
    requirements: tuple[ServiceGPURequirement, ...]
    active_services: frozenset[str]
    incremental_required_gib: float
    safety_reserve_gib: float
    remaining_gib: float
    passed: bool


def load_service_gpu_requirement(
    service: str,
    path: str | Path,
    inventory: tuple[GPUDevice, ...],
    *,
    vllm: bool,
) -> ServiceGPURequirement:
    """Read one service requirement, retaining utilization as a fallback."""
    config_path = Path(path).resolve()
    gpu_text = read_config_scalar(config_path, "cuda_visible_devices", "0")
    if "," in gpu_text:
        raise GPUMemoryError(
            f"{config_path}: multi-GPU services need an explicit planning policy"
        )
    try:
        gpu_index = int(gpu_text)
    except ValueError as exc:
        raise GPUMemoryError(
            f"{config_path}: cuda_visible_devices must be one numeric GPU index"
        ) from exc

    by_index = {gpu.index: gpu for gpu in inventory}
    if gpu_index not in by_index:
        raise GPUMemoryError(
            f"{config_path}: assigns {service!r} to unavailable GPU {gpu_index}"
        )

    memory_text = read_config_scalar(config_path, "gpu_memory_reservation_gib")
    utilization_text = read_config_scalar(config_path, "gpu_memory_utilization")
    if not memory_text and not utilization_text:
        raise GPUMemoryError(
            f"{config_path}: declare gpu_memory_reservation_gib "
            "(or legacy gpu_memory_utilization)"
        )
    try:
        if memory_text:
            memory_gib = float(memory_text)
        else:
            utilization = float(utilization_text)
            if not 0 < utilization < 1:
                raise ValueError
            total_memory_gib = _require_memory_telemetry(
                by_index[gpu_index], by_index[gpu_index].total_memory_gib, "total",
            )
            memory_gib = utilization * total_memory_gib
    except ValueError as exc:
        raise GPUMemoryError(f"{config_path}: invalid GPU requirement") from exc
    if not math.isfinite(memory_gib) or memory_gib <= 0:
        raise GPUMemoryError(f"{config_path}: GPU requirements must be positive")

    return ServiceGPURequirement(
        service=service,
        gpu=gpu_index,
        memory_gib=memory_gib,
        vllm=vllm,
        config_path=config_path,
    )


def resolve_gpu_memory_plan(
    *,
    stack: str,
    inventory: tuple[GPUDevice, ...],
    service_configs: dict[str, tuple[Path, bool]],
    device_safety_reserve_gib: float = 2.0,
) -> GPUMemoryPlan:
    """Build a plan directly from the selected services' existing YAML files."""
    if not math.isfinite(device_safety_reserve_gib) or device_safety_reserve_gib < 0:
        raise GPUMemoryError("device safety reserve cannot be negative")
    services = tuple(
        load_service_gpu_requirement(service, path, inventory, vllm=vllm)
        for service, (path, vllm) in service_configs.items()
    )
    if not services:
        raise GPUMemoryError(f"{stack}: selected services declare no GPU requirements")
    return GPUMemoryPlan(stack, device_safety_reserve_gib, services)


def preflight_gpu_memory(
    plan: GPUMemoryPlan,
    inventory: tuple[GPUDevice, ...],
    *,
    active_services: frozenset[str] = frozenset(),
) -> tuple[GPUPreflight, ...]:
    """Check incremental requirements against each device's current free memory."""
    by_index = {gpu.index: gpu for gpu in inventory}
    results: list[GPUPreflight] = []
    for index in sorted({item.gpu for item in plan.services}):
        gpu = by_index[index]
        requirements = tuple(item for item in plan.services if item.gpu == index)
        active = frozenset(
            item.service for item in requirements if item.service in active_services
        )
        incremental = sum(
            item.memory_gib for item in requirements if item.service not in active
        )
        free_memory_gib = _require_memory_telemetry(
            gpu, gpu.free_memory_gib, "free",
        )
        remaining = free_memory_gib - incremental - plan.device_safety_reserve_gib
        results.append(GPUPreflight(
            gpu=gpu,
            requirements=requirements,
            active_services=active,
            incremental_required_gib=incremental,
            safety_reserve_gib=plan.device_safety_reserve_gib,
            remaining_gib=remaining,
            passed=remaining >= 0,
        ))
    return tuple(results)


def require_gpu_memory_preflight(results: tuple[GPUPreflight, ...]) -> None:
    """Raise an actionable capacity error when any device cannot fit its services."""
    details = []
    for result in results:
        if result.passed:
            continue
        details.append(
            f"GPU {result.gpu.index} ({result.gpu.name}) has "
            f"{result.gpu.free_memory_gib:.1f} GiB free but needs "
            f"{result.incremental_required_gib:.1f} GiB for services plus "
            f"{result.safety_reserve_gib:.1f} GiB safety reserve "
            f"(shortfall {-result.remaining_gib:.1f} GiB)"
        )
    if details:
        raise GPUMemoryError("GPU MEMORY PREFLIGHT FAILED\n" + "\n".join(details))


def derive_gpu_memory_utilization(memory_gib: float, total_gib: float) -> float:
    """Convert an absolute service requirement to a physical-memory fraction."""
    if (
        not math.isfinite(memory_gib)
        or not math.isfinite(total_gib)
        or memory_gib <= 0
        or total_gib <= 0
    ):
        raise GPUMemoryError("GPU memory requirement and device total must be positive")
    utilization = memory_gib / total_gib
    if utilization >= 1:
        raise GPUMemoryError(
            f"{memory_gib:.1f} GiB cannot fit a {total_gib:.1f} GiB GPU"
        )
    return int(utilization * 10_000 + 0.999999) / 10_000


def utilization_overrides(
    plan: GPUMemoryPlan, inventory: tuple[GPUDevice, ...],
) -> dict[str, str]:
    """Return derived vLLM utilization for services with absolute requirements."""
    by_index = {gpu.index: gpu for gpu in inventory}
    overrides = {}
    for item in plan.services:
        if not item.vllm or not read_config_scalar(
            item.config_path, "gpu_memory_reservation_gib"
        ):
            continue
        gpu = by_index[item.gpu]
        total_memory_gib = _require_memory_telemetry(
            gpu, gpu.total_memory_gib, "total",
        )
        overrides[item.service] = str(derive_gpu_memory_utilization(
            item.memory_gib, total_memory_gib,
        ))
    return overrides


def format_gpu_memory_preflight(
    plan: GPUMemoryPlan, results: tuple[GPUPreflight, ...],
) -> str:
    """Render a compact per-device allocation table."""
    lines = [f"XR-AI GPU memory preflight: {plan.stack}"]
    for result in results:
        total = (
            "N/A" if result.gpu.total_memory_gib is None
            else f"{result.gpu.total_memory_gib:.1f} GiB"
        )
        free = (
            "N/A" if result.gpu.free_memory_gib is None
            else f"{result.gpu.free_memory_gib:.1f} GiB"
        )
        lines.extend([
            "",
            f"GPU {result.gpu.index}: {result.gpu.name}, "
            f"{total} total, {free} free",
        ])
        for item in result.requirements:
            suffix = " (already active)" if item.service in result.active_services else ""
            lines.append(f"  {item.service:<24} {item.memory_gib:>6.1f} GiB{suffix}")
        lines.append(
            f"  {'device safety reserve':<24} {result.safety_reserve_gib:>6.1f} GiB"
        )
        verdict = "PASS" if result.passed else "FAIL"
        lines.append(f"  Result: {verdict}, {result.remaining_gib:.1f} GiB remaining")
    return "\n".join(lines)
