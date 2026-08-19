# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-YAML VRAM reservations, derived vLLM budgets, and preflight."""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._config import read_config_scalar
from ._gpu import GPUDevice, GPUHardwareProfile

VRAM_UTILIZATION_ENV = "XR_AI_GPU_MEMORY_UTILIZATION"

_CERTIFICATION_KEYS = {
    "gpu_memory_certification_driver",
    "gpu_memory_certification_git",
    "gpu_memory_certification_sha256",
}


class VRAMProfileError(ValueError):
    """Raised when service reservations are invalid or cannot fit."""


@dataclass(frozen=True)
class ServiceReservation:
    """Absolute capacity reserved by one service's existing YAML config."""

    service: str
    gpu: int
    reservation_gib: float
    vllm: bool
    config_path: Path


@dataclass(frozen=True)
class VRAMProfile:
    """Resolved reservation contract for one selected set of services."""

    hardware_profile: str
    stack: str
    device_safety_reserve_gib: float
    services: tuple[ServiceReservation, ...]


@dataclass(frozen=True)
class GPUPreflight:
    """One GPU's resolved allocation plan."""

    gpu: GPUDevice
    reservations: tuple[ServiceReservation, ...]
    active_services: frozenset[str]
    incremental_required_gib: float
    safety_reserve_gib: float
    remaining_gib: float
    passed: bool


def read_service_port(path: str | Path) -> int | None:
    """Read a service's top-level HTTP port without duplicating it in Python."""
    config_path = Path(path)
    raw = read_config_scalar(config_path, "port") or read_config_scalar(
        config_path, "http_port",
    )
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise VRAMProfileError(f"{config_path}: port must be an integer") from exc


def service_config_fingerprint(path: str | Path) -> str:
    """Hash runtime config while excluding certification bookkeeping fields."""
    try:
        config_path = Path(path)
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VRAMProfileError(f"cannot read service config {path}: {exc}") from exc
    kept = [
        line for line in lines
        if line.split(":", 1)[0].strip() not in _CERTIFICATION_KEYS
    ]
    body = "\n".join(kept).rstrip() + "\n"
    return hashlib.sha256(body.encode()).hexdigest()


def _validate_certification(path: Path) -> None:
    expected_hash = read_config_scalar(path, "gpu_memory_certification_sha256")
    expected_driver = read_config_scalar(path, "gpu_memory_certification_driver")
    expected_git = read_config_scalar(path, "gpu_memory_certification_git")
    if not expected_hash and not expected_driver and not expected_git:
        return
    if not expected_hash or not expected_driver or not expected_git:
        raise VRAMProfileError(f"{path}: certified reservation lacks signature fields")
    if service_config_fingerprint(path) != expected_hash:
        raise VRAMProfileError(
            f"{path}: service config changed since its VRAM reservation was certified"
        )
    try:
        driver_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise VRAMProfileError(f"{path}: cannot validate certified driver") from exc
    driver = ",".join(dict.fromkeys(driver_output.splitlines()))
    if driver != expected_driver:
        raise VRAMProfileError(
            f"{path}: reservation used driver {expected_driver}; current driver is {driver}"
        )


def load_service_reservation(
    service: str,
    path: str | Path,
    inventory: tuple[GPUDevice, ...],
    *,
    vllm: bool,
) -> ServiceReservation:
    """Resolve an absolute reservation, retaining utilization as a legacy fallback."""
    config_path = Path(path).resolve()
    gpu_text = read_config_scalar(config_path, "cuda_visible_devices", "0")
    if "," in gpu_text:
        raise VRAMProfileError(
            f"{config_path}: multi-GPU reservations require an explicit placement policy"
        )
    try:
        gpu_index = int(gpu_text)
    except ValueError as exc:
        raise VRAMProfileError(
            f"{config_path}: cuda_visible_devices must be one numeric GPU index"
        ) from exc
    by_index = {gpu.index: gpu for gpu in inventory}
    if gpu_index not in by_index:
        raise VRAMProfileError(
            f"{config_path}: assigns {service!r} to unavailable GPU {gpu_index}"
        )

    reservation_text = read_config_scalar(config_path, "gpu_memory_reservation_gib")
    utilization_text = read_config_scalar(config_path, "gpu_memory_utilization")
    if not reservation_text and not utilization_text:
        raise VRAMProfileError(
            f"{config_path}: declare gpu_memory_reservation_gib "
            "(or legacy gpu_memory_utilization)"
        )
    try:
        if reservation_text:
            reservation = float(reservation_text)
        else:
            utilization = float(utilization_text)
            if not 0 < utilization < 1:
                raise ValueError
            reservation = utilization * by_index[gpu_index].total_memory_gib
    except ValueError as exc:
        raise VRAMProfileError(
            f"{config_path}: invalid GPU memory reservation/utilization"
        ) from exc
    if reservation <= 0:
        raise VRAMProfileError(f"{config_path}: GPU reservation must be positive")
    _validate_certification(config_path)
    return ServiceReservation(
        service=service,
        gpu=gpu_index,
        reservation_gib=reservation,
        vllm=vllm,
        config_path=config_path,
    )


def resolve_vram_profile(
    *,
    stack: str,
    hardware: GPUHardwareProfile,
    inventory: tuple[GPUDevice, ...],
    service_configs: dict[str, tuple[Path, bool]],
) -> VRAMProfile:
    """Build the stack contract directly from its selected service YAML files."""
    reservations = tuple(
        load_service_reservation(service, path, inventory, vllm=vllm)
        for service, (path, vllm) in service_configs.items()
    )
    if not reservations:
        raise VRAMProfileError(f"{stack}: selected services declare no GPU reservations")
    return VRAMProfile(
        hardware_profile=hardware.name,
        stack=stack,
        device_safety_reserve_gib=hardware.device_safety_reserve_gib,
        services=reservations,
    )


def derive_gpu_memory_utilization(reservation_gib: float, total_gib: float) -> float:
    """Convert an absolute vLLM reservation to its physical-memory fraction."""
    if reservation_gib <= 0 or total_gib <= 0:
        raise VRAMProfileError("VRAM reservation and physical total must be positive")
    utilization = reservation_gib / total_gib
    if utilization >= 1.0:
        raise VRAMProfileError(
            f"{reservation_gib:.1f} GiB reservation cannot fit a {total_gib:.1f} GiB GPU"
        )
    # Round upward so decimal serialization never undercuts the GiB contract.
    return int(utilization * 10_000 + 0.999999) / 10_000


def preflight_vram(
    profile: VRAMProfile,
    inventory: tuple[GPUDevice, ...],
    *,
    active_services: frozenset[str] = frozenset(),
) -> tuple[GPUPreflight, ...]:
    """Validate incremental stack reservations against current per-GPU free VRAM."""
    by_index = {gpu.index: gpu for gpu in inventory}
    results: list[GPUPreflight] = []
    for index in sorted({item.gpu for item in profile.services}):
        gpu = by_index[index]
        reservations = tuple(item for item in profile.services if item.gpu == index)
        active = frozenset(
            item.service for item in reservations if item.service in active_services
        )
        incremental = sum(
            item.reservation_gib
            for item in reservations
            if item.service not in active
        )
        remaining = gpu.free_memory_gib - incremental - profile.device_safety_reserve_gib
        results.append(GPUPreflight(
            gpu=gpu,
            reservations=reservations,
            active_services=active,
            incremental_required_gib=incremental,
            safety_reserve_gib=profile.device_safety_reserve_gib,
            remaining_gib=remaining,
            passed=remaining >= 0,
        ))
    return tuple(results)


def require_vram_preflight(results: tuple[GPUPreflight, ...]) -> None:
    """Raise a clear capacity error for any failed GPU allocation."""
    failed = [result for result in results if not result.passed]
    if not failed:
        return
    details = []
    for result in failed:
        shortfall = -result.remaining_gib
        details.append(
            f"GPU {result.gpu.index} ({result.gpu.name}) has "
            f"{result.gpu.free_memory_gib:.1f} GiB free but needs "
            f"{result.incremental_required_gib:.1f} GiB for services plus "
            f"{result.safety_reserve_gib:.1f} GiB safety reserve "
            f"(shortfall {shortfall:.1f} GiB)"
        )
    raise VRAMProfileError("VRAM PREFLIGHT FAILED\n" + "\n".join(details))


def utilization_overrides(
    profile: VRAMProfile,
    inventory: tuple[GPUDevice, ...],
) -> dict[str, str]:
    """Return per-service vLLM utilization values derived from GiB reservations."""
    by_index = {gpu.index: gpu for gpu in inventory}
    return {
        item.service: str(derive_gpu_memory_utilization(
            item.reservation_gib, by_index[item.gpu].total_memory_gib,
        ))
        for item in profile.services
        if item.vllm
    }


def format_vram_preflight(
    profile: VRAMProfile, results: tuple[GPUPreflight, ...],
) -> str:
    """Render a compact, user-facing allocation plan."""
    lines = [
        f"XR-AI GPU preflight: {profile.stack}",
    ]
    for result in results:
        lines.extend([
            "",
            f"GPU {result.gpu.index}: {result.gpu.name}, "
            f"{result.gpu.total_memory_gib:.1f} GiB total, "
            f"{result.gpu.free_memory_gib:.1f} GiB currently free",
        ])
        for item in result.reservations:
            suffix = " (already active)" if item.service in result.active_services else ""
            lines.append(
                f"  {item.service:<24} {item.reservation_gib:>6.1f} GiB{suffix}"
            )
        lines.append(
            f"  {'device safety reserve':<24} "
            f"{result.safety_reserve_gib:>6.1f} GiB"
        )
        verdict = "PASS" if result.passed else "FAIL"
        lines.append(f"  Result: {verdict}, {result.remaining_gib:.1f} GiB remaining")
        for process in result.gpu.processes:
            used = (
                f"{process.used_memory_gib:.1f} GiB"
                if process.used_memory_gib is not None else "unknown VRAM"
            )
            lines.append(f"  Existing PID {process.pid}: {process.name} ({used})")
    return "\n".join(lines)
