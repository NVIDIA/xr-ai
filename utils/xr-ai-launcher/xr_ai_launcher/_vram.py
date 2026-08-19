# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Absolute VRAM reservations, derived vLLM budgets, and stack preflight."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._gpu import GPUDevice

VRAM_UTILIZATION_ENV = "XR_AI_GPU_MEMORY_UTILIZATION"


class VRAMProfileError(ValueError):
    """Raised when a reservation profile is invalid or cannot fit."""


@dataclass(frozen=True)
class ServiceReservation:
    """Absolute capacity reserved for one service on one physical GPU."""

    service: str
    gpu: int
    reservation_gib: float
    vllm: bool
    measurement_signature: dict | None = None


@dataclass(frozen=True)
class VRAMProfile:
    """Measured or provisional reservation contract for one complete stack."""

    hardware_profile: str
    stack: str
    status: str
    device_safety_reserve_gib: float
    services: tuple[ServiceReservation, ...]
    source: str | None = None
    source_path: Path | None = None


@dataclass(frozen=True)
class GPUPreflight:
    """One GPU's resolved allocation plan."""

    gpu: GPUDevice
    reservations: tuple[ServiceReservation, ...]
    incremental_required_gib: float
    safety_reserve_gib: float
    remaining_gib: float
    passed: bool


def load_vram_profile(path: str | Path) -> VRAMProfile:
    """Load and validate a versioned JSON reservation profile."""
    profile_path = Path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VRAMProfileError(f"cannot load VRAM profile {profile_path}: {exc}") from exc
    if raw.get("schema_version") != 1:
        raise VRAMProfileError(f"{profile_path}: unsupported schema_version")
    status = raw.get("status")
    if status not in {"provisional", "certified"}:
        raise VRAMProfileError(
            f"{profile_path}: status must be 'provisional' or 'certified'"
        )
    try:
        safety = float(raw["device_safety_reserve_gib"])
        services_raw = raw["services"]
        services = tuple(
            ServiceReservation(
                service=str(service),
                gpu=int(spec["gpu"]),
                reservation_gib=float(spec["reservation_gib"]),
                vllm=bool(spec.get("vllm", False)),
                measurement_signature=(
                    spec.get("measurement_signature")
                    if isinstance(spec.get("measurement_signature"), dict) else None
                ),
            )
            for service, spec in services_raw.items()
        )
        profile = VRAMProfile(
            hardware_profile=str(raw["hardware_profile"]),
            stack=str(raw["stack"]),
            status=status,
            device_safety_reserve_gib=safety,
            services=services,
            source=str(raw["source"]) if raw.get("source") else None,
            source_path=profile_path.resolve(),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise VRAMProfileError(f"{profile_path}: invalid reservation data: {exc}") from exc
    if safety < 0:
        raise VRAMProfileError(f"{profile_path}: safety reserve cannot be negative")
    if not services:
        raise VRAMProfileError(f"{profile_path}: services cannot be empty")
    if any(item.gpu < 0 or item.reservation_gib <= 0 for item in services):
        raise VRAMProfileError(
            f"{profile_path}: GPU indexes must be non-negative and reservations positive"
        )
    return profile


def validate_vram_certification(
    profile: VRAMProfile, service_configs: dict[str, Path],
) -> None:
    """Reject a certified profile when code, driver, or service config changed."""
    if profile.status != "certified":
        return
    assert profile.source_path is not None
    try:
        current_driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise VRAMProfileError("cannot validate certified profile driver version") from exc

    for item in profile.services:
        signature = item.measurement_signature
        if not signature:
            raise VRAMProfileError(
                f"certified service {item.service!r} has no measurement signature"
            )
        if signature.get("driver_version") != current_driver:
            raise VRAMProfileError(
                f"certification for {item.service!r} used driver "
                f"{signature.get('driver_version')}; current driver is {current_driver}"
            )
        config = service_configs.get(item.service)
        hashes = signature.get("config_sha256")
        if config is None or not isinstance(hashes, dict) or not hashes:
            raise VRAMProfileError(
                f"cannot validate certified config for service {item.service!r}"
            )
        try:
            current_hash = hashlib.sha256(config.read_bytes()).hexdigest()
        except OSError as exc:
            raise VRAMProfileError(f"cannot hash service config {config}: {exc}") from exc
        if current_hash not in hashes.values():
            raise VRAMProfileError(
                f"service config changed since {item.service!r} was measured: {config}"
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
    return math.ceil(utilization * 10_000) / 10_000


def preflight_vram(
    profile: VRAMProfile,
    inventory: tuple[GPUDevice, ...],
    *,
    active_services: frozenset[str] = frozenset(),
) -> tuple[GPUPreflight, ...]:
    """Validate incremental stack reservations against current per-GPU free VRAM."""
    by_index = {gpu.index: gpu for gpu in inventory}
    unknown = sorted({item.gpu for item in profile.services} - by_index.keys())
    if unknown:
        raise VRAMProfileError(
            f"VRAM profile assigns services to unavailable GPU indexes: {unknown}"
        )

    results: list[GPUPreflight] = []
    for index in sorted({item.gpu for item in profile.services}):
        gpu = by_index[index]
        reservations = tuple(item for item in profile.services if item.gpu == index)
        incremental = sum(
            item.reservation_gib
            for item in reservations
            if item.service not in active_services
        )
        remaining = gpu.free_memory_gib - incremental - profile.device_safety_reserve_gib
        results.append(GPUPreflight(
            gpu=gpu,
            reservations=reservations,
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
        f"XR-AI GPU preflight: {profile.stack} ({profile.status})",
    ]
    if profile.source:
        lines.append(f"Reservation source: {profile.source}")
    for result in results:
        lines.extend([
            "",
            f"GPU {result.gpu.index}: {result.gpu.name}, "
            f"{result.gpu.total_memory_gib:.1f} GiB total, "
            f"{result.gpu.free_memory_gib:.1f} GiB currently free",
        ])
        lines.extend(
            f"  {item.service:<24} {item.reservation_gib:>6.1f} GiB"
            for item in result.reservations
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
