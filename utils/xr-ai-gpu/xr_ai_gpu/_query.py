# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal implementation for xr_ai_gpu driver queries; use the package root API."""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass

_MIB_PER_GIB = 1024.0


class GPUInventoryError(RuntimeError):
    """Raised when a driver query fails or strict inventory cannot be parsed."""


@dataclass(frozen=True)
class GPUProcess:
    """One compute process reported by ``nvidia-smi``."""

    gpu_uuid: str
    """Driver UUID of the GPU hosting the process."""
    pid: int
    """Host process ID."""
    name: str
    """Driver-reported process name."""
    used_memory_gib: float | None
    """Query-time GPU memory use in GiB, or None when unavailable."""


@dataclass(frozen=True)
class GPUDevice:
    """Driver-visible GPU facts, independent of CUDA ordinal remapping.

    Memory values are observations at query time, not reservations. Unknown
    memory is None; it is never interpreted as zero or host memory.
    """

    index: int
    """Physical driver index reported by nvidia-smi, not a CUDA-remapped ordinal."""
    uuid: str
    """Stable GPU UUID."""
    pci_bus_id: str
    """PCI bus address reported by the driver."""
    name: str
    """Driver-reported product name."""
    compute_capability: float
    """CUDA compute capability."""
    total_memory_gib: float | None
    """Total memory in GiB, or None when unavailable."""
    free_memory_gib: float | None
    """Free memory in GiB, or None when unavailable."""
    used_memory_gib: float | None
    """Used memory in GiB, or None when unavailable."""


def _parse_mib(value: str) -> float | None:
    text = value.strip()
    if text in {"N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    if not text:
        raise ValueError("empty memory field")
    return float(text.split()[0]) / _MIB_PER_GIB


def query_gpu_processes(*, timeout: float | None = None) -> dict[str, list[GPUProcess]]:
    """Return compute processes grouped by physical GPU UUID.

    CUDA_VISIBLE_DEVICES does not filter this query. Unknown memory is None.

    Args:
        timeout: Positive, finite driver-command deadline in seconds; None
            permits cold-driver initialization without a deadline.

    Raises:
        GPUInventoryError: The command fails, times out, or contains malformed fields.

        TypeError: timeout is not a float, integer, or None, or is a boolean.

        ValueError: timeout is not positive and finite.
    """
    raw = _query("compute-apps", "gpu_uuid,pid,process_name,used_gpu_memory", timeout=timeout)

    result: dict[str, list[GPUProcess]] = {}
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
        result.setdefault(uuid, []).append(GPUProcess(uuid, pid, name, used))
    return result


def query_gpu_inventory(*, timeout: float | None = None) -> tuple[GPUDevice, ...]:
    """Return a strict inventory of all driver-visible physical GPUs.

    Ignores CUDA_VISIBLE_DEVICES; indices are physical driver indices rather
    than remapped CUDA ordinals. Memory is observed at query time, in GiB,
    with unavailable values represented as None. No device filter is applied.

    Args:
        timeout: Positive, finite driver-command deadline in seconds; None
            permits cold-driver initialization without a deadline.

    Raises:
        GPUInventoryError: The query fails, times out, returns no devices,
            or contains malformed fields.

        TypeError: timeout is not a float, integer, or None, or is a boolean.

        ValueError: timeout is not positive and finite.
    """
    raw = _query(
        "gpu", "index,uuid,pci.bus_id,name,compute_cap,memory.total,memory.free,memory.used",
        timeout=timeout,
    )

    devices: list[GPUDevice] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=7)]
        if len(parts) != 8:
            raise GPUInventoryError(f"unparseable nvidia-smi GPU row: {line!r}")
        try:
            device = GPUDevice(
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


def query_gpu_pci_bus_ids(*, timeout: float | None = None) -> dict[int, str]:
    """Return physical driver indices mapped to driver PCI bus addresses.

    Ignores CUDA_VISIBLE_DEVICES and skips malformed index/address rows. Empty
    output returns an empty mapping. This narrow query does not require compute
    capability or memory support. PCI addresses retain the driver's formatting.

    Args:
        timeout: Positive, finite deadline in seconds; None means no deadline.

    Raises:
        GPUInventoryError: The command cannot run, fails, or times out.

        TypeError: timeout is not a float, integer, or None, or is a boolean.

        ValueError: timeout is not positive and finite.
    """
    raw = _query("gpu", "index,pci.bus_id", timeout=timeout)
    result: dict[int, str] = {}
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        result[index] = parts[1]
    return result


def _query(kind: str, fields: str, *, timeout: float | None) -> str:
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (float, int)):
            raise TypeError("timeout must be a float, integer, or None")
        try:
            finite = math.isfinite(timeout)
        except OverflowError as exc:
            raise ValueError("timeout must be positive and finite") from exc
        if not finite or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
    kwargs = {"timeout": timeout} if timeout is not None else {}
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.PIPE, **kwargs,
        ).strip()
    except FileNotFoundError as exc:
        raise GPUInventoryError("nvidia-smi is unavailable; XR-AI cannot inspect GPUs") from exc
    except subprocess.TimeoutExpired as exc:
        raise GPUInventoryError(f"nvidia-smi {kind} inventory timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise GPUInventoryError(f"nvidia-smi {kind} inventory failed{suffix}") from exc
    except OSError as exc:
        raise GPUInventoryError(f"nvidia-smi {kind} inventory could not run: {exc}") from exc
