# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU profile auto-detection via nvidia-smi (stdlib-only)."""
from __future__ import annotations

import logging
import shutil
import subprocess

log = logging.getLogger(__name__)


def detect_gpu_config() -> str:
    """Return the GPU config profile by querying nvidia-smi.

    Profiles
    --------
    dual_48G_ada   — 2× ADA 48 GB (default / current dev box)
    spark          — 1× Blackwell GB10 (DGX Spark; ~96 GiB GPU-visible HBM)
    96G_blackwell  — 1× Blackwell ~96 GB

    Falls back to ``dual_48G_ada`` on any detection failure.
    """
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except Exception as exc:
        log.warning("nvidia-smi unavailable (%s) — using dual_48G_ada", exc)
        return "dual_48G_ada"

    _SPARK_NAMES = {"gb10", "b10"}

    gpus: list[tuple[str, float, float]] = []
    for line in raw:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, cap_str, mem_str = parts[0], parts[1], parts[2]
        try:
            cap = float(cap_str)
        except ValueError:
            continue
        mem = 0.0
        for tok in mem_str.split():
            try:
                mem = float(tok)
                break
            except ValueError:
                pass
        gpus.append((name.lower(), cap, mem))

    if not gpus:
        log.warning("GPU detection returned no parseable data — using dual_48G_ada")
        return "dual_48G_ada"

    n_gpus       = len(gpus)
    first_name   = gpus[0][0]
    first_cap    = gpus[0][1]
    is_blackwell = first_cap >= 10.0
    is_spark     = any(s in first_name for s in _SPARK_NAMES)
    known_mem    = [m for _, _, m in gpus if m > 0]
    total_mem_gb = sum(known_mem) / 1024 if known_mem else 0.0

    if is_blackwell and (is_spark or (not known_mem)):
        cfg = "spark"
    elif is_blackwell and total_mem_gb >= 120:
        cfg = "spark"
    elif is_blackwell:
        cfg = "96G_blackwell"
    elif n_gpus >= 2:
        cfg = "dual_48G_ada"
    else:
        cfg = "dual_48G_ada"

    mem_display = f"{total_mem_gb:.0f} GiB" if known_mem else "unified memory"
    log.info(
        "GPU config: %s  (%dx %s, %s, SM%.1f)",
        cfg, n_gpus, gpus[0][0].upper(), mem_display, first_cap,
    )
    return cfg


def pick_freest_gpu_env() -> tuple[dict[str, str] | None, str]:
    """Pick the NVIDIA GPU with the most free VRAM and return env vars that
    pin a Vulkan child process to it.

    Returns ``(env_overlay, log_msg)``. ``env_overlay`` is a dict to merge
    into ``os.environ`` (or ``None`` if we couldn't query the GPUs / there
    is nothing to choose from). Failure to pick is non-fatal; callers should
    just not pin.

    Sets ``DRI_PRIME=pci-DDDD_BB_DD_F`` (Mesa device-select implicit-layer
    mechanism that disambiguates same-model GPUs without root),
    ``VK_LOADER_DEVICE_SELECT=PCI:bus:dev:func`` for hosts on Vulkan-Loader
    >= 1.3.207, and ``CUDA_VISIBLE_DEVICES`` for any incidental CUDA paths
    the child or its libs might touch.
    """
    if not shutil.which("nvidia-smi"):
        return None, "no nvidia-smi on PATH; not pinning GPU"

    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,pci.bus_id,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5.0,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return None, f"nvidia-smi failed ({exc}); not pinning GPU"

    rows: list[tuple[int, str, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            bus_id = parts[1]
            free_mb = int(parts[2])
        except ValueError:
            continue
        rows.append((idx, bus_id, free_mb))

    if not rows:
        return None, "nvidia-smi returned no GPUs; not pinning"
    if len(rows) == 1:
        return None, f"only one NVIDIA GPU visible (idx={rows[0][0]}); no pinning needed"

    rows.sort(key=lambda r: r[2], reverse=True)
    best_idx, best_bus, best_free = rows[0]

    # nvidia-smi gives bus_id "00000000:41:00.0" with an 8-hex-digit domain;
    # both pin envs want the standard 4-digit-domain PCI form.
    try:
        domain8, bus, dev_func = best_bus.split(":")
        dev, func = dev_func.split(".")
        domain = domain8[-4:]
    except ValueError:
        return None, f"couldn't parse PCI bus_id {best_bus!r}; not pinning"

    env: dict[str, str] = {
        "DRI_PRIME": f"pci-{domain}_{bus}_{dev}_{func}",
        "VK_LOADER_DEVICE_SELECT": f"PCI:{bus}:{dev}:{func}",
        "CUDA_VISIBLE_DEVICES": str(best_idx),
    }
    summary = (f"pinning to GPU {best_idx} (PCI {best_bus}, "
               f"{best_free} MB free)")
    return env, summary
