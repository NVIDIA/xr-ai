# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU profile auto-detection via nvidia-smi (stdlib-only)."""
from __future__ import annotations

import logging
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
