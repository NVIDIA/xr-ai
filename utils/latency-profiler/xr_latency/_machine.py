# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe the machine and servers a run was measured on, so results from different boxes stay comparable."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import urllib.request
from typing import Any


def _run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith(("model name", "Model")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _memory_gb() -> float | None:
    try:
        with open("/proc/meminfo") as handle:
            return round(int(handle.readline().split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None


def _gpus() -> list[dict[str, str]]:
    out = _run("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")
    rows = [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]
    return [dict(zip(("name", "memory", "driver"), r, strict=False)) for r in rows]


def _served_models(base_url: str) -> list[str]:
    """Model ids an OpenAI-compatible server reports; empty when it is unreachable or not OpenAI-compatible."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=2) as response:  # noqa: S310
            return [m["id"] for m in json.load(response).get("data", [])]
    except (OSError, ValueError, KeyError):
        return []


def fingerprint(endpoints: dict[str, str] | None = None) -> dict[str, Any]:
    """Hardware, OS and served-model facts for this host; ``endpoints`` maps role to the base URL to query."""
    return {
        "host": platform.node(),
        "arch": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "gpus": _gpus(),
        "git_sha": _run("git", "rev-parse", "--short", "HEAD") or None,
        "served_models": {role: _served_models(url) for role, url in (endpoints or {}).items()},
    }


def label(machine: dict[str, Any] | None) -> str:
    """A short human label such as ``spark: 1x NVIDIA GB10``."""
    if not machine:
        return "unknown machine"
    names = [g["name"] for g in machine.get("gpus", [])]
    gpus = f"{len(names)}x {names[0]}" if names and len(set(names)) == 1 else ", ".join(names) or "no GPU"
    return f"{machine.get('host', '?')}: {gpus}"
