# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only prerequisite checks for sample launchers."""
from __future__ import annotations

import ctypes.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from ._config import read_config_scalar
from ._credentials import _KNOWN

if TYPE_CHECKING:
    from ._stack import Parallel, Process

_FIELDS = {"nvidia_driver", "docker", "nvidia_container_toolkit", "vulkan",
           "lovr_config", "disk_gb_free", "ports"}


def _row(name: str, ok: bool, detected: str, required: str, remediation: str) -> dict[str, object]:
    return {"name": name, "ok": ok, "detected": detected,
            "required": required, "remediation": remediation}


def _contract(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: could not read requirements.json"
    if not isinstance(value, dict):
        return None, "contract must be an object"
    if unknown := set(value) - _FIELDS:
        return None, f"contract contains unsupported fields: {', '.join(sorted(unknown))}"
    return value, ""


def _run(command: list[str]) -> tuple[bool, str]:
    if shutil.which(command[0]) is None:
        return False, f"{command[0]} not found"
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, type(exc).__name__
    if result.returncode == 0:
        return True, result.stdout.strip()
    errors = result.stderr.strip().splitlines() or result.stdout.strip().splitlines()
    return False, errors[0] if errors else f"exit {result.returncode}"


def _version(value: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)*", value)
    return tuple(map(int, match.group().split("."))) if match else ()


def _version_check(name: str, command: list[str], minimum: str | int, remediation: str) -> dict[str, object]:
    ran, found = _run(command)
    actual, required = _version(found), _version(str(minimum))
    width = max(len(actual), len(required))
    ok = bool(actual) and actual + (0,) * (width - len(actual)) >= required + (0,) * (width - len(required))
    if not ran and found != f"{command[0]} not found":
        remediation = f"Run `{shlex.join(command)}` and fix the reported error."
    return _row(name, ran and ok, found, f">= {minimum}", remediation)


def _flatten(items: Sequence[Process | Parallel]) -> list[Process]:
    return [process for item in items for process in getattr(item, "processes", (item,))]


def _port_check(process: Process, port: int, proto: str, label: str) -> dict[str, object]:
    mode = process.launch_mode
    flag = "t" if proto == "tcp" else "u"
    command = ["ss", "-H", f"-ln{flag}p", f"sport = :{port}"]
    ran, output = _run(command)
    name = f"port:{label}:{proto}:{port}"
    if not ran:
        remediation = (
            "Install iproute2 and retry."
            if output == "ss not found" else
            f"Run `{shlex.join(command)}` manually. Check that ss runs for this user."
        )
        return _row(name, False, output, "passive port inspection", remediation)
    output = output.splitlines()[0] if output else ""
    occupied = bool(output)
    if mode == "persist" and occupied:
        return _row(name, True, f"listening ({output}); wrapper will validate reuse",
                    f"available or reusable {proto} port", "")
    remediation = "Stop the listener on this port, or change the port in this service's config."
    if occupied and "users:" not in output:
        remediation += " No owning process is visible; check `docker ps`."
    return _row(name, not occupied, f"listening ({output})" if occupied else "available",
                f"available {proto} port", remediation)


def preflight(processes: Sequence[Process | Parallel], base: Path, *,
              credentials: Sequence[str] = (), runtime: bool = True) -> list[dict[str, object]]:
    """Return one result row per applicable check."""
    base = Path(base)
    rows = []
    for name in dict.fromkeys(credentials):
        _label, url, why = _KNOWN.get(name, (name, "", ""))
        remediation = (f"Set {name}. Get credentials at {url}. {why}" if url else
                       f"Set {name}. {why}").rstrip()
        rows.append(_row(f"credential:{name}", bool(os.environ.get(name)),
                         "set" if os.environ.get(name) else "not set", "set", remediation))
    path = base / "requirements.json"
    if not path.is_file():
        return rows
    contract, error = _contract(path)
    if contract is None:
        rows.append(_row("contract", False, error, "valid requirements.json", f"Fix {path}."))
        return rows
    flat = _flatten(processes)
    local = any(process.launch_mode != "reuse" for process in flat)
    if local and (minimum := contract.get("nvidia_driver")):
        rows.append(_version_check("nvidia_driver", ["nvidia-smi", "--query-gpu=driver_version",
                    "--format=csv,noheader"], minimum, "Upgrade the driver: https://nvidia.github.io/xr-ai/latest/getting_started/requirements.html#software"))
    if local and (minimum := contract.get("docker")):
        rows.append(_version_check("docker", ["docker", "version", "--format", "{{.Server.Version}}"],
                    minimum, f"Install Docker {minimum} or newer and allow this user to run it: "
                    "https://nvidia.github.io/xr-ai/latest/getting_started/requirements.html#docker-host-setup"))
    if local and contract.get("nvidia_container_toolkit"):
        ran, output = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
        try:
            runtimes = json.loads(output) if ran else None
        except json.JSONDecodeError:
            runtimes = None
        inspected = isinstance(runtimes, dict)
        available = inspected and "nvidia" in runtimes
        remediation = (
            "Install the NVIDIA Container Toolkit, then run "
            "`sudo nvidia-ctk runtime configure --runtime=docker` and restart Docker."
            if inspected else
            "Run `docker info --format '{{json .Runtimes}}'` manually and fix the reported error."
        )
        rows.append(_row("nvidia_container_toolkit", available,
                         output, "nvidia Docker runtime", remediation))
    if runtime and local and contract.get("vulkan"):
        found = ctypes.util.find_library("vulkan")
        rows.append(_row("vulkan", found is not None, found or "not found", "Vulkan loader",
                         "Install the Vulkan loader (libvulkan1 on Ubuntu)."))
    if config_name := contract.get("lovr_config"):
        needed = platform.machine().lower() == "aarch64"
        found = os.environ.get("LOVR_BIN") or read_config_scalar(base / config_name, "lovr_bin")
        rows.append(_row("lovr", not needed or bool(found), found or ("not required" if not needed else "not set"),
                         "LOVR_BIN or lovr_bin on aarch64", "Build LOVR and set LOVR_BIN: https://nvidia.github.io/xr-ai/latest/guides/troubleshooting.html#dgx-spark-lovr-auto-download-is-not-supported"))
    if local and (minimum := contract.get("disk_gb_free")):
        caches: set[Path] = set()
        for process in (item for item in flat
                        if item.prepare and item.launch_mode != "reuse" and item.config is not None):
            config = (base / process.config).resolve()
            raw = read_config_scalar(config, "model_cache") or read_config_scalar(config, "nim_cache")
            if raw:
                caches.add((config.parent / raw).resolve())
        cold = []
        for cache in caches:
            try:
                if not cache.is_dir() or not any(cache.iterdir()):
                    cold.append(cache)
            except OSError:
                cold.append(cache)
        if cold:
            roots = []
            for cache in cold:
                root = cache
                while not root.exists():
                    root = root.parent
                roots.append(shutil.disk_usage(root).free / 1_000_000_000)
            free = min(roots)
            rows.append(_row("disk", free >= minimum, f"{free:.1f} GB free before first download",
                             f">= {minimum} GB free",
                             f"Free at least {minimum} GB on the model-cache filesystem."))
    if not runtime:
        return rows
    by_name = {process.name: process for process in flat}
    checked: set[tuple[str, int]] = set()
    for item in contract.get("ports", []):
        process = by_name.get(item["name"])
        if process is None or process.launch_mode == "reuse":
            continue
        raw = read_config_scalar((base / process.config).resolve(),
                                 item["config_key"], str(item["port"]))
        try:
            port = int(raw)
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            rows.append(_row(f"port:{item['name']}", False, f"invalid port {raw!r}", "integer 1..65535",
                             f"Fix {item['config_key']} in the process config."))
            continue
        rows.append(_port_check(process, port, item["proto"], item["name"]))
        checked.add((item["name"], port))
    for process in flat:
        port = process.port
        if port is not None and process.launch_mode != "reuse" \
                and (process.name, port) not in checked:
            rows.append(_port_check(process, port, "tcp", process.name))
    return rows
