# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only prerequisite checks for sample launchers."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from ._checks import _row, _run
from ._config import read_config_scalar
from ._credentials import _KNOWN
from ._endpoints import endpoint_checks

if TYPE_CHECKING:
    from ._stack import Parallel, Process

_FIELDS = {"nvidia_driver", "docker", "disk_gb_free"}
_HUB_PORTS = (
    ("lk_port_ws", 7880, "tcp"),
    ("lk_port_tcp", 7881, "tcp"),
    ("lk_port_udp", 7882, "udp"),
)


def _contract(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: could not read requirements.json"
    if not isinstance(value, dict):
        return None, "contract must be an object"
    if unknown := set(value) - _FIELDS:
        return None, f"contract contains unsupported fields: {', '.join(sorted(unknown))}"
    for name in ("nvidia_driver", "docker"):
        if name in value and re.fullmatch(r"\d+(?:\.\d+)*", str(value[name])) is None:
            return None, f"{name} must be a dot-separated numeric version"
    minimum = value.get("disk_gb_free")
    if "disk_gb_free" in value and (
        type(minimum) not in (int, float) or not 0 < minimum < float("inf")
    ):
        return None, "disk_gb_free must be a positive number"
    return value, ""


def _version(value: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)*", value)
    return tuple(map(int, match.group().split("."))) if match else ()


def _version_check(name: str, command: list[str], minimum: str | int | float, remediation: str) -> dict[str, object]:
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
        return _row(name, None, f"listening ({output}); ownership and readiness unverified",
                    f"wrapper-validated service on {proto} port", "The service wrapper validates reuse at startup.")
    remediation = "Stop the listener on this port, or change the port in this service's config."
    if occupied and "users:" not in output:
        remediation += " No owning process is visible; check `docker ps`."
    return _row(name, not occupied, f"listening ({output})" if occupied else "available",
                f"available {proto} port", remediation)


def preflight(processes: Sequence[Process | Parallel], base: Path, *,
              credentials: Sequence[str] = (), runtime: bool = True,
              model_profile: Path | None = None) -> list[dict[str, object]]:
    """Return prerequisite rows with ``passed``, ``failed``, or ``skipped`` status.

    Skipped checks remain unverified. Service wrappers own readiness and reuse.
    """
    base = Path(base)
    rows = []
    for name in dict.fromkeys(credentials):
        _label, url, why = _KNOWN.get(name, (name, "", ""))
        remediation = (f"Set {name}. Get credentials at {url}. {why}" if url else
                       f"Set {name}. {why}").rstrip()
        rows.append(_row(f"credential:{name}", bool(os.environ.get(name)),
                         "set" if os.environ.get(name) else "not set", "set", remediation))
    path = base / "requirements.json"
    contract, error = _contract(path) if path.is_file() else ({}, "")
    if contract is None:
        rows.append(_row("contract", False, error, "valid requirements.json", f"Fix {path}."))
        return rows
    flat = _flatten(processes)
    local = [process for process in flat if process.launch_mode != "reuse"]
    containers = any(
        process.command in {"nim_server", "nim_riva_server"}
        or (process.config is not None
            and read_config_scalar((base / process.config).resolve(), "vllm_backend") == "docker")
        for process in local
    )
    hub = next((process for process in local if process.command == "device_io_hub"), None)
    if local and (minimum := contract.get("nvidia_driver")):
        rows.append(_version_check("nvidia_driver", ["nvidia-smi", "--query-gpu=driver_version",
                    "--format=csv,noheader"], minimum, "Upgrade the driver: https://nvidia.github.io/xr-ai/latest/getting_started/requirements.html#software"))
    if (hub is not None or containers) and (minimum := contract.get("docker")):
        rows.append(_version_check("docker", ["docker", "version", "--format", "{{.Server.Version}}"],
                    minimum, f"Install Docker {minimum} or newer and allow this user to run it: "
                    "https://nvidia.github.io/xr-ai/latest/getting_started/requirements.html#docker-host-setup"))
    if containers:
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
    checked: set[tuple[str, int]] = set()
    if hub is not None:
        for config_key, default, proto in _HUB_PORTS:
            raw = (
                read_config_scalar((base / hub.config).resolve(), config_key, str(default))
                if hub.config is not None else str(default)
            )
            try:
                port = int(raw)
                if not 1 <= port <= 65535:
                    raise ValueError
            except ValueError:
                rows.append(_row(
                    f"port:{hub.name}", False, f"invalid port {raw!r}",
                    "integer 1..65535", f"Fix {config_key} in the DeviceIOHub config.",
                ))
                continue
            rows.append(_port_check(hub, port, proto, hub.name))
            checked.add((hub.name, port))
    for process in flat:
        port = process.port
        if port is not None and process.launch_mode != "reuse" \
                and (process.name, port) not in checked:
            rows.append(_port_check(process, port, "tcp", process.name))
    if hub is not None:
        rows.append(_row("video_codecs", None,
                         "DeviceIOHub checks codec libraries during preparation and startup",
                         "NVIDIA NVDEC and NVENC libraries", ""))
    rows.extend(endpoint_checks(model_profile, {process.name for process in local}))
    return rows
