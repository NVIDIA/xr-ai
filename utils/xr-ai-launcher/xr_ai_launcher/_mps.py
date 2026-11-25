# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit CUDA MPS configuration before service model initialization."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

log = logging.getLogger(__name__)
CUDA_MPS_ENV_KEYS = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUDA_MPS_CLIENT_PRIORITY")
"""Client environment keys managed by explicit CUDA MPS configuration."""


def cuda_mps_identity(environment: Mapping[str, str] | None = None) -> dict:
    """Probe the effective CUDA MPS environment without changing it.

    Uses this process's environment when *environment* is omitted. An empty or
    absent pipe directory returns ``{"enabled": False}``. Otherwise validate
    the absolute controller path, optional integer compute cap and priority,
    controller ownership and reachability, then return effective settings and
    the probed socket identity. The host control CLI uses the host environment
    with only the pipe directory overridden. Does not initialize CUDA or prove
    participation of a future CUDA context.

    Raises:
        RuntimeError: The configured controller or control CLI is unavailable.
    """
    env = os.environ if environment is None else environment
    raw = env.get("CUDA_MPS_PIPE_DIRECTORY", "")
    if not raw:
        return {"enabled": False}
    if not Path(raw).is_absolute():
        raise ValueError("CUDA MPS requires an absolute CUDA_MPS_PIPE_DIRECTORY")
    pipe = Path(raw).resolve()
    if ":" in str(pipe):
        raise ValueError("CUDA_MPS_PIPE_DIRECTORY cannot contain ':'")
    values = {}
    for key, default, low, high in (
        ("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", 100, 1, 100),
        ("CUDA_MPS_CLIENT_PRIORITY", 0, 0, 1),
    ):
        try:
            value = int(env.get(key, str(default)))
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer from {low} to {high}") from exc
        if not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
        values[key] = value
    control = pipe / "control"
    try:
        if not control.is_socket():
            raise RuntimeError(f"Start a CUDA MPS controller at {pipe} before launching the service")
        if control.stat().st_uid != os.getuid():
            raise RuntimeError("The CUDA MPS controller must belong to the launching user")
        result = subprocess.run(
            ["nvidia-cuda-mps-control"], input="get_server_list\n",
            capture_output=True, text=True, timeout=5,
            env=os.environ | {"CUDA_MPS_PIPE_DIRECTORY": str(pipe)}, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Cannot contact CUDA MPS controller at {pipe}: {result.stderr.strip()}")
        identity = control.stat()
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"nvidia-cuda-mps-control failed for {pipe}: {exc}") from exc
    return {
        "enabled": True, "pipe_directory": str(pipe),
        "active_thread_percentage": values["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"],
        "client_priority": values["CUDA_MPS_CLIENT_PRIORITY"],
        "controller": [identity.st_dev, identity.st_ino, identity.st_ctime_ns],
    }


def configure_cuda_mps(settings: dict) -> dict:
    """Validate service YAML MPS settings and apply them to this process.

    Call before importing inference libraries or initializing CUDA. ``settings``
    requires a boolean ``enabled``. When enabled, optional integer
    ``active_thread_percentage`` (1..100, default 100) and ``client_priority``
    (0..1, default 0) override inherited values. The controller socket directory
    comes from the required absolute ``CUDA_MPS_PIPE_DIRECTORY`` environment
    variable. The controller must be reachable and owned by the launching user.

    With ``enabled: false``, clear all three MPS environment variables. Returns
    a JSON-serializable identity containing effective settings and the controller
    socket identity, suitable for service reuse checks. Does not initialize CUDA
    or start, reconfigure, or stop the controller. Successful preflight does not
    prove that a subsequent CUDA context joined MPS; inspect the controller's
    client list after model initialization.

    Raises:
        RuntimeError: The configured MPS controller is unavailable or inaccessible.
    """
    allowed = {"enabled", "active_thread_percentage", "client_priority"}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError(f"mps must be a mapping containing only {sorted(allowed)}")
    enabled = settings.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("mps.enabled must be a boolean")
    if not enabled:
        if set(settings) != {"enabled"}:
            raise ValueError("disabled mps must contain only enabled: false")
        for key in CUDA_MPS_ENV_KEYS:
            os.environ.pop(key, None)
        log.info("CUDA MPS disabled by service configuration")
        return {"enabled": False}
    values = {}
    for key, default, low, high in (
        ("active_thread_percentage", 100, 1, 100),
        ("client_priority", 0, 0, 1),
    ):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"mps.{key} must be an integer from {low} to {high}")
        values[key] = value
    if not os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        raise ValueError("mps.enabled requires CUDA_MPS_PIPE_DIRECTORY")
    updates = {
        "CUDA_MPS_PIPE_DIRECTORY": os.environ["CUDA_MPS_PIPE_DIRECTORY"],
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(values["active_thread_percentage"]),
        "CUDA_MPS_CLIENT_PRIORITY": str(values["client_priority"]),
    }
    effective = cuda_mps_identity(os.environ | updates)
    updates["CUDA_MPS_PIPE_DIRECTORY"] = effective["pipe_directory"]
    os.environ.update(updates)
    log.info("CUDA MPS requested: %s; verify CUDA clients after initialization", effective)
    return effective
