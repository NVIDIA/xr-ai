# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict identity probes for persistent model services."""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path

from xr_ai_launcher import OwnershipProbeMismatch

from . import _docker, _lifecycle
from ._config import (
    _LOCAL_CONFIG_DIGEST_ENV,
    _LOCAL_SERVICE_IDENTITY_ENV,
    service_config_digest,
)

_LAUNCH_IDENTITY_RE = re.compile(r"[0-9a-f]{20}")
_DOCKER_PROBE_TIMEOUT_S = 2.0
_MODEL_SERVERS_STOP = (
    "uv run --project model-server-samples/model-servers "
    "model_servers --stop"
)


def _health_ok(port: int, path: str) -> bool:
    return _lifecycle.local_health_ok(
        f"http://127.0.0.1:{port}{path}", timeout=2
    )


def _owner_remediation(owner: str, port: int) -> str:
    return (
        f"Port {port} is owned by {owner}. Stop its owning stack "
        f"(`{_MODEL_SERVERS_STOP}` for model-servers) or configure a "
        "different port, then rerun."
    )


def _local_process_matches(
    config_path: Path,
    command: str,
    port: int,
    env: Mapping[str, str] | None,
    mismatch_remediation: str | None,
) -> bool:
    pid, checked, listening = _docker.pid_on_port_checked(port)
    if not checked or not listening or pid is None:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False
    identity_prefix = f"{_LOCAL_CONFIG_DIGEST_ENV}=".encode()
    actual_identity = next(
        (entry for entry in environment if entry.startswith(identity_prefix)),
        None,
    )
    if actual_identity is None:
        return False
    service_prefix = f"{_LOCAL_SERVICE_IDENTITY_ENV}=".encode()
    actual_service = next(
        (entry for entry in environment if entry.startswith(service_prefix)),
        None,
    )
    if actual_service is not None:
        if actual_service != f"{_LOCAL_SERVICE_IDENTITY_ENV}={command}".encode():
            raise OwnershipProbeMismatch(
                f"managed process {pid} belongs to a different service",
                mismatch_remediation
                or _owner_remediation(f"managed process {pid}", port),
            )
    else:
        if not any(command.encode() in item for item in cmdline):
            return False
        expected_path = str(config_path.resolve()).encode()
        if expected_path not in cmdline:
            raise OwnershipProbeMismatch(
                f"managed process {pid} uses a different service config",
                mismatch_remediation
                or _owner_remediation(f"managed process {pid}", port),
            )
    expected_digest = service_config_digest(config_path, env=env)
    if expected_digest is None:
        return False
    expected_entry = f"{_LOCAL_CONFIG_DIGEST_ENV}={expected_digest}".encode()
    if actual_identity != expected_entry:
        raise OwnershipProbeMismatch(
            f"managed process {pid} has a different launch identity",
            mismatch_remediation
            or _owner_remediation(f"managed process {pid}", port),
        )
    return True


def _describe_launch_identity(
    config_path: Path,
    command: str,
    project: Path | None,
    env: Mapping[str, str] | None,
) -> str | None:
    """Query a wrapper's read-only effective launch description."""
    if project is None:
        return None
    uv = shutil.which("uv")
    if uv is None:
        return None
    child_env = dict(env) if env is not None else None
    try:
        process = subprocess.Popen(
            [
                uv,
                "run",
                "--quiet",
                "--offline",
                "--no-sync",
                "--project",
                str(project.resolve()),
                command,
                "--config",
                str(config_path.resolve()),
                "--describe-launch",
            ],
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError:
        return None
    try:
        stdout, _ = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        _terminate_describe_process_group(process)
        return None
    except (OSError, subprocess.SubprocessError):
        _terminate_describe_process_group(process)
        return None
    except BaseException:
        _terminate_describe_process_group(process)
        raise
    if process.returncode != 0:
        return None
    prefix = _docker._LAUNCH_IDENTITY_PREFIX
    identities = [
        line.removeprefix(prefix)
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]
    if len(identities) != 1 or _LAUNCH_IDENTITY_RE.fullmatch(identities[0]) is None:
        return None
    return identities[0]


def _terminate_describe_process_group(process: subprocess.Popen[str]) -> None:
    """Kill and reap the process group created for one describe query."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        try:
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass


def managed_service_matches(
    config_path: str | Path,
    command: str,
    port: int,
    env: Mapping[str, str] | None = None,
    *,
    needs_docker: bool,
    health_path: str = "/health",
    mismatch_remediation: str | None = None,
    project: str | Path | None = None,
) -> bool:
    """Return whether a healthy listener matches its managed launch identity.

    Docker services must carry matching entry-point and effective-launch
    labels. Their source-config label is informational and is not compared.
    *project* identifies the service's uv project and is required to verify
    Docker compatibility by invoking the wrapper's read-only
    ``--describe-launch`` mode with ``--offline --no-sync``. A missing project,
    failed description, inconclusive Docker query, or ambiguous set of
    labelled containers leaves ownership unverified.

    Local services must expose matching command, config path, and config digest
    through ``/proc``. *env* is the effective environment of a process the
    launcher would start. *health_path* selects the HTTP readiness endpoint.
    *mismatch_remediation* replaces the default non-destructive owner/port
    recovery guidance.

    Raises:
        OwnershipProbeMismatch: A listener is conclusively managed by a
            different service or effective launch, or a legacy Docker
            container lacks the service label needed for strict ownership.
    """
    path = Path(config_path)
    if not _health_ok(port, health_path):
        return False
    if not needs_docker:
        return _local_process_matches(
            path, command, port, env, mismatch_remediation
        )

    holders, checked = _docker.containers_on_port_checked(
        port, timeout=_DOCKER_PROBE_TIMEOUT_S
    )
    if not checked or len(holders) != 1:
        return False
    holder = holders[0]
    snapshot, snapshot_checked = _docker.container_ownership_snapshot_checked(
        holder, timeout=_DOCKER_PROBE_TIMEOUT_S
    )
    if not snapshot_checked or snapshot is None or not snapshot.running:
        return False
    config_label = snapshot.config_label
    service_label = snapshot.service_label
    owner = (
        f"managed container {holder!r} "
        f"(id {snapshot.container_id[:12]})"
    )
    remediation = mismatch_remediation or _owner_remediation(
        owner, port
    )
    if config_label and not service_label:
        raise OwnershipProbeMismatch(
            f"legacy {owner} lacks launch identity labels",
            remediation,
        )
    if not config_label:
        return False
    if service_label != command:
        raise OwnershipProbeMismatch(
            f"{owner} belongs to service {service_label!r}",
            remediation,
        )
    expected_launch = _describe_launch_identity(
        path,
        command,
        Path(project) if project is not None else None,
        env,
    )
    if expected_launch is None:
        return False
    if config_label != expected_launch:
        raise OwnershipProbeMismatch(
            f"{owner} has a different effective launch identity",
            remediation,
        )
    return True
