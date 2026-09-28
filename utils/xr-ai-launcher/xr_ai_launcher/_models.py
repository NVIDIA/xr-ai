# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only process and credential view of a model deployment profile."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ._config import read_config_scalar

LaunchMode = Literal["own", "reuse"]
Ownership = Literal["own", "reuse", "external"]
"""Resolved lifecycle ownership reported as ``own``, ``reuse``, or ``external``.

Deployment profiles spell the corresponding values ``managed``, ``reused``,
and ``external``.
"""


@dataclass(frozen=True)
class EndpointProbe:
    """Resolved readiness endpoint for a managed, reused, or external service."""

    name: str
    """Managed service name, or the model role for an external endpoint."""

    health_url: str | None
    """Resolved readiness URL.

    ``None`` fails a ``health`` readiness policy and is valid only when
    ``readiness`` is ``none``.
    """

    ownership: Ownership
    """Whether the launcher owns, reuses, or connects to the endpoint."""

    readiness: Literal["health", "none"] = "health"
    """Readiness policy selected by the deployment profile."""

    api_key_env: str | None = None
    """Environment variable used as a bearer credential."""

    required_credentials: tuple[str, ...] = ()
    """Credential names needed to launch a managed endpoint."""

    timeout: float = 3.0
    """Maximum seconds for one endpoint readiness probe."""

    role: str | None = None
    """Model role used to distinguish shared-service endpoint checks."""

    endpoint_url: str | None = None
    """Configured service base URL, before appending the readiness path."""


@dataclass(frozen=True)
class ModelDeployment:
    """Launcher-facing process ownership and credentials for model endpoints."""

    profile_path: Path
    """Resolved path of the selected model profile."""

    services: dict[str, Literal["own", "reuse"]]
    """Launcher ownership mode keyed by managed service name."""

    required_credentials: tuple[str, ...]
    """Environment-variable names required by configured model endpoints."""

    endpoint_probes: tuple[EndpointProbe, ...] = field(default_factory=tuple)
    """Profile-selected endpoints with their launcher ownership."""

    def launch_mode(self, service: str) -> Literal["own", "reuse"] | None:
        """Return the configured ownership mode for *service*, if managed."""
        return self.services.get(service)


def load_model_deployment(worker_config: Path) -> ModelDeployment:
    """Load the model deployment selected by a worker configuration.

    Use ``models_config`` when set, otherwise an adjacent ``models.json`` when
    present, and finally ``models.local.json``.
    """

    raw_path = read_config_scalar(worker_config, "models_config")
    if not raw_path:
        adjacent = worker_config.parent / "models.json"
        raw_path = adjacent.name if adjacent.is_file() else "models.local.json"
    profile_path = Path(raw_path)
    if not profile_path.is_absolute():
        profile_path = worker_config.parent / profile_path
    return load_deployment_profile(profile_path)


def load_deployment_profile(profile_path: Path) -> ModelDeployment:
    """Load a JSON deployment profile directly (no worker YAML indirection)."""

    if profile_path.suffix.lower() != ".json":
        raise ValueError(
            f"{profile_path}: launcher model profiles must use a .json file; "
            "YAML profiles are supported only by worker-side xr-ai-models"
        )
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load model profile {profile_path}: {exc}") from exc

    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict):
        raise ValueError(f"{profile_path}: 'models' must be an object")

    services: dict[str, LaunchMode] = {}
    credentials: set[str] = set()
    probes: dict[tuple[str, Ownership], EndpointProbe] = {}
    for role, model in models.items():
        if not isinstance(role, str) or not role:
            raise ValueError(f"{profile_path}: model role names must be strings")
        if not isinstance(model, dict):
            raise ValueError(f"{profile_path}: model role {role!r} must be an object")

        adapter = model.get("adapter")
        endpoint = model.get("endpoint")
        deployment = model.get("deployment", {})
        if not all(isinstance(section, dict) for section in (adapter, endpoint)):
            raise ValueError(
                f"{profile_path}: {role!r} must define adapter, endpoint objects"
            )
        if not isinstance(deployment, dict):
            raise ValueError(
                f"{profile_path}: deployment for {role!r} must be an object"
            )

        readiness = endpoint.get("readiness", "health")
        if readiness not in {"health", "none"}:
            raise ValueError(
                f"{profile_path}: unsupported readiness {readiness!r} for {role!r}"
            )
        credential = endpoint.get("api_key_env")
        if credential is not None:
            if not isinstance(credential, str) or not credential:
                raise ValueError(
                    f"{profile_path}: api_key_env for {role!r} "
                    "must be a non-empty string"
                )
            credentials.add(credential)

        base_url = endpoint.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError(
                f"{profile_path}: role {role!r} needs endpoint.base_url"
            )
        health_path = endpoint.get("health_path", "/health")
        if not isinstance(health_path, str) or not health_path.startswith("/"):
            raise ValueError(
                f"{profile_path}: health_path for {role!r} must start with '/'"
            )
        health_url = base_url.rstrip("/") + health_path

        # Keys the launched service itself needs; see the rationale on
        # xr_ai_models DeploymentSpec.credentials.
        deployment_credentials = deployment.get("credentials", [])
        if not isinstance(deployment_credentials, list):
            raise ValueError(
                f"{profile_path}: deployment credentials for {role!r} must be a list"
            )
        for name in deployment_credentials:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"{profile_path}: deployment credentials for {role!r} must be non-empty strings"
                )

        ownership = deployment.get("ownership", "external")
        if ownership == "external":
            probe = EndpointProbe(
                name=role,
                role=role,
                health_url=health_url,
                endpoint_url=base_url,
                ownership="external",
                readiness=readiness,
                api_key_env=credential,
                required_credentials=tuple(sorted(deployment_credentials)),
            )
            probes[(role, "external")] = probe
            continue
        if ownership == "managed":
            launch_mode: LaunchMode = "own"
            credentials.update(deployment_credentials)
        elif ownership == "reused":
            launch_mode = "reuse"
        else:
            raise ValueError(
                f"{profile_path}: unsupported ownership {ownership!r}"
            )

        service = deployment.get("service")
        if not isinstance(service, str) or not service:
            raise ValueError(
                f"{profile_path}: {ownership} role {role!r} needs a service"
            )
        previous = services.setdefault(service, launch_mode)
        if previous != launch_mode:
            raise ValueError(
                f"{profile_path}: conflicting ownership for service {service!r}"
            )
        probe = EndpointProbe(
            name=service,
            role=role,
            health_url=health_url,
            endpoint_url=base_url,
            ownership=launch_mode,
            readiness=readiness,
            api_key_env=credential,
            required_credentials=tuple(sorted(deployment_credentials)),
        )
        probes[(role, launch_mode)] = probe

    return ModelDeployment(
        profile_path=profile_path,
        services=services,
        required_credentials=tuple(sorted(credentials)),
        endpoint_probes=tuple(probes.values()),
    )
