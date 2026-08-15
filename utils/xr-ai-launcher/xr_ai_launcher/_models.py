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
Readiness = Literal["health", "none"]


@dataclass(frozen=True)
class DeploymentEndpoint:
    """Profile-selected endpoint and readiness policy for a dependency."""

    base_url: str
    readiness: Readiness
    api_key_env: str | None = None

    @property
    def health_url(self) -> str:
        return self.base_url.rstrip("/") + "/health"


@dataclass(frozen=True)
class ModelDeployment:
    """Launcher-facing process ownership and credentials for model endpoints."""

    profile_path: Path
    services: dict[str, LaunchMode]
    required_credentials: tuple[str, ...]
    reused_endpoints: dict[str, DeploymentEndpoint] = field(default_factory=dict)
    external_endpoints: dict[str, DeploymentEndpoint] = field(default_factory=dict)

    def launch_mode(self, service: str) -> LaunchMode | None:
        return self.services.get(service)


def load_model_deployment(worker_config: Path) -> ModelDeployment:
    """Load the JSON profile selected by a YAML worker configuration."""

    raw_path = read_config_scalar(
        worker_config,
        "models_config",
        "models.local.json",
    ) or "models.local.json"
    profile_path = Path(raw_path)
    if not profile_path.is_absolute():
        profile_path = worker_config.parent / profile_path
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
    reused_endpoints: dict[str, DeploymentEndpoint] = {}
    external_endpoints: dict[str, DeploymentEndpoint] = {}
    credentials: set[str] = set()
    for role, model in models.items():
        if not isinstance(role, str) or not role:
            raise ValueError(f"{profile_path}: model role names must be strings")
        if not isinstance(model, dict):
            raise ValueError(f"{profile_path}: model role {role!r} must be an object")

        adapter = model.get("adapter")
        endpoint = model.get("endpoint")
        deployment = model.get("deployment")
        if not all(
            isinstance(section, dict)
            for section in (adapter, endpoint, deployment)
        ):
            raise ValueError(
                f"{profile_path}: {role!r} must define adapter, endpoint, "
                "and deployment objects"
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

        ownership = deployment.get("ownership", "external")
        if ownership == "external":
            base_url = endpoint.get("base_url")
            if not isinstance(base_url, str) or not base_url:
                raise ValueError(
                    f"{profile_path}: external role {role!r} needs endpoint.base_url"
                )
            external_endpoints[role] = DeploymentEndpoint(
                base_url,
                readiness,
                credential,
            )
            continue
        if ownership == "managed":
            launch_mode: LaunchMode = "own"
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
        if launch_mode == "reuse":
            base_url = endpoint.get("base_url")
            if not isinstance(base_url, str) or not base_url:
                raise ValueError(
                    f"{profile_path}: reused role {role!r} needs endpoint.base_url"
                )
            reused_endpoint = DeploymentEndpoint(base_url, readiness, credential)
            previous_endpoint = reused_endpoints.setdefault(service, reused_endpoint)
            if previous_endpoint != reused_endpoint:
                raise ValueError(
                    f"{profile_path}: conflicting endpoints for reused service "
                    f"{service!r}"
                )

    return ModelDeployment(
        profile_path=profile_path,
        services=services,
        reused_endpoints=reused_endpoints,
        external_endpoints=external_endpoints,
        required_credentials=tuple(sorted(credentials)),
    )
