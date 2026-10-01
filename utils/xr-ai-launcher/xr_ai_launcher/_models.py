# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only process, credential, and endpoint views of model profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ._config import read_config_scalar

LaunchMode = Literal["own", "reuse"]
EndpointReadiness = Literal["health", "none"]


@dataclass(frozen=True)
class _ModelEndpoint:
    role: str
    kind: str | None
    base_url: str
    health_path: str | None
    readiness: EndpointReadiness | None
    api_key_env: str | None
    managed_service: str | None


@dataclass(frozen=True)
class ModelDeployment:
    """Launcher-facing process ownership and credentials for model endpoints."""

    profile_path: Path
    """Resolved path of the selected model profile."""

    services: dict[str, Literal["own", "reuse"]]
    """Launcher ownership mode keyed by managed service name."""

    required_credentials: tuple[str, ...]
    """Environment keys required by endpoints and managed services."""

    def launch_mode(self, service: str) -> Literal["own", "reuse"] | None:
        """Return the configured ownership mode for *service*, if managed."""
        return self.services.get(service)


def _read_model_profile(profile_path: Path) -> Any:
    if profile_path.suffix.lower() != ".json":
        raise ValueError(
            f"{profile_path}: launcher model profiles must use a .json file; "
            "YAML profiles are supported only by worker-side xr-ai-models"
        )
    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load model profile {profile_path}: {exc}") from exc


def _load_model_endpoints(profile_path: Path) -> tuple[_ModelEndpoint, ...]:
    """Project literal endpoint fields from worker-compatible JSON shapes."""

    raw = _read_model_profile(profile_path)
    models = raw.get("models", raw)
    endpoints: list[_ModelEndpoint] = []
    for role, model in models.items():
        adapter = model.get("adapter", model)
        endpoint = model.get("endpoint", model)
        deployment = model.get("deployment", {})
        kind = adapter.get("kind")
        base_url = endpoint.get("base_url")
        health_path = endpoint.get("health_path")
        credential = endpoint.get("api_key_env")
        has_legacy = "health_check" in endpoint
        legacy = endpoint.get("health_check")
        readiness = endpoint.get("readiness")
        if (
            not isinstance(role, str) or not role
            or kind is not None and not isinstance(kind, str)
            or not isinstance(base_url, str) or not base_url
            or health_path is not None and not isinstance(health_path, str)
            or credential is not None
            and (not isinstance(credential, str) or not credential)
            or has_legacy and not isinstance(legacy, bool)
            or readiness is not None and readiness not in {"health", "none"}
            or readiness is not None
            and has_legacy and (readiness == "health") != legacy
        ):
            raise ValueError("invalid model endpoint profile")
        if readiness is None:
            readiness = (
                "health" if legacy is True else "none" if legacy is False else None
            )
        endpoints.append(
            _ModelEndpoint(
                role=role,
                kind=kind,
                base_url=base_url,
                health_path=health_path,
                readiness=readiness,
                api_key_env=credential,
                managed_service=(
                    deployment.get("service")
                    if deployment.get("ownership") == "managed"
                    and isinstance(deployment.get("service"), str)
                    else None
                ),
            )
        )
    return tuple(endpoints)


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
    return load_deployment_profile(profile_path)


def load_deployment_profile(profile_path: Path) -> ModelDeployment:
    """Load a JSON deployment profile directly (no worker YAML indirection)."""

    raw = _read_model_profile(profile_path)
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict):
        raise ValueError(f"{profile_path}: 'models' must be an object")

    services: dict[str, LaunchMode] = {}
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
        if not isinstance(readiness, str) or readiness not in {"health", "none"}:
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
        if ownership == "managed":
            launch_mode: LaunchMode = "own"
        elif ownership == "reused":
            launch_mode = "reuse"
        elif ownership != "external":
            raise ValueError(
                f"{profile_path}: unsupported ownership {ownership!r}"
            )

        deployment_credentials = deployment.get("credentials", [])
        if not isinstance(deployment_credentials, list):
            raise ValueError(
                f"{profile_path}: deployment credentials for {role!r} must be a list"
            )
        for name in deployment_credentials:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"{profile_path}: deployment credentials for {role!r} "
                    "must be non-empty strings"
                )
            if ownership == "managed":
                credentials.add(name)

        if ownership == "external":
            continue

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

    return ModelDeployment(
        profile_path=profile_path,
        services=services,
        required_credentials=tuple(sorted(credentials)),
    )
