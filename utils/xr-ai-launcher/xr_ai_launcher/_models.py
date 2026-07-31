# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process ownership from a model deployment profile."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._config import read_config_scalar

LaunchMode = Literal["own", "reuse"]


@dataclass(frozen=True)
class ModelDeployment:
    profile_path: Path
    services: dict[str, LaunchMode]
    required_credentials: tuple[str, ...]

    def launch_mode(self, service: str) -> LaunchMode | None:
        return self.services.get(service)


def load_model_deployment(worker_config: Path) -> ModelDeployment:
    """Load the profile selected by a worker config without YAML dependencies."""
    selected = read_config_scalar(worker_config, "models_config", "models.local.json")
    profile_path = Path(selected)
    if not profile_path.is_absolute():
        profile_path = worker_config.parent / profile_path
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load model profile {profile_path}: {exc}") from exc

    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict):
        raise ValueError(f"{profile_path}: 'models' must be an object")

    services: dict[str, LaunchMode] = {}
    credentials: set[str] = set()
    for role, model in models.items():
        if not isinstance(model, dict):
            raise ValueError(f"{profile_path}: model role {role!r} must be an object")
        endpoint = model.get("endpoint") or {}
        deployment = model.get("deployment") or {}
        if not isinstance(endpoint, dict) or not isinstance(deployment, dict):
            raise ValueError(f"{profile_path}: invalid endpoint or deployment for {role!r}")
        credential = endpoint.get("api_key_env")
        if credential:
            if not isinstance(credential, str):
                raise ValueError(f"{profile_path}: api_key_env for {role!r} must be a string")
            credentials.add(credential)

        ownership = deployment.get("ownership", "external")
        if ownership == "external":
            continue
        if ownership not in {"managed", "reused"}:
            raise ValueError(f"{profile_path}: unsupported ownership {ownership!r}")
        service = deployment.get("service")
        if not isinstance(service, str) or not service:
            raise ValueError(f"{profile_path}: {ownership} role {role!r} needs a service")
        launch_mode: LaunchMode = "own" if ownership == "managed" else "reuse"
        previous = services.setdefault(service, launch_mode)
        if previous != launch_mode:
            raise ValueError(f"{profile_path}: conflicting ownership for service {service!r}")

    return ModelDeployment(
        profile_path=profile_path,
        services=services,
        required_credentials=tuple(sorted(credentials)),
    )
