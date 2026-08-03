# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process ownership from a model deployment profile."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LaunchMode = Literal["own", "reuse"]

def _read_top_level_scalar(path: Path, key: str, default: str) -> str:
    """Read one single-line, top-level YAML scalar without a YAML dependency."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return default

    key_pattern = re.compile(rf"^{re.escape(key)}[ \t]*:[ \t]*(.*)$")
    for line in lines:
        match = key_pattern.fullmatch(line)
        if match is None:
            continue
        raw = match.group(1).strip()
        if not raw or raw[0] in "#|>":
            return default
        if raw[0] in "\"'":
            quote = raw[0]
            end = raw.find(quote, 1)
            suffix = raw[end + 1:].strip() if end >= 0 else ""
            if end < 0 or (suffix and not suffix.startswith("#")):
                return default
            return raw[1:end] or default
        comment = re.search(r"[ \t]+#", raw)
        return raw[:comment.start()].rstrip() if comment else raw
    return default


@dataclass(frozen=True)
class ModelDeployment:
    profile_path: Path
    services: dict[str, LaunchMode]
    required_credentials: tuple[str, ...]

    def launch_mode(self, service: str) -> LaunchMode | None:
        return self.services.get(service)


def load_model_deployment(worker_config: Path) -> ModelDeployment:
    """Load a structured JSON profile selected by a worker config."""
    selected = _read_top_level_scalar(
        worker_config, "models_config", "models.local.json"
    )
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
        adapter = model.get("adapter")
        endpoint = model.get("endpoint")
        deployment = model.get("deployment")
        if not all(isinstance(part, dict) for part in (adapter, endpoint, deployment)):
            raise ValueError(
                f"{profile_path}: {role!r} must define adapter, endpoint, and deployment objects"
            )
        credential = endpoint.get("api_key_env")
        if credential is not None:
            if not isinstance(credential, str) or not credential:
                raise ValueError(
                    f"{profile_path}: api_key_env for {role!r} must be a non-empty string"
                )
            credentials.add(credential)

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
            credentials.add(name)

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
