# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code-first application descriptors with a sample YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True, slots=True)
class ApplicationDescriptor:
    id: str
    title: str
    mode: Literal["foreground", "background"]
    route: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ApplicationCatalog:
    root_prompt: str
    capabilities: dict[str, str]
    applications: dict[str, ApplicationDescriptor]

    def application(self, app_id: str) -> ApplicationDescriptor:
        return self.applications[app_id]


def load_application_catalog(path: Path) -> ApplicationCatalog:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    apps = {
        app_id: ApplicationDescriptor(
            id=app_id,
            title=str(config["title"]),
            mode=str(config["mode"]),
            route=str(config["route"]),
            settings=_resolve_paths(dict(config.get("settings", {})), path.parent),
        )
        for app_id, config in raw["applications"].items()
    }
    if {app.mode for app in apps.values()} - {"foreground", "background"}:
        raise ValueError("application mode must be foreground or background")
    return ApplicationCatalog(
        root_prompt=str(raw["root_prompt"]).strip(),
        capabilities={str(name): str(route) for name, route in raw.get("capabilities", {}).items()},
        applications=apps,
    )


def _resolve_paths(settings: dict[str, Any], base: Path) -> dict[str, Any]:
    if "output_dir" in settings:
        output = Path(str(settings["output_dir"]))
        settings["output_dir"] = output if output.is_absolute() else (base / output).resolve()
    return settings


__all__ = ["ApplicationDescriptor", "ApplicationCatalog", "load_application_catalog"]
