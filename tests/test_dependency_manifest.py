# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generated repository-wide dependency manifest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / ".github" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dependency_map = _load("generate_dependency_map")
dependency_manifest = _load("generate_dependency_manifest")


def _write_project(
    root: Path,
    relative_directory: str,
    *,
    name: str,
    dependencies: tuple[str, ...] = (),
    optional_dependencies: dict[str, tuple[str, ...]] | None = None,
    sources: str = "",
    requires_python: str = ">=3.11,<3.13",
) -> None:
    project = root / relative_directory
    project.mkdir(parents=True, exist_ok=True)
    lines = [
        "[project]",
        f"name = {name!r}",
        "version = '0.1.0'",
        f"requires-python = {requires_python!r}",
        f"dependencies = {list(dependencies)!r}",
    ]
    if optional_dependencies:
        lines.extend(("", "[project.optional-dependencies]"))
        lines.extend(f"{extra} = {list(values)!r}" for extra, values in sorted(optional_dependencies.items()))
    if sources:
        lines.extend(("", "[tool.uv.sources]", sources))
    project.joinpath("pyproject.toml").write_text("\n".join(lines) + "\n")


def test_manifest_lists_every_project_with_all_extras(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        "agent-sdk/library",
        name="library",
        optional_dependencies={"vision": ("pillow",), "audio": ("numpy",)},
    )
    _write_project(
        tmp_path,
        "services/server",
        name="server",
        dependencies=("library",),
        sources='library = { path = "../../agent-sdk/library", editable = true }',
    )

    manifest = dependency_manifest.render_manifest(dependency_map.discover_projects(tmp_path))

    assert '    "library[audio,vision]",' in manifest
    assert '    "server",' in manifest
    assert 'library = { path = "../agent-sdk/library", editable = true }' in manifest
    assert 'server  = { path = "../services/server", editable = true }' in manifest
    assert 'requires-python = ">=3.11,<3.13"' in manifest
    assert "package = false" in manifest


def test_manifest_excludes_itself(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library")
    _write_project(tmp_path, "dependency-manifest", name=dependency_manifest.MANIFEST_NAME)

    manifest = dependency_manifest.render_manifest(dependency_map.discover_projects(tmp_path))

    assert dependency_manifest.MANIFEST_NAME not in manifest.split("[tool.uv.sources]")[1]
    assert '"library"' in manifest


def test_manifest_requires_one_python_range(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library")
    _write_project(tmp_path, "services/server", name="server", requires_python=">=3.12")

    with pytest.raises(ValueError, match="requires-python"):
        dependency_manifest.render_manifest(dependency_map.discover_projects(tmp_path))


def test_repository_manifest_is_current() -> None:
    expected = dependency_manifest.render_manifest(dependency_map.discover_projects(_ROOT))

    assert (_ROOT / "dependency-manifest" / "pyproject.toml").read_text() == expected
