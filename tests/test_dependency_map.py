# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generated Python dependency inventory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GENERATOR_PATH = _ROOT / ".github" / "scripts" / "generate_dependency_map.py"
_SPEC = importlib.util.spec_from_file_location("generate_dependency_map", _GENERATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
dependency_map = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = dependency_map
_SPEC.loader.exec_module(dependency_map)


def _write_project(
    root: Path,
    relative_directory: str,
    *,
    name: str,
    dependencies: tuple[str, ...] = (),
    optional_dependencies: dict[str, tuple[str, ...]] | None = None,
    sources: str = "",
    scripts: dict[str, str] | None = None,
) -> None:
    project = root / relative_directory
    project.mkdir(parents=True)
    lines = [
        "[build-system]",
        "requires = ['hatchling>=1.0']",
        "build-backend = 'hatchling.build'",
        "",
        "[project]",
        f"name = {name!r}",
        "version = '0.1.0'",
        "requires-python = '>=3.11,<3.13'",
        f"dependencies = {list(dependencies)!r}",
    ]
    if optional_dependencies:
        lines.extend(("", "[project.optional-dependencies]"))
        lines.extend(f"{extra} = {list(values)!r}" for extra, values in sorted(optional_dependencies.items()))
    if scripts:
        lines.extend(("", "[project.scripts]"))
        lines.extend(f"{name} = {target!r}" for name, target in sorted(scripts.items()))
    if sources:
        lines.extend(("", "[tool.uv.sources]", sources))
    project.joinpath("pyproject.toml").write_text("\n".join(lines) + "\n")


def _write_document(root: Path) -> Path:
    document = root / "DEPENDENCIES.md"
    document.write_text(f"# Dependency Map\n\n{dependency_map.START_MARKER}\n\nold\n\n{dependency_map.END_MARKER}\n")
    return document


def test_inventory_is_generated_from_project_metadata(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library")
    _write_project(
        tmp_path,
        "agent-samples/demo",
        name="demo",
        dependencies=("library>=1", "httpx>=0.27"),
        optional_dependencies={"images": ("Pillow>=10",)},
        sources="library = { path = '../../agent-sdk/library', editable = true }",
        scripts={"demo": "demo:main"},
    )
    document = _write_document(tmp_path)

    generated = dependency_map.generate_document(tmp_path, document)

    assert "### Agent SDK" in generated
    assert "### Agent samples" in generated
    assert "`library>=1` → [`library`](agent-sdk/library/) (local, editable)" in generated
    assert "`httpx>=0.27`" in generated
    assert "`images`" in generated
    assert "`Pillow>=10`" in generated
    assert "`demo` → `demo:main`" in generated
    assert generated == dependency_map.replace_generated_section(
        generated,
        dependency_map.render_inventory(dependency_map.discover_projects(tmp_path)),
    )


def test_internal_dependency_requires_matching_local_source(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library")
    _write_project(
        tmp_path,
        "agent-samples/demo",
        name="demo",
        dependencies=("library",),
    )

    with pytest.raises(ValueError, match="needs a tool.uv.sources path"):
        dependency_map.discover_projects(tmp_path)


def test_internal_source_must_target_the_named_project(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library")
    _write_project(tmp_path, "agent-sdk/other", name="other")
    _write_project(
        tmp_path,
        "agent-samples/demo",
        name="demo",
        dependencies=("library",),
        sources="library = { path = '../../agent-sdk/other' }",
    )

    with pytest.raises(ValueError, match="expected agent-sdk/library"):
        dependency_map.discover_projects(tmp_path)


def test_duplicate_project_names_are_rejected(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/one", name="Example.Name")
    _write_project(tmp_path, "utils/two", name="example-name")

    with pytest.raises(ValueError, match="duplicate project name"):
        dependency_map.discover_projects(tmp_path)


@pytest.mark.parametrize(
    "document",
    [
        "# Missing markers\n",
        f"{dependency_map.START_MARKER}\n",
        f"{dependency_map.END_MARKER}\n{dependency_map.START_MARKER}\n",
        (f"{dependency_map.START_MARKER}\n{dependency_map.START_MARKER}\n{dependency_map.END_MARKER}\n"),
    ],
)
def test_generated_section_requires_one_ordered_marker_pair(document: str) -> None:
    with pytest.raises(ValueError):
        dependency_map.replace_generated_section(document, "generated")
