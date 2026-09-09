# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Generate the mechanical Python project inventory in DEPENDENCIES.md."""

from __future__ import annotations

import argparse
import difflib
import re
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

START_MARKER = "<!-- BEGIN GENERATED PYTHON DEPENDENCY MAP -->"
END_MARKER = "<!-- END GENERATED PYTHON DEPENDENCY MAP -->"

# The aggregate manifest is generated from the other projects and validated by
# its own workflow; discovery skips its sources so a moved project does not
# fail here before the manifest can be regenerated.
MANIFEST_DIRECTORY = "dependency-manifest"

_IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "_build",
    "build",
    "dist",
    "node_modules",
}
_DEPENDENCY_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_CANONICAL_NAME = re.compile(r"[-_.]+")
_GROUPS = (
    (".", "Repository root"),
    ("agent-sdk", "Agent SDK"),
    ("utils", "Utilities"),
    ("services", "Services"),
    ("agent-samples", "Agent samples"),
    ("tests", "Tests"),
    ("dependency-manifest", "Dependency manifest"),
)


@dataclass(frozen=True)
class Project:
    """Dependency metadata read from one Python project."""

    directory: Path
    relative_directory: Path
    name: str
    requires_python: str
    dependencies: tuple[str, ...]
    optional_dependencies: dict[str, tuple[str, ...]]
    scripts: dict[str, str]
    build_dependencies: tuple[str, ...]
    uv_sources: dict[str, object]


def canonical_name(name: str) -> str:
    """Return the PEP 503 comparison form of a distribution name."""

    return _CANONICAL_NAME.sub("-", name).lower()


def dependency_name(specification: str) -> str:
    """Return the distribution name at the start of a PEP 508 requirement."""

    match = _DEPENDENCY_NAME.match(specification)
    if match is None:
        raise ValueError(f"cannot read dependency name from {specification!r}")
    return canonical_name(match.group(1))


def _string_list(value: object, *, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}: {field} must be a list of strings")
    return tuple(value)


def _string_mapping(value: object, *, field: str, path: Path) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{path}: {field} must map strings to strings")
    return dict(value)


def _is_ignored(path: Path, root: Path) -> bool:
    return any(part in _IGNORED_DIRECTORIES for part in path.relative_to(root).parts)


def discover_projects(root: Path) -> tuple[Project, ...]:
    """Read every repository Python project without importing project code."""

    projects: list[Project] = []
    for pyproject in sorted(root.rglob("pyproject.toml")):
        if _is_ignored(pyproject, root):
            continue
        if pyproject.is_symlink():
            raise ValueError(f"{pyproject}: project metadata must not be a symbolic link")

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project_data = data.get("project")
        if not isinstance(project_data, Mapping):
            raise ValueError(f"{pyproject}: missing [project] table")
        name = project_data.get("name")
        requires_python = project_data.get("requires-python")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{pyproject}: project.name must be a non-empty string")
        if not isinstance(requires_python, str) or not requires_python:
            raise ValueError(f"{pyproject}: project.requires-python must be a string")

        dependencies = _string_list(
            project_data.get("dependencies", []),
            field="project.dependencies",
            path=pyproject,
        )
        raw_optional = project_data.get("optional-dependencies", {})
        if not isinstance(raw_optional, Mapping):
            raise ValueError(f"{pyproject}: project.optional-dependencies must be a table")
        optional = {
            extra: _string_list(
                values,
                field=f"project.optional-dependencies.{extra}",
                path=pyproject,
            )
            for extra, values in raw_optional.items()
            if isinstance(extra, str)
        }
        if len(optional) != len(raw_optional):
            raise ValueError(f"{pyproject}: optional dependency names must be strings")

        scripts = _string_mapping(
            project_data.get("scripts", {}),
            field="project.scripts",
            path=pyproject,
        )
        build_data = data.get("build-system", {})
        if not isinstance(build_data, Mapping):
            raise ValueError(f"{pyproject}: build-system must be a table")
        build_dependencies = _string_list(
            build_data.get("requires", []),
            field="build-system.requires",
            path=pyproject,
        )
        tool_data = data.get("tool", {})
        uv_data = tool_data.get("uv", {}) if isinstance(tool_data, Mapping) else {}
        raw_sources = uv_data.get("sources", {}) if isinstance(uv_data, Mapping) else {}
        if not isinstance(raw_sources, Mapping) or not all(isinstance(key, str) for key in raw_sources):
            raise ValueError(f"{pyproject}: tool.uv.sources must be a string-keyed table")

        directory = pyproject.parent.resolve()
        projects.append(
            Project(
                directory=directory,
                relative_directory=directory.relative_to(root.resolve()),
                name=name,
                requires_python=requires_python,
                dependencies=dependencies,
                optional_dependencies=optional,
                scripts=scripts,
                build_dependencies=build_dependencies,
                uv_sources=dict(raw_sources),
            )
        )

    if not projects:
        raise ValueError(f"{root}: no pyproject.toml files found")
    by_name: dict[str, Project] = {}
    for project in projects:
        key = canonical_name(project.name)
        if previous := by_name.get(key):
            raise ValueError(
                f"duplicate project name {project.name!r}: "
                f"{previous.relative_directory} and {project.relative_directory}"
            )
        by_name[key] = project
    _validate_local_sources(projects, by_name, root.resolve())
    return tuple(projects)


def _all_dependency_names(project: Project) -> set[str]:
    specifications = [*project.dependencies]
    for dependencies in project.optional_dependencies.values():
        specifications.extend(dependencies)
    return {dependency_name(specification) for specification in specifications}


def _validate_local_sources(
    projects: Sequence[Project],
    by_name: Mapping[str, Project],
    root: Path,
) -> None:
    for project in projects:
        if project.relative_directory.parts[:1] == (MANIFEST_DIRECTORY,):
            continue
        dependency_names = _all_dependency_names(project)
        source_names = {canonical_name(name) for name in project.uv_sources}
        unused_sources = source_names - dependency_names
        if unused_sources:
            names = ", ".join(sorted(unused_sources))
            raise ValueError(
                f"{project.relative_directory / 'pyproject.toml'}: "
                f"tool.uv.sources entries are not dependencies: {names}"
            )

        for dependency in dependency_names:
            target = by_name.get(dependency)
            if target is None:
                continue
            source = next(
                (value for name, value in project.uv_sources.items() if canonical_name(name) == dependency),
                None,
            )
            if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
                raise ValueError(
                    f"{project.relative_directory / 'pyproject.toml'}: internal dependency "
                    f"{target.name!r} needs a tool.uv.sources path"
                )
            source_path = (project.directory / source["path"]).resolve()
            if not source_path.is_relative_to(root):
                raise ValueError(
                    f"{project.relative_directory / 'pyproject.toml'}: source for "
                    f"{target.name!r} escapes the repository"
                )
            if source_path != target.directory:
                raise ValueError(
                    f"{project.relative_directory / 'pyproject.toml'}: source for "
                    f"{target.name!r} points to {source_path.relative_to(root)}, "
                    f"expected {target.relative_directory}"
                )


def _source_for(project: Project, name: str) -> object | None:
    return next(
        (value for source_name, value in project.uv_sources.items() if canonical_name(source_name) == name),
        None,
    )


def _external_source_note(source: object) -> str:
    if not isinstance(source, Mapping):
        return ""
    if isinstance(source.get("git"), str):
        revision = next(
            (f"{key}={source[key]}" for key in ("tag", "branch", "rev") if isinstance(source.get(key), str)),
            None,
        )
        suffix = f", {revision}" if revision else ""
        return f" (source: `{source['git']}{suffix}`)"
    if isinstance(source.get("url"), str):
        return f" (source: `{source['url']}`)"
    if isinstance(source.get("index"), str):
        return f" (index: `{source['index']}`)"
    return ""


def _format_dependency(
    specification: str,
    project: Project,
    projects_by_name: Mapping[str, Project],
) -> str:
    name = dependency_name(specification)
    target = projects_by_name.get(name)
    source = _source_for(project, name)
    if target is None:
        return f"`{specification}`{_external_source_note(source)}"

    link = target.relative_directory.as_posix() + "/"
    editable = bool(isinstance(source, Mapping) and source.get("editable") is True)
    suffix = ", editable" if editable else ""
    return f"`{specification}` → [`{target.name}`]({link}) ({'local' + suffix})"


def _render_dependency_list(
    label: str,
    dependencies: Sequence[str],
    project: Project,
    projects_by_name: Mapping[str, Project],
    *,
    indent: str = "",
) -> list[str]:
    if not dependencies:
        return [f"{indent}- {label}: none"]
    lines = [f"{indent}- {label}:"]
    lines.extend(
        f"{indent}  - {_format_dependency(specification, project, projects_by_name)}" for specification in dependencies
    )
    return lines


def render_inventory(projects: Sequence[Project]) -> str:
    """Render a deterministic Markdown inventory for *projects*."""

    projects_by_name = {canonical_name(project.name): project for project in projects}
    grouped: dict[str, list[Project]] = defaultdict(list)
    for project in projects:
        parts = project.relative_directory.parts
        grouped[parts[0] if parts else "."].append(project)

    lines = [
        "<!-- Generated by .github/scripts/generate_dependency_map.py; do not edit. -->",
    ]
    known_groups = {name for name, _title in _GROUPS}
    ordered_groups = [*_GROUPS]
    ordered_groups.extend((name, name.replace("-", " ").title()) for name in sorted(grouped.keys() - known_groups))
    for group, title in ordered_groups:
        group_projects = sorted(
            grouped.get(group, []),
            key=lambda project: project.relative_directory.as_posix(),
        )
        if not group_projects:
            continue
        lines.extend(("", f"### {title}"))
        for project in group_projects:
            path = project.relative_directory.as_posix() + "/"
            lines.extend(
                (
                    "",
                    f"#### `{project.name}` — [`{path}`]({path})",
                    "",
                    f"- Python: `{project.requires_python}`",
                )
            )
            lines.extend(
                _render_dependency_list(
                    "Build dependencies",
                    project.build_dependencies,
                    project,
                    projects_by_name,
                )
            )
            lines.extend(
                _render_dependency_list(
                    "Runtime dependencies",
                    project.dependencies,
                    project,
                    projects_by_name,
                )
            )
            if project.optional_dependencies:
                lines.append("- Optional dependency groups:")
                for extra, dependencies in sorted(project.optional_dependencies.items()):
                    lines.extend(
                        _render_dependency_list(
                            f"`{extra}`",
                            dependencies,
                            project,
                            projects_by_name,
                            indent="  ",
                        )
                    )
            else:
                lines.append("- Optional dependency groups: none")
            if project.scripts:
                lines.append("- Commands:")
                lines.extend(f"  - `{name}` → `{target}`" for name, target in sorted(project.scripts.items()))
            else:
                lines.append("- Commands: none")
    return "\n".join(lines)


def replace_generated_section(document: str, generated: str) -> str:
    """Replace the single generated dependency-map section in *document*."""

    if document.count(START_MARKER) != 1 or document.count(END_MARKER) != 1:
        raise ValueError("DEPENDENCIES.md must contain exactly one generated marker pair")
    start = document.index(START_MARKER) + len(START_MARKER)
    end = document.index(END_MARKER)
    if start >= end:
        raise ValueError("DEPENDENCIES.md generated markers are out of order")
    return f"{document[:start]}\n\n{generated.rstrip()}\n\n{document[end:]}"


def generate_document(root: Path, document_path: Path) -> str:
    """Return the dependency document generated from *root*."""

    projects = discover_projects(root)
    document = document_path.read_text(encoding="utf-8")
    return replace_generated_section(document, render_inventory(projects))


def main(argv: Sequence[str] | None = None) -> int:
    """Update DEPENDENCIES.md, or verify it with ``--check``."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when DEPENDENCIES.md does not match the pyproject files",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    document_path = root / "DEPENDENCIES.md"
    try:
        current = document_path.read_text(encoding="utf-8")
        expected = generate_document(root, document_path)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"dependency map generation failed: {error}")
        return 1

    if args.check:
        if current == expected:
            print("Generated dependency map is current.")
            return 0
        print("DEPENDENCIES.md is stale. Regenerate it with:")
        print("  uv run --script .github/scripts/generate_dependency_map.py")
        print()
        print(
            "".join(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    expected.splitlines(keepends=True),
                    fromfile="DEPENDENCIES.md",
                    tofile="generated DEPENDENCIES.md",
                )
            ),
            end="",
        )
        return 1

    if current == expected:
        print("Generated dependency map is already current.")
        return 0
    document_path.write_text(expected, encoding="utf-8")
    print("Updated DEPENDENCIES.md from repository pyproject.toml files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
