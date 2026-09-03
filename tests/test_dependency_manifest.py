# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dependency manifest generator."""

from __future__ import annotations

import importlib.util
import subprocess
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

MANIFEST = dependency_manifest.MANIFEST_DIRECTORY


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


def _render(root: Path) -> str:
    return dependency_manifest.manifest_toml(dependency_map.discover_projects(root))


def _write_repo(root: Path, *, stale: bool = False, lock: str = "lock v1\n") -> Path:
    """Create a library, a server, and a manifest directory ready for main()."""

    _write_project(root, "agent-sdk/library", name="library")
    _write_project(
        root,
        "services/server",
        name="server",
        dependencies=("library",),
        sources='library = { path = "../../agent-sdk/library", editable = true }',
    )
    manifest_dir = root / MANIFEST
    manifest_dir.mkdir()
    manifest = _render(root)
    if stale:
        manifest = manifest.replace('version = "0.0.0"', 'version = "0.0.1"')
    manifest_dir.joinpath("pyproject.toml").write_text(manifest)
    manifest_dir.joinpath("uv.lock").write_text(lock)
    return manifest_dir


def _fake_lock(calls: list[Path], *, write: str | None = None, error: Exception | None = None):
    def lock_manifest(root: Path) -> None:
        calls.append(root)
        if error is not None:
            raise error
        if write is not None:
            (root / MANIFEST / "uv.lock").write_text(write)

    return lock_manifest


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

    manifest = _render(tmp_path)

    assert '    "library[audio,vision]",' in manifest
    assert '    "server",' in manifest
    assert '"library" = { path = "../agent-sdk/library", editable = true }' in manifest
    assert '"server"  = { path = "../services/server", editable = true }' in manifest
    assert 'requires-python = ">=3.11,<3.13"' in manifest
    assert "package = false" in manifest


def test_manifest_orders_by_path_not_name(tmp_path: Path) -> None:
    _write_project(tmp_path, "zeta/first", name="aaa")
    _write_project(tmp_path, "alpha/second", name="zzz")

    manifest = _render(tmp_path)

    assert manifest.index('"zzz"') < manifest.index('"aaa"')


def test_manifest_excludes_its_own_directory_only(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library")
    _write_project(tmp_path, MANIFEST, name="anything")
    _write_project(tmp_path, "utils/twin", name=dependency_manifest.MANIFEST_NAME)

    manifest = _render(tmp_path)

    assert '"anything"' not in manifest
    assert "anything =" not in manifest
    assert f'"{dependency_manifest.MANIFEST_NAME}"' in manifest
    assert '"library"' in manifest


def test_manifest_narrows_python_range_to_what_every_project_accepts(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library")
    _write_project(tmp_path, "services/server", name="server", requires_python=">= 3.12")

    assert 'requires-python = ">=3.12,<3.13"' in _render(tmp_path)


def test_manifest_rejects_disjoint_python_ranges(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library", requires_python=">=3.11,<3.12")
    _write_project(tmp_path, "services/server", name="server", requires_python=">=3.12")

    with pytest.raises(ValueError, match="no supported Python version"):
        _render(tmp_path)


def test_manifest_quotes_dotted_project_names(tmp_path: Path) -> None:
    _write_project(tmp_path, "utils/dotted", name="foo.bar")

    manifest = _render(tmp_path)

    assert '"foo.bar" = { path = "../utils/dotted", editable = true }' in manifest
    assert '    "foo.bar",' in manifest


def test_manifest_requires_projects() -> None:
    with pytest.raises(ValueError, match="no projects"):
        dependency_manifest.manifest_toml(())


def test_discovery_tolerates_stale_manifest_sources(tmp_path: Path) -> None:
    _write_project(tmp_path, "services/moved", name="library")
    _write_project(
        tmp_path,
        MANIFEST,
        name=dependency_manifest.MANIFEST_NAME,
        dependencies=("library",),
        sources='library = { path = "../agent-sdk/library", editable = true }',
    )

    names = {project.name for project in dependency_map.discover_projects(tmp_path)}

    assert names == {"library", dependency_manifest.MANIFEST_NAME}


def test_check_reports_stale_manifest_without_locking(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest_dir = _write_repo(tmp_path, stale=True)
    calls: list[Path] = []
    monkeypatch.setattr(dependency_manifest, "lock_manifest", _fake_lock(calls))

    assert dependency_manifest.main(["--check"], root=tmp_path) == 1

    out = capsys.readouterr().out
    assert "pyproject.toml is stale" in out
    assert "+++ generated dependency-manifest/pyproject.toml" in out
    assert calls == []
    assert 'version = "0.0.1"' in manifest_dir.joinpath("pyproject.toml").read_text()
    assert manifest_dir.joinpath("uv.lock").read_text() == "lock v1\n"


def test_check_reports_stale_lock_and_restores_it(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest_dir = _write_repo(tmp_path)
    monkeypatch.setattr(dependency_manifest, "lock_manifest", _fake_lock([], write="lock v2\n"))

    assert dependency_manifest.main(["--check"], root=tmp_path) == 1

    assert "uv.lock is stale" in capsys.readouterr().out
    assert manifest_dir.joinpath("uv.lock").read_text() == "lock v1\n"


def test_check_reports_lock_failure_distinctly(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest_dir = _write_repo(tmp_path)
    error = subprocess.CalledProcessError(2, ["uv", "lock"])
    monkeypatch.setattr(dependency_manifest, "lock_manifest", _fake_lock([], error=error))

    assert dependency_manifest.main(["--check"], root=tmp_path) == 1

    out = capsys.readouterr().out
    assert "lock failed" in out
    assert "stale" not in out
    assert manifest_dir.joinpath("uv.lock").read_text() == "lock v1\n"


def test_check_passes_when_current(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_repo(tmp_path)
    monkeypatch.setattr(dependency_manifest, "lock_manifest", _fake_lock([], write="lock v1\n"))

    assert dependency_manifest.main(["--check"], root=tmp_path) == 0
    assert "are current" in capsys.readouterr().out


def test_check_reports_missing_lock(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest_dir = _write_repo(tmp_path)
    manifest_dir.joinpath("uv.lock").unlink()
    monkeypatch.setattr(dependency_manifest, "lock_manifest", _fake_lock([]))

    assert dependency_manifest.main(["--check"], root=tmp_path) == 1
    assert "generation failed" in capsys.readouterr().out


def test_write_regenerates_and_locks(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest_dir = _write_repo(tmp_path, stale=True)
    calls: list[Path] = []
    monkeypatch.setattr(dependency_manifest, "lock_manifest", _fake_lock(calls, write="lock v2\n"))

    assert dependency_manifest.main([], root=tmp_path) == 0

    assert "Updated" in capsys.readouterr().out
    assert calls == [tmp_path]
    assert manifest_dir.joinpath("pyproject.toml").read_text() == _render(tmp_path)
    assert manifest_dir.joinpath("uv.lock").read_text() == "lock v2\n"


def test_write_reports_already_current(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_repo(tmp_path)
    monkeypatch.setattr(dependency_manifest, "lock_manifest", _fake_lock([], write="lock v1\n"))

    assert dependency_manifest.main([], root=tmp_path) == 0
    assert "already current" in capsys.readouterr().out


def test_generation_error_is_reported(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_repo(tmp_path)
    _write_project(tmp_path, "utils/other", name="other", requires_python=">=3.13")
    monkeypatch.setattr(dependency_manifest, "lock_manifest", _fake_lock([]))

    assert dependency_manifest.main(["--check"], root=tmp_path) == 1
    assert "generation failed" in capsys.readouterr().out
