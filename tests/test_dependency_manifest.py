# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dependency manifest generator."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

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
    projects = dependency_map.discover_projects(root)
    return dependency_manifest.manifest_toml(projects, dependency_manifest.read_isolated_inventories(root, projects))


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
    assert 'conflicts = [[{ extra = "repository" }, { extra = "vllm" }]]' in manifest


def test_manifest_isolates_vllm_projects_without_overriding_dependencies(
    tmp_path: Path,
) -> None:
    _write_project(tmp_path, "services/server", name="server")
    _write_project(tmp_path, "services/embedding", name="embedding-server")

    manifest = _render(tmp_path)

    repository = manifest.split("repository = [", 1)[1].split("]", 1)[0]
    vllm = manifest.split("vllm = [", 1)[1].split("]", 1)[0]
    assert '"server"' in repository
    assert '"embedding-server"' not in repository
    assert '"embedding-server"' in vllm
    assert '"server"' not in vllm
    assert "override-dependencies" not in manifest


def test_manifest_lock_inventories_both_pynvvideocodec_environments() -> None:
    lock = tomllib.loads((_ROOT / MANIFEST / "uv.lock").read_text())
    versions = {package["version"] for package in lock["package"] if package["name"] == "pynvvideocodec"}

    assert {"2.0.4", "2.2.2"} <= versions


def test_manifest_lock_preserves_generated_environment_split() -> None:
    lock = tomllib.loads((_ROOT / MANIFEST / "uv.lock").read_text())
    manifest_package = next(
        package
        for package in lock["package"]
        if package["name"] == dependency_manifest.MANIFEST_NAME
    )
    extras = manifest_package["optional-dependencies"]
    vllm_names = {item["name"] for item in extras["vllm"]}
    repository_names = {item["name"] for item in extras["repository"]}

    assert vllm_names == dependency_manifest.VLLM_PROJECTS
    assert vllm_names.isdisjoint(repository_names)
    assert lock["conflicts"] == [[
        {"package": dependency_manifest.MANIFEST_NAME, "extra": "repository"},
        {"package": dependency_manifest.MANIFEST_NAME, "extra": "vllm"},
    ]]


def test_manifest_orders_by_path_not_name(tmp_path: Path) -> None:
    _write_project(tmp_path, "zeta/first", name="aaa")
    _write_project(tmp_path, "alpha/second", name="zzz")

    manifest = _render(tmp_path)

    assert manifest.index('"zzz"') < manifest.index('"aaa"')


def test_manifest_excludes_generated_and_application_projects(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library")
    _write_project(tmp_path, MANIFEST, name="anything")
    _write_project(tmp_path, "apps/private-app", name="private-app")
    _write_project(tmp_path, "utils/twin", name=dependency_manifest.MANIFEST_NAME)

    manifest = _render(tmp_path)

    assert '"anything"' not in manifest
    assert "anything =" not in manifest
    assert '"private-app"' not in manifest
    assert f'"{dependency_manifest.MANIFEST_NAME}"' in manifest
    assert '"library"' in manifest


def test_manifest_narrows_python_range_to_what_every_project_accepts(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library")
    _write_project(tmp_path, "services/server", name="server", requires_python=">= 3.12")

    assert 'requires-python = ">=3.12,<3.13"' in _render(tmp_path)


def _configured_inventory(root: Path):
    _write_project(
        root, "services/custom-asr", name="custom-asr", requires_python=">=3.12,<3.14",
        optional_dependencies={"audio": ("soundfile",)},
    )
    project_path = root / "services/custom-asr/pyproject.toml"
    project_path.write_text(project_path.read_text() + '\n[tool.uv]\nenvironments = ["sys_platform == \'win32\'"]\n')
    config = root / dependency_manifest.INVENTORY_CONFIG
    config.parent.mkdir(exist_ok=True)
    config.write_text(
        '[isolated.custom]\nproject = "services/custom-asr"\npython = "3.12.12"\n'
        'refresh-packages = ["nemo-toolkit"]\nenvironment = { NO_VCS_VERSION = "1" }\n'
    )
    return dependency_manifest.read_isolated_inventories(root, dependency_map.discover_projects(root))[0]


def test_configured_inventory_derives_project_metadata_and_platform(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library", requires_python=">=3.11,<3.12")
    inventory = _configured_inventory(tmp_path)

    shared = tomllib.loads(_render(tmp_path))
    isolated = tomllib.loads(dependency_manifest.isolated_manifest_toml(inventory))

    assert shared["project"]["requires-python"] == ">=3.11,<3.12"
    assert "custom-asr" not in shared["tool"]["uv"]["sources"]
    assert isolated["project"]["name"] == "custom-asr-dependency-manifest"
    assert isolated["project"]["requires-python"] == ">=3.12,<3.14"
    assert isolated["project"]["dependencies"] == ["custom-asr[audio]"]
    assert SpecifierSet(isolated["project"]["requires-python"]).contains(inventory.python)
    assert isolated["tool"]["uv"]["environments"] == ["sys_platform == 'win32'"]
    assert isolated["tool"]["uv"]["sources"] == {
        "custom-asr": {"path": "../../services/custom-asr", "editable": True},
    }


def test_configuration_isolates_multiple_projects(tmp_path: Path) -> None:
    _write_project(tmp_path, "agent-sdk/library", name="library")
    first = _configured_inventory(tmp_path)
    _write_project(tmp_path, "services/other", name="other", requires_python=">=3.13")
    config = tmp_path / dependency_manifest.INVENTORY_CONFIG
    config.write_text(config.read_text() + '\n[isolated.other]\nproject = "services/other"\npython = "3.13.7"\n')
    projects = dependency_map.discover_projects(tmp_path)
    inventories = dependency_manifest.read_isolated_inventories(tmp_path, projects)

    assert [item.directory for item in inventories] == [first.directory, Path(MANIFEST) / "other"]
    assert [project.name for project in dependency_manifest.manifest_members(projects, inventories)] == ["library"]
    other = tomllib.loads(dependency_manifest.isolated_manifest_toml(inventories[1]))
    assert other["project"]["dependencies"] == ["other"]
    assert "environments" not in other["tool"]["uv"]


@pytest.mark.parametrize("replacement", [
    'project = "services/missing"\npython = "3.12.12"',
    'project = "services/custom-asr"\npython = "3.11.14"',
    'project = "services/custom-asr"\npython = "3.12"',
])
def test_configuration_rejects_missing_projects_or_incompatible_python(tmp_path: Path, replacement: str) -> None:
    _configured_inventory(tmp_path)
    (tmp_path / dependency_manifest.INVENTORY_CONFIG).write_text('[isolated.custom]\n' + replacement + '\n')
    with pytest.raises(ValueError):
        dependency_manifest.read_isolated_inventories(tmp_path, dependency_map.discover_projects(tmp_path))


def test_committed_inventories_preserve_both_nemo_sources() -> None:
    inventories = dependency_manifest.read_isolated_inventories(_ROOT, dependency_map.discover_projects(_ROOT))
    inventory = next(item for item in inventories if item.project.relative_directory == Path("services/speaker-stt"))
    shared = tomllib.loads((_ROOT / MANIFEST / "uv.lock").read_text())
    isolated = tomllib.loads((_ROOT / inventory.directory / "uv.lock").read_text())
    runtime = tomllib.loads((inventory.project.directory / "pyproject.toml").read_text())
    manifest = tomllib.loads((_ROOT / inventory.directory / "pyproject.toml").read_text())
    assert manifest["tool"]["uv"]["environments"] == runtime["tool"]["uv"]["environments"]

    registry_nemo = [package for package in shared["package"] if package["name"] == "nemo-toolkit"]
    isolated_nemo = next(package for package in isolated["package"] if package["name"] == "nemo-toolkit")
    isolated_lhotse = next(package for package in isolated["package"] if package["name"] == "lhotse")
    source = runtime["tool"]["uv"]["sources"]["nemo-toolkit"]
    assert registry_nemo
    assert all(package["source"] == {"registry": "https://pypi.org/simple"} for package in registry_nemo)
    assert source["git"] in isolated_nemo["source"]["git"]
    assert source["rev"] in isolated_nemo["source"]["git"]
    assert isolated_lhotse["source"] == runtime["tool"]["uv"]["sources"]["lhotse"]


@pytest.mark.parametrize("fresh", ["original lock\n", "changed lock\n"])
def test_isolated_check_compares_and_restores_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh: str) -> None:
    inventory = _configured_inventory(tmp_path)
    directory = tmp_path / inventory.directory
    directory.mkdir(parents=True)
    (directory / "pyproject.toml").write_text(dependency_manifest.isolated_manifest_toml(inventory))
    lock = directory / "uv.lock"
    lock.write_text("original lock\n")
    calls = []

    def run(args, *, cwd, check, env):
        calls.append((args, cwd, check, env))
        lock.write_text(fresh)

    monkeypatch.setattr(dependency_manifest.subprocess, "run", run)

    result = dependency_manifest.refresh_isolated_manifest(tmp_path, inventory, check=True)
    assert result == (fresh != "original lock\n")
    assert lock.read_text() == "original lock\n"
    args = calls[0][0]
    assert args[args.index("--python") + 1] == inventory.python
    assert args[args.index("--project") + 1] == inventory.directory.as_posix()
    assert args[args.index("--refresh-package") + 1] == inventory.refresh_packages[0]
    assert calls[0][3]["NO_VCS_VERSION"] == inventory.environment["NO_VCS_VERSION"]


def test_isolated_check_restores_lock_after_resolver_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = _configured_inventory(tmp_path)
    directory = tmp_path / inventory.directory
    directory.mkdir(parents=True)
    (directory / "pyproject.toml").write_text(dependency_manifest.isolated_manifest_toml(inventory))
    lock = directory / "uv.lock"
    lock.write_text("original lock\n")

    def run(args, *, cwd, check, env):
        lock.write_text("partial lock\n")
        raise subprocess.CalledProcessError(2, args)

    monkeypatch.setattr(dependency_manifest.subprocess, "run", run)

    assert dependency_manifest.refresh_isolated_manifest(tmp_path, inventory, check=True) == 1
    assert lock.read_text() == "original lock\n"


def test_main_generates_configured_inventories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_repo(tmp_path)
    inventory = _configured_inventory(tmp_path)
    directory = tmp_path / inventory.directory
    monkeypatch.setattr(dependency_manifest, "lock_manifest", _fake_lock([]))
    monkeypatch.setattr(
        dependency_manifest.subprocess, "run",
        lambda args, *, cwd, check, env: (directory / "uv.lock").write_text("isolated lock\n"),
    )

    assert dependency_manifest.main([], root=tmp_path) == 0
    assert (directory / "pyproject.toml").read_text() == dependency_manifest.isolated_manifest_toml(inventory)
    assert (directory / "uv.lock").read_text() == "isolated lock\n"


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


def test_manifest_lock_pins_uv_and_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(
        dependency_manifest.subprocess,
        "run",
        lambda args, *, cwd, check: calls.append((args, cwd)),
    )

    dependency_manifest.lock_manifest(tmp_path)

    assert calls == [
        ([
            "uvx",
            f"uv@{dependency_manifest.UV_VERSION}",
            "--config-file",
            "uv.toml",
            "lock",
            "--upgrade",
            "--python",
            dependency_manifest.LOCK_PYTHON,
            "--project",
            dependency_manifest.MANIFEST_DIRECTORY,
        ], tmp_path),
    ]


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
    assert "uv.lock is missing" in capsys.readouterr().out


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
