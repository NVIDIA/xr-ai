# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline regression tests for SBOM scope, completeness, and provenance."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("generate_sbom", _ROOT / ".github/scripts/generate_sbom.py")
assert _spec is not None and _spec.loader is not None
sbom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sbom)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    manifest = tmp_path / "dependency-manifest"
    (manifest / "web-xr-build").mkdir(parents=True)
    (manifest / "android").mkdir()
    (manifest / "pyproject.toml").write_text('[project]\nname = "snapshot"\nversion = "0"\n')
    packages = [
        ("Some_Package.Name", "1", 'registry = "https://pypi.org/simple"'),
        ("local-worker", "2", 'editable = "../worker"'),
        ("snapshot", "0", 'virtual = "."'),
        ("git-lib", "3", 'git = "https://example.com/lib#abcdef"'),
    ]
    (manifest / "uv.lock").write_text(
        "\n".join(
            f'[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {{ {source} }}\n'
            for name, version, source in packages
        )
    )
    (manifest / "web-xr-build/package.json").write_text('{"name":"web","version":"1"}')
    (manifest / "web-xr-build/package-lock.json").write_text(
        json.dumps(
            {
                "packages": {
                    "": {"name": "web", "version": "1"},
                    "node_modules/@scope/dev": {"version": "2", "dev": True},
                }
            }
        )
    )
    (manifest / "android/gradle.lockfile").write_text("# lock\ngroup:app:1=runtime\nempty=\n")
    (manifest / "android/buildscript-gradle.lockfile").write_text("group:plugin:2=classpath\n")
    (tmp_path / "uv.toml").write_text(
        'exclude-newer = "2026-08-27T17:15:48Z"\n[exclude-newer-package]\npillow = "2026-10-17T00:00:00Z"\n'
    )
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    monkeypatch.setattr(sbom, "ROOT", tmp_path)
    monkeypatch.setattr(sbom, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["generate_sbom.py", "--output-dir", str(tmp_path / "output")])
    return tmp_path


def _scanner(monkeypatch, *, omit=None, unexpected=False):
    original_run = subprocess.run

    def run(args, **kwargs):
        if args[0] != "syft":
            return original_run(args, **kwargs)
        if args[1] == "version":
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"version": sbom.SYFT_VERSION}))
        if args[1] == "scan":
            staged = Path(args[2].removeprefix("dir:"))
            assert "include-dev-dependencies: true" in (staged / "syft.yaml").read_text()
            assert (staged / "android/buildscript/gradle.lockfile").read_text() == "group:plugin:2=classpath\n"
            artifacts = [
                {
                    "id": name,
                    "name": name,
                    "version": version,
                    "type": kind,
                    "metadata": {"pomProject": {"groupId": "group"}},
                }
                for name, version, kind in [
                    ("some-package-name", "1", "python"),
                    ("local-worker", "2", "python"),
                    ("snapshot", "0", "python"),
                    ("git-lib", "3", "python"),
                    ("web", "1", "npm"),
                    ("@scope/dev", "2", "npm"),
                    ("app", "1", "java-archive"),
                    ("plugin", "2", "java-archive"),
                ]
                if name != omit
            ]
            if unexpected:
                artifacts.append({"id": "extra", "name": "extra", "version": "9", "type": "npm"})
            (staged / "xr-ai.syft.json").write_text(
                json.dumps(
                    {
                        "artifacts": artifacts,
                        "artifactRelationships": [
                            {"parent": "local-worker", "child": "some-package-name"},
                            {"parent": "web", "child": "@scope/dev"},
                        ],
                    }
                )
            )
        elif args[1] == "convert":
            document = json.loads(Path(args[2]).read_text())
            assert {p["name"] for p in document["artifacts"]}.isdisjoint({"local-worker", "snapshot", "git-lib"})
            for output in args:
                if output.startswith("spdx-json="):
                    Path(output.split("=", 1)[1]).write_text(
                        json.dumps(
                            {
                                "packages": [
                                    {"downloadLocation": "file:../../sdk.tgz", "sourceInfo": "npm lock"},
                                    {"downloadLocation": "https://example.com/a.tgz"},
                                ]
                            }
                        )
                    )
                elif output.startswith("cyclonedx-json="):
                    Path(output.split("=", 1)[1]).write_text(json.dumps({"components": document["artifacts"]}))
        else:
            pytest.fail(f"Unexpected Syft invocation: {args}")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", run)


def test_expected_packages(repo):
    assert sbom._expected_packages() == {
        "python": {("some-package-name", "1")},
        "npm": {("web", "1"), ("@scope/dev", "2")},
        "java-archive": {("group:app", "1"), ("group:plugin", "2")},
    }


@pytest.mark.parametrize(
    "omit,unexpected",
    [
        ("some-package-name", False),
        ("web", False),
        ("@scope/dev", False),
        ("plugin", False),
        (None, True),
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_rejects_incomplete_inventory_without_publishing(repo, monkeypatch, omit, unexpected, existing):
    output = repo / "output"
    if existing:
        output.mkdir()
        (output / "xr-ai.spdx.json").write_text("previous report")
    _scanner(monkeypatch, omit=omit, unexpected=unexpected)
    with pytest.raises(SystemExit, match="Incomplete .* inventory"):
        sbom.main()
    assert {p.name: p.read_text() for p in output.glob("*")} == (
        {"xr-ai.spdx.json": "previous report"} if existing else {}
    )


def test_outputs_scope_locations_and_provenance(repo, monkeypatch):
    _scanner(monkeypatch)
    sbom.main()
    output = repo / "output"
    native = json.loads((output / "xr-ai.syft.json").read_text())
    assert native["artifactRelationships"] == [{"parent": "web", "child": "@scope/dev"}]
    spdx = json.loads((output / "xr-ai.spdx.json").read_text())
    assert spdx["packages"] == [
        {
            "downloadLocation": "NOASSERTION",
            "sourceInfo": "npm lock; lockfile local download reference: file:../../sdk.tgz",
        },
        {"downloadLocation": "https://example.com/a.tgz"},
    ]
    evidence = json.loads((output / "provenance.json").read_text())
    assert evidence["qualification"] == {
        "exclude-newer": "2026-08-27T17:15:48Z",
        "exclude-newer-package": {"pillow": "2026-10-17T00:00:00Z"},
    }
    assert evidence["packages"] == {"python": 1, "npm": 2, "java-archive": 2}
    assert "uv.toml" in evidence["inputs"]
    for name, digest in evidence["inputs"].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
    assert set(evidence["outputs"]) == {"xr-ai.syft.json", "xr-ai.spdx.json", "xr-ai.cdx.json"}
    for name, digest in evidence["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("change", ["unstaged", "staged", "untracked", "deleted", "qualification"])
def test_rejects_dirty_inputs(repo, monkeypatch, change):
    if change == "qualification":
        path = repo / "uv.toml"
    elif change == "untracked":
        path = repo / "dependency-manifest/android/new.lockfile"
    else:
        path = repo / "dependency-manifest/android/gradle.lockfile"
    if change == "deleted":
        path.unlink()
    else:
        path.write_text(path.read_text() + "\n" if path.exists() else "group:new:1=runtime\n")
    if change == "staged":
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
    _scanner(monkeypatch)
    with pytest.raises(SystemExit, match="inputs differ from HEAD"):
        sbom.main()
    assert not (repo / "output").exists()
