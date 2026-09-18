# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for application-owned workspace exclusions."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SPDX_PATH = _ROOT / ".github" / "scripts" / "check_spdx_headers.py"
_SPEC = importlib.util.spec_from_file_location("check_spdx_headers", _SPDX_PATH)
assert _SPEC is not None and _SPEC.loader is not None
spdx = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = spdx
_SPEC.loader.exec_module(spdx)


def test_spdx_discovery_excludes_top_level_applications(tmp_path: Path, monkeypatch) -> None:
    app_source = tmp_path / "apps" / "private-app" / "main.py"
    repository_source = tmp_path / "services" / "server.py"
    app_source.parent.mkdir(parents=True)
    repository_source.parent.mkdir(parents=True)
    app_source.write_text("print('private')\n")
    repository_source.write_text("print('repository')\n")
    monkeypatch.setattr(spdx, "_REPO_ROOT", tmp_path)

    discovered = spdx.discover(tmp_path)

    assert app_source not in discovered
    assert spdx.comment_style(app_source) is None
    assert spdx.main([str(app_source)]) == 0
    assert repository_source in discovered
    assert spdx.comment_style(repository_source) == "hash"


def test_repository_file_checks_exclude_applications() -> None:
    pre_commit = yaml.safe_load((_ROOT / ".pre-commit-config.yaml").read_text())
    ruff = tomllib.loads((_ROOT / "ruff.toml").read_text())
    lock_workflow = (_ROOT / ".github" / "workflows" / "lock-check.yml").read_text()

    assert pre_commit["exclude"] == "^apps/"
    assert "apps" in ruff["extend-exclude"]
    assert "-not -path './apps/*'" in lock_workflow
