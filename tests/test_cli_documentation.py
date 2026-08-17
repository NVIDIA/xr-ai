# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for static sample command discovery."""

from __future__ import annotations

from pathlib import Path
from runpy import run_path

_ROOT = Path(__file__).resolve().parents[1]
_CLI_REFERENCE = run_path(str(_ROOT / "docs" / "source" / "_cli_reference.py"))
load_cli_catalog = _CLI_REFERENCE["load_cli_catalog"]


def test_sample_command_catalog_matches_top_level_projects() -> None:
    commands = {command.program: command for command in load_cli_catalog(_ROOT)}

    assert set(commands) == {"model_servers", "simple_vlm_example", "xr_render_demo"}
    assert [argument.flags for argument in commands["model_servers"].arguments] == [
        ("--stop",),
        ("--allow-anonymous",),
    ]
    assert [argument.flags for argument in commands["simple_vlm_example"].arguments] == [
        ("--allow-anonymous",),
    ]
    assert commands["xr_render_demo"].arguments == ()


def test_catalog_builds_repository_root_invocation() -> None:
    commands = {command.program: command for command in load_cli_catalog(_ROOT)}

    assert commands["model_servers"].invocation == (
        "uv run --project agent-samples/model-servers model_servers [--stop] [--allow-anonymous]"
    )
