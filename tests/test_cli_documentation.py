# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for static sample command discovery."""

from __future__ import annotations

from pathlib import Path
from runpy import run_path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CLI_REFERENCE = run_path(str(_ROOT / "docs" / "source" / "_cli_reference.py"))
_extract_arguments = _CLI_REFERENCE["_extract_arguments"]
load_cli_catalog = _CLI_REFERENCE["load_cli_catalog"]


def test_sample_command_catalog_matches_top_level_projects() -> None:
    commands = {command.program: command for command in load_cli_catalog(_ROOT)}

    assert set(commands) == {
        "lab_instrument_monitoring",
        "model_servers",
        "simple_vlm_example",
        "xr_render_demo",
    }
    assert [argument.flags for argument in commands["lab_instrument_monitoring"].arguments] == [
        ("--vlm-mode",),
        ("--expose-web-events",),
    ]
    assert [argument.flags for argument in commands["model_servers"].arguments] == [
        ("--stop",),
        ("--models",),
        ("--allow-anonymous",),
        ("--gpu-profile",),
    ]
    assert [argument.flags for argument in commands["simple_vlm_example"].arguments] == [
        ("--allow-anonymous",),
    ]
    assert commands["xr_render_demo"].arguments == ()


def test_catalog_builds_repository_root_invocation() -> None:
    commands = {command.program: command for command in load_cli_catalog(_ROOT)}

    assert commands["model_servers"].invocation == (
        "uv run --project agent-samples/model-servers model_servers "
        "[--stop] [--models NAME_OR_PATH] [--allow-anonymous] "
        "[--gpu-profile {dual_48G_ada,96G_blackwell,spark}]"
    )


def test_formatter_handles_positionals_and_nargs(tmp_path: Path) -> None:
    module = tmp_path / "main.py"
    module.write_text(
        '''
parser.add_argument("query", help="Required query.")
parser.add_argument("output", nargs="?", help="Optional output.")
parser.add_argument("items", nargs="*", help="Optional items.")
parser.add_argument("paths", nargs="+", metavar="PATH", help="One or more paths.")
parser.add_argument("pair", nargs=2, metavar="VALUE", help="A pair.")
parser.add_argument("--tag", nargs="+", choices=("a", "b"), help="Tags.")
''',
        encoding="utf-8",
    )

    arguments = {argument.flags[0]: argument for argument in _extract_arguments(module)}

    assert (arguments["query"].label, arguments["query"].usage) == ("query", "query")
    assert (arguments["output"].label, arguments["output"].usage) == (
        "output",
        "[output]",
    )
    assert (arguments["items"].label, arguments["items"].usage) == (
        "items",
        "[items ...]",
    )
    assert (arguments["paths"].label, arguments["paths"].usage) == (
        "PATH",
        "PATH [PATH ...]",
    )
    assert (arguments["pair"].label, arguments["pair"].usage) == (
        "VALUE",
        "VALUE VALUE",
    )
    assert (arguments["--tag"].label, arguments["--tag"].usage) == (
        "--tag {a,b} [{a,b} ...]",
        "[--tag {a,b} [{a,b} ...]]",
    )


def test_formatter_handles_all_no_value_actions(tmp_path: Path) -> None:
    module = tmp_path / "main.py"
    actions = (
        "append_const",
        "count",
        "help",
        "store_const",
        "store_false",
        "store_true",
        "version",
    )
    module.write_text(
        "\n".join(
            f'parser.add_argument("--{action}", action="{action}", help="{action}.")'
            for action in actions
        ),
        encoding="utf-8",
    )

    arguments = _extract_arguments(module)

    assert [argument.label for argument in arguments] == [
        f"--{action}" for action in actions
    ]
    assert [argument.usage for argument in arguments] == [
        f"[--{action}]" for action in actions
    ]


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ('"--mode", action="custom", help="Mode."', "unsupported action"),
        ('"--value", nargs="...", help="Value."', "unsupported nargs"),
        (
            '"--version", action="version", nargs="?", help="Version."',
            "no-value action cannot use",
        ),
    ],
)
def test_formatter_rejects_unrepresentable_declarations(
    tmp_path: Path,
    declaration: str,
    message: str,
) -> None:
    module = tmp_path / "main.py"
    module.write_text(f"parser.add_argument({declaration})\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _extract_arguments(module)
