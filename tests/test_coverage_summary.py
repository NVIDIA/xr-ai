# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-project coverage summary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / ".github" / "scripts" / "coverage_summary.py"
_SPEC = importlib.util.spec_from_file_location("coverage_summary", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
coverage_summary = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = coverage_summary
_SPEC.loader.exec_module(coverage_summary)

_SOURCE = """\
def run(task):
    try:
        task()
    except ValueError:
        return 1
    except OSError:
        return 2
    return 0
"""


def test_handler_counts_marks_handlers_whose_body_executed() -> None:
    # Line 5 is the ValueError body, line 7 the OSError body.
    assert coverage_summary.handler_counts(_SOURCE, {1, 2, 3, 5, 8}) == (2, 1)
    assert coverage_summary.handler_counts(_SOURCE, {1, 2, 3, 8}) == (2, 0)


def test_area_of_names_nested_projects_only() -> None:
    assert coverage_summary.area_of("agent-sdk/xr-ai-hub/xr_ai_hub/_codec.py") == "agent-sdk/xr-ai-hub"
    assert coverage_summary.area_of("services/device-io-hub/device_io_hub/_x.py") == "services/device-io-hub"
    assert coverage_summary.area_of("agent-samples/demo/main.py") is None
    assert coverage_summary.area_of("tests/test_x.py") is None


def test_summarize_aggregates_files_and_skips_files_outside_the_repository(tmp_path: Path) -> None:
    module = tmp_path / "agent-sdk" / "pkg" / "pkg" / "mod.py"
    module.parent.mkdir(parents=True)
    module.write_text(_SOURCE, encoding="utf-8")
    outside = tmp_path.parent / "elsewhere.py"
    report = {
        "files": {
            str(module): {
                "executed_lines": [1, 2, 3, 5, 8],
                "summary": {
                    "num_statements": 8,
                    "covered_lines": 5,
                    "num_branches": 4,
                    "covered_branches": 3,
                },
            },
            str(outside): {"executed_lines": [], "summary": {"num_statements": 99, "covered_lines": 0}},
        }
    }

    areas = coverage_summary.summarize(report, tmp_path.resolve())

    assert list(areas) == ["agent-sdk/pkg"]
    stats = areas["agent-sdk/pkg"]
    assert (stats.statements, stats.covered_lines) == (8, 5)
    assert (stats.branches, stats.covered_branches) == (4, 3)
    assert (stats.handlers, stats.handlers_run) == (2, 1)


def test_render_formats_percentages_and_handles_empty_denominators() -> None:
    areas = {"agent-sdk/a": coverage_summary.AreaStats(10, 9, 4, 2, 2, 1), "utils/b": coverage_summary.AreaStats()}

    table = coverage_summary.render(areas)

    assert "| agent-sdk/a | 10 | 90.0% | 50.0% | 1 of 2 (50.0%) |" in table
    assert "| utils/b | 0 | n/a | n/a | 0 of 0 (n/a) |" in table
