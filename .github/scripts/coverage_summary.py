# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Summarize test coverage per project, including exception-handler coverage.

Reads the JSON report written by ``coverage json`` and prints a Markdown table
with line, branch, and exception-handler coverage for each project under
``agent-sdk/``, ``utils/``, and ``services/``. When ``GITHUB_STEP_SUMMARY`` is
set the table is also appended to the job summary. The report is informational
and always exits 0.

Line coverage alone overstates how well failure handling is tested: an
``except`` body runs only when a test provokes the failure. The handler columns
count the ``except`` clauses in each file and how many of them ran at least
once, which is the number that matters before changing exception handling.

Usage (from ``tests/``, after ``coverage combine``)::

    coverage json && python3 ../.github/scripts/coverage_summary.py coverage.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]

# Directories whose first two path components name the owning project.
_NESTED_AREAS = ("agent-sdk", "utils", "services")


@dataclass
class AreaStats:
    """Coverage totals for one project."""

    statements: int = 0
    covered_lines: int = 0
    branches: int = 0
    covered_branches: int = 0
    handlers: int = 0
    handlers_run: int = 0


def handler_counts(source: str, executed_lines: Collection[int]) -> tuple[int, int]:
    """Return ``(handlers, handlers_that_ran)`` for the ``except`` clauses in ``source``.

    A handler counts as run when any statement directly in its body executed.
    """
    total = ran = 0
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ExceptHandler):
            total += 1
            if any(statement.lineno in executed_lines for statement in node.body):
                ran += 1
    return total, ran


def area_of(relative_path: str) -> str | None:
    """Name the project that owns ``relative_path``, or ``None`` outside the reported areas."""
    parts = relative_path.split("/")
    if parts[0] in _NESTED_AREAS and len(parts) > 2:
        return "/".join(parts[:2])
    return None


def summarize(report: Mapping[str, Any], root: Path) -> dict[str, AreaStats]:
    """Aggregate a ``coverage json`` report into per-project totals."""
    areas: dict[str, AreaStats] = defaultdict(AreaStats)
    for name, data in report["files"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        area = area_of(relative)
        if area is None:
            continue
        stats = areas[area]
        summary = data["summary"]
        stats.statements += summary["num_statements"]
        stats.covered_lines += summary["covered_lines"]
        stats.branches += summary.get("num_branches", 0)
        stats.covered_branches += summary.get("covered_branches", 0)
        handlers, ran = handler_counts(path.read_text(encoding="utf-8"), set(data["executed_lines"]))
        stats.handlers += handlers
        stats.handlers_run += ran
    return dict(areas)


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def render(areas: Mapping[str, AreaStats]) -> str:
    """Render the per-project Markdown table."""
    lines = [
        "### Test coverage by project",
        "",
        "| Project | Statements | Lines | Branches | Exception handlers run |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(areas):
        s = areas[name]
        handlers = f"{s.handlers_run} of {s.handlers} ({_percent(s.handlers_run, s.handlers)})"
        lines.append(
            f"| {name} | {s.statements} | {_percent(s.covered_lines, s.statements)} "
            f"| {_percent(s.covered_branches, s.branches)} | {handlers} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path, help="JSON report written by `coverage json`")
    args = parser.parse_args(argv)

    table = render(summarize(json.loads(args.report.read_text(encoding="utf-8")), _ROOT))
    print(table)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
