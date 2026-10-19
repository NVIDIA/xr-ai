# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ratcheted Ruff rule baseline."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / ".github" / "scripts" / "ruff_baseline.py"
_SPEC = importlib.util.spec_from_file_location("ruff_baseline", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
ruff_baseline = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ruff_baseline
_SPEC.loader.exec_module(ruff_baseline)


def _pinned_ruff_on_path() -> bool:
    ruff = shutil.which("ruff")
    if ruff is None:
        return False
    version = subprocess.run([ruff, "--version"], capture_output=True, text=True, check=True).stdout
    return version.split()[-1] == ruff_baseline.RUFF_VERSION


# These tests run the real linter. The lint workflow runs them where the pinned
# Ruff is installed; elsewhere they skip rather than download it.
needs_ruff = pytest.mark.skipif(
    not _pinned_ruff_on_path(),
    reason=f"requires ruff {ruff_baseline.RUFF_VERSION} on PATH; lint.yml runs these tests",
)

_BLIND_EXCEPT = "try:\n    pass\nexcept Exception:\n    pass\n"


def test_render_and_parse_round_trip() -> None:
    counts = {("pkg/a.py", "BLE001"): 3, ("pkg/a.py", "S110"): 1, ("pkg/b.py", "B904"): 2}

    assert ruff_baseline.parse(ruff_baseline.render(counts)) == counts


def test_compare_reports_growth_and_stale_entries() -> None:
    recorded = {("a.py", "BLE001"): 2, ("b.py", "BLE001"): 1}
    actual = {("a.py", "BLE001"): 3, ("c.py", "S110"): 1}

    problems = ruff_baseline.compare(actual, recorded)

    assert any("a.py: BLE001 has 3" in p and "fix the new code" in p for p in problems)
    assert any("b.py: BLE001 has 0" in p and "run --write" in p for p in problems)
    assert any("c.py: S110 has 1" in p and "fix the new code" in p for p in problems)
    assert ruff_baseline.compare(recorded, recorded) == []


def test_area_groups_nested_projects() -> None:
    assert ruff_baseline._area("agent-sdk/xr-ai-voice/xr_ai_voice/_runtime.py") == "agent-sdk/xr-ai-voice"
    assert ruff_baseline._area("tests/test_x.py") == "tests"
    assert ruff_baseline._area("services/rag.py") == "services"


@needs_ruff
def test_scan_counts_violations_and_honors_noqa(tmp_path: Path) -> None:
    (tmp_path / "ruff.toml").write_text('extend-exclude = ["skipped"]\n', encoding="utf-8")
    (tmp_path / "bad.py").write_text(_BLIND_EXCEPT, encoding="utf-8")
    (tmp_path / "ok.py").write_text(
        "try:\n    pass\nexcept Exception:  # noqa: BLE001, S110\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "skipped").mkdir()
    (tmp_path / "skipped" / "bad.py").write_text(_BLIND_EXCEPT, encoding="utf-8")

    counts = ruff_baseline.scan(tmp_path)

    assert counts == {("bad.py", "BLE001"): 1, ("bad.py", "S110"): 1}


@needs_ruff
def test_repository_baseline_matches_the_code() -> None:
    recorded = ruff_baseline.parse((_ROOT / ruff_baseline.BASELINE_NAME).read_text(encoding="utf-8"))

    assert ruff_baseline.compare(ruff_baseline.scan(_ROOT), recorded) == []
