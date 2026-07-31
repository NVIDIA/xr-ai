# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for documentation release selection policy."""
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SELECTOR = _ROOT / ".github" / "scripts" / "select_latest_docs_release.py"


def _select(*tags: str) -> str:
    result = subprocess.run(
        [sys.executable, str(_SELECTOR)],
        input="\n".join(tags),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_latest_release_uses_semver_precedence_not_input_order() -> None:
    assert _select("v2.0.0", "v10.0.0", "v1.99.0") == "v10.0.0"


def test_stable_release_is_preferred_over_newer_prerelease() -> None:
    assert _select("v2.0.0-rc.1", "v1.9.0", "v0.1.0") == "v1.9.0"


def test_highest_prerelease_is_used_until_a_stable_release_exists() -> None:
    assert _select("v1.0.0-beta.2", "v1.0.0-rc.1", "v1.0.0-beta.11") == (
        "v1.0.0-rc.1"
    )


def test_invalid_semver_tags_are_ignored() -> None:
    assert _select("release-2", "v1.0", "v1.0.0-01") == ""
