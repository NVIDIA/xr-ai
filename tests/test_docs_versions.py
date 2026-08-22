# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for documentation release selection policy."""
import re
import runpy
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SELECTOR = _ROOT / ".github" / "scripts" / "select_latest_docs_release.py"
_CONF = _ROOT / "docs" / "source" / "conf.py"


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


def test_tag_whitelist_rejects_the_same_invalid_semver_tags() -> None:
    whitelist = runpy.run_path(str(_CONF))["smv_tag_whitelist"]

    assert re.fullmatch(whitelist, "v1.0.0")
    assert re.fullmatch(whitelist, "v1.0.0-rc.1")
    assert not re.fullmatch(whitelist, "v01.0.0")
    assert not re.fullmatch(whitelist, "v1.0.0-01")


def test_source_links_use_the_current_documentation_ref(monkeypatch) -> None:
    config = runpy.run_path(str(_CONF))
    for environment, ref in (
        ("XR_AI_DOCS_GITHUB_REF", "0123456789abcdef0123456789abcdef01234567"),
        ("SPHINX_MULTIVERSION_NAME", "v2.0.0"),
        (None, "main"),
    ):
        monkeypatch.delenv("SPHINX_MULTIVERSION_NAME", raising=False)
        monkeypatch.delenv("XR_AI_DOCS_GITHUB_REF", raising=False)
        if environment:
            monkeypatch.setenv(environment, ref)
        source = [
            "https://github.com/NVIDIA/xr-ai/blob/main/docs/example.md\n"
            "https://github.com/NVIDIA/xr-ai/tree/main/docs\n"
            "https://raw.githubusercontent.com/NVIDIA/xr-ai/main/skills/getting-started/SKILL.md"
        ]

        config["_rewrite_github_links"](None, "example", source)

        assert source == [
            f"https://github.com/NVIDIA/xr-ai/blob/{ref}/docs/example.md\n"
            f"https://github.com/NVIDIA/xr-ai/tree/{ref}/docs\n"
            f"https://raw.githubusercontent.com/NVIDIA/xr-ai/{ref}/skills/getting-started/SKILL.md"
        ]


def test_agent_prompt_is_owned_by_docs_snippet() -> None:
    snippet_path = _ROOT / "docs" / "source" / "_snippets" / "agent-setup-prompt.txt"
    snippet = snippet_path.read_text()
    readme = (_ROOT / "README.md").read_text()

    assert snippet not in readme
    assert "getting_started/skills" in readme
    for page in (
        _ROOT / "docs" / "source" / "index.md",
        _ROOT / "docs" / "source" / "getting_started" / "skills.md",
    ):
        assert "```{literalinclude} /_snippets/agent-setup-prompt.txt" in page.read_text()


def test_sample_readmes_use_sample_directory_commands() -> None:
    commands = {
        "lab-instrument-monitoring": "lab_instrument_monitoring",
        "model-servers": "model_servers",
        "simple-vlm-example": "simple_vlm_example",
        "tea-making-sample": "tea_making_sample",
        "xr-render-demo": "xr_render_demo",
    }

    for directory, command in commands.items():
        readme = (_ROOT / "agent-samples" / directory / "README.md").read_text()

        assert f"Run all commands from `agent-samples/{directory}/`" in readme
        assert f"--project agent-samples/{directory}" not in readme
        assert "another terminal" not in readme
        assert f"uv run {command}" in readme
        if directory != "model-servers":
            assert "uv run --project ../model-servers model_servers" in readme
            assert "same terminal" in readme
