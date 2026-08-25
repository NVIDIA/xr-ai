# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for documentation release selection policy."""
import re
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SELECTOR = _ROOT / ".github" / "scripts" / "select_latest_docs_release.py"
_LATEST_LINK_CHECKER = _ROOT / ".github" / "scripts" / "check_latest_docs_links.py"
_CONF = _ROOT / "docs" / "source" / "conf.py"
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_BASH_FENCE = re.compile(r"^```bash\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)\s]+)(?:\s+[^)]*)?\)")


def _visible_markdown(source: str) -> str:
    return _HTML_COMMENT.sub("", source)


def _section(source: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(.*?)(?=^##\s|\Z)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing {heading!r} section"
    return match.group(1)


def _sample_projects() -> list[tuple[Path, str]]:
    projects: list[tuple[Path, str]] = []
    for project_path in sorted((_ROOT / "agent-samples").glob("*/pyproject.toml")):
        sample_dir = project_path.parent
        if not (sample_dir / "main.py").is_file():
            continue
        metadata = tomllib.loads(project_path.read_text(encoding="utf-8"))
        scripts = metadata["project"].get("scripts", {})
        assert len(scripts) == 1, f"{project_path}: expected one top-level script"
        projects.append((sample_dir, next(iter(scripts))))
    return projects


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


def test_latest_docs_alias_contains_complete_rendered_version() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "docs.yaml").read_text()
    readme = (_ROOT / "README.md").read_text()

    assert 'cp -R "docs/_build/${latest_source}/." docs/_build/latest/' in workflow
    assert "check_latest_docs_links.py --quiet" in workflow
    assert '"docs/_build/${latest_source}"' in workflow
    assert workflow.count('- "**/README.md"') == 2
    assert not (_ROOT / "docs/source/_static/latest-redirect.html").exists()
    for page in (
        "getting_started/skills.html",
        "getting_started/quickstart.html",
        "getting_started/requirements.html",
        "overview/architecture.html",
    ):
        assert f"https://nvidia.github.io/xr-ai/latest/{page}" in readme


def test_latest_docs_link_checker_validates_pages_and_fragments(tmp_path) -> None:
    check_links = runpy.run_path(str(_LATEST_LINK_CHECKER))["check_latest_docs_links"]
    repository = tmp_path / "repository"
    rendered = tmp_path / "rendered"
    readme = repository / "README.md"
    page = rendered / "guide" / "start.html"
    readme.parent.mkdir()
    page.parent.mkdir(parents=True)
    readme.write_text(
        "[Start](https://nvidia.github.io/xr-ai/latest/guide/start.html#setup)\n"
        "[Missing](https://nvidia.github.io/xr-ai/latest/guide/missing.html)\n",
        encoding="utf-8",
    )
    page.write_text('<section id="setup"></section>', encoding="utf-8")

    errors = check_links(repository, rendered, (readme,))

    assert len(errors) == 1
    assert "rendered page does not exist" in errors[0]

    readme.write_text(
        "[Start](https://nvidia.github.io/xr-ai/latest/guide/start.html#other)\n",
        encoding="utf-8",
    )
    errors = check_links(repository, rendered, (readme,))

    assert len(errors) == 1
    assert "rendered fragment does not exist" in errors[0]


def test_getting_started_skill_routes_to_versioned_setup_docs() -> None:
    skill = (_ROOT / "skills" / "getting-started" / "SKILL.md").read_text()
    normalized = " ".join(skill.split())

    assert "root `README.md` Requirements" not in skill
    assert (
        "https://nvidia.github.io/xr-ai/<ref>/getting_started/requirements.html"
        in skill
    )
    assert (
        "https://nvidia.github.io/xr-ai/<ref>/getting_started/quickstart.html"
        in skill
    )
    assert (
        "Start `agent-samples/model-servers` and wait for it to report readiness "
        "before starting `agent-samples/simple-vlm-example`"
    ) in normalized


def test_sample_readmes_use_sample_directory_commands() -> None:
    for sample_dir, command in _sample_projects():
        directory = sample_dir.name
        readme = _visible_markdown((sample_dir / "README.md").read_text())
        run_section = _section(readme, "Run")
        configure_section = _section(readme, "Configure")
        bash = "\n".join(_BASH_FENCE.findall(run_section))

        assert f"Run all commands from `agent-samples/{directory}/`" in run_section
        assert "another terminal" not in readme
        assert not re.search(r"^\s*cd\s", bash, flags=re.MULTILINE)
        assert "uv run --directory" not in bash
        assert f"--project agent-samples/{directory}" not in bash
        assert re.search(rf"^uv run {re.escape(command)}(?:\s|$)", bash, re.MULTILINE)
        assert re.search(r"^uv run main\.py\s*$", bash, re.MULTILINE)
        assert "yaml/" in configure_section
        assert (
            "https://nvidia.github.io/xr-ai/latest/reference/configuration.html"
            in configure_section
        )
        if directory != "model-servers":
            assert "uv run --project ../model-servers model_servers" in bash
            assert "same terminal" in run_section
            assert "sample configuration guide" in configure_section


def test_readme_relative_links_resolve_and_docs_links_are_rendered() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "*README.md"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    for relative in tracked:
        readme_path = _ROOT / relative
        source = _visible_markdown(readme_path.read_text(encoding="utf-8"))
        for target in _MARKDOWN_LINK.findall(source):
            assert "docs/source/" not in target, (
                f"{relative}: link to rendered versioned documentation instead"
            )
            path = target.split("#", 1)[0]
            if not path or "://" in path or path.startswith("mailto:"):
                continue
            assert (readme_path.parent / path).resolve().exists(), (
                f"{relative}: unresolved link {target!r}"
            )


def test_service_root_commands_declare_their_working_directory() -> None:
    for readme_path in sorted((_ROOT / "services").glob("*/README.md")):
        source = _visible_markdown(readme_path.read_text(encoding="utf-8"))
        if "uv run --project services/" not in source:
            continue
        first_fence = source.index("```bash")
        assert "repository root" in source[:first_fence], readme_path


def test_reworded_headings_keep_compatibility_anchors() -> None:
    clients = (_ROOT / "docs/source/getting_started/clients.md").read_text()
    xr_render = (_ROOT / "docs/source/reference/xr-render-demo.md").read_text()

    for anchor in (
        "which-clients-exist",
        "network-telemetry",
        "the-connect-flow",
        "self-signed-certificate-trust",
        "web-basic-sample",
        "requirements",
        "build-and-run",
        "connect",
        "android-xr",
        "create-the-xcode-project",
        "adding-a-client-for-a-new-platform",
    ):
        assert f"({anchor})=" in clients
    assert "(worker-configuration)=" in xr_render
