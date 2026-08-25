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
    assert workflow.count('- "CONTRIBUTING.md"') == 2
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

    readme.write_text(
        "[Start](https://nvidia.github.io/xr-ai/latest/guide/start.html#setup)\n",
        encoding="utf-8",
    )
    contributing = repository / "CONTRIBUTING.md"
    contributing.write_text(
        "[Missing](https://nvidia.github.io/xr-ai/latest/guide/contributing.html)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "add", "README.md", "CONTRIBUTING.md"],
        cwd=repository,
        check=True,
    )

    errors = check_links(repository, rendered)

    assert len(errors) == 1
    assert "CONTRIBUTING.md" in errors[0]
    assert "rendered page does not exist" in errors[0]


def test_latest_docs_link_checker_reads_reference_bare_and_html_links(tmp_path) -> None:
    check_links = runpy.run_path(str(_LATEST_LINK_CHECKER))["check_latest_docs_links"]
    repository = tmp_path / "repository"
    rendered = tmp_path / "rendered"
    readme = repository / "README.md"
    repository.mkdir()
    rendered.mkdir()
    readme.write_text(
        "[Reference][missing-reference]\n"
        "[missing-reference]: "
        "https://nvidia.github.io/xr-ai/latest/missing-reference.html\n"
        "https://nvidia.github.io/xr-ai/latest/missing-bare.html\n"
        '<a href="https://nvidia.github.io/xr-ai/latest/missing-html.html">'
        "HTML</a>\n",
        encoding="utf-8",
    )

    errors = check_links(repository, rendered, (readme,))

    assert len(errors) == 3
    assert all("rendered page does not exist" in error for error in errors)


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


def test_consolidated_readme_headings_keep_github_compatibility_anchors() -> None:
    # GitHub drops the punctuation in "NIM (STT/TTS)" while MyST replaces the
    # slash with a hyphen, so the README and published-page fragments differ.
    expected = {
        "README.md": """
            public-beta-notice what-is-xr-ai requirements architecture quickstart
            model-servers-shared-ai-services simple-vlm-example-vision-qa-over-voice--text
            step-1--start-the-server step-2--connect-a-client
            lab-instrument-monitoring-marker-associated-readings--foreground-voice
            xr-render-demo-voice-driven-sphere-in-cloudxr step-1--start-model-servers-once
            step-2--start-the-demo hub-only-standalone clients web android
            ios-and-visionos networking tests deeper-docs project-meta ios--visionos
        """,
        "agent-samples/lab-instrument-monitoring/README.md": """
            file-outputs foreground-routing-eval
        """,
        "agent-samples/simple-vlm-example/README.md": "relay-visibility",
        "agent-samples/tea-making-sample/README.md": """
            run-it foreground-behavior foreground-routing-eval file-outputs
            configuration safety
        """,
        "agent-samples/xr-render-demo/README.md": """
            file-map composition-chain how-to-extend add-a-subagent
            add-a-scene-tool--function-group add-an-eval-case edit-a-prompt running
        """,
        "agent-samples/xr-render-demo/eval/README.md": """
            live-drivers prompt-tuning-law prompt-tuning-loop writing-a-case
            dont-train-on-the-test-set what-the-harness-does-not-cover
        """,
        "agent-sdk/README.md": "removed-in-this-release",
        "agent-sdk/xr-ai-hub/README.md": """
            subscription-and-roster-contract frames return-path readiness
            shared-memory-and-codec-extensions
        """,
        "agent-sdk/xr-ai-models/README.md": """
            contract quickstart profile-contract deployment-profiles protocols
            remote-and-hosted-nim-endpoints riva-grpc-speech-nim-stttts tests
            remote--hosted-nim-endpoints remote-hosted-nim-endpoints
        """,
        "agent-sdk/xr-ai-tools/README.md": """
            native-tools-and-model-tool-calls typed-capability-services
            image-selection-and-vlm-query-tools marker-tracking
            magenta-polygon-image-editing
        """,
        "agent-sdk/xr-ai-voice/README.md": """
            usage multiple-voice-producers voice-tuning-and-data-echo
        """,
        "client-samples/android/README.md": """
            feature-set architecture setup requirements open-in-android-studio
            connecting-to-the-server permissions dependencies
        """,
        "client-samples/ios-visionos/README.md": """
            ai-sdk-sample repository-layout creating-the-xcode-project 1-new-project
            2-add-destinations 3-add-the-streamkit-package
            4-replace-the-generated-source-files 5-infoplist-entries
            6-visionos-passthrough-camera--device-only bundling-enterpriselicense
            7-build-and-run simulator-camera-feed
            trusting-the-hubs-self-signed-cert-one-time-per-device
            enable-full-trust-toggle-does-not-appear
            connection-fails-with-errsslbadcert---1202-after-the-cert-is-trusted
            tls-succeeds-but-the-room-rejects-the-token-with-401
            microphone-fails-to-start-with-a-timed-out-error
            orange-mic-indicator-stays-lit-after-stopping-audio
            mic--camera-go-dead-while-the-ui-still-says-on launching-xr-cloudxr
            two-parallel-transports
            server-prerequisite-change-nv_device_profile-to-auto-native
            cloudxrkit-spm-dependency apple-developer-program on-device-flow
            cert--trust-notes render-target quick-start-usage adding-a-custom-backend
            token-server-livekit
        """,
        "client-samples/native/README.md": """
            streamkit-for-native-c--livekit-backed-client running-the-tests
            constraints-in-the-current-native-backend what-streamkit-is-and-isnt
            what-streamkit-adds-on-top-of-livekit
            1-a-single-entry-point-with-decoupled-media
            2-a-typed-connectionstate-enum 3-typed-errors 4-the-agent-status-channel
            5-audioconfig-and-microphonemode 6-token-acquisition
            7-frame-injection-optional-for-external-video-sources
            the-streamingbackend-interface-you-need-to-implement
            implementing-livekitbackend-in-c what-you-get-for-free-once-the-backend-is-done
        """,
        "client-samples/web-xr-build/README.md": """
            web-xr-build--web-vendor-bundles usage bumping-the-cloudxr-sdk-version
            bumping-livekit-client files
        """,
        "services/embedding-server/README.md": """
            quickstart endpoints config-keys-embedding_serveryaml matryoshka-dimensions
            example-request choosing-the-vllm-runtime-pip-vs-docker
        """,
        "services/llama-nemotron-llm/README.md": """
            quickstart endpoints config-keys-llama_nemotron_llm_serveryaml
            tool-calling-native-llama-31-format reasoning-toggle--per-turn-via-system-prompt
            choosing-the-vllm-runtime-pip-vs-docker swap-models license
        """,
        "services/nemotron3-nano-llm/README.md": """
            quickstart endpoints config-keys-nemotron3_nano_llm_serveryaml
            tool-calling-native-qwen3-coder-format reasoning-mode-thinking
            sampling-recommendations-from-the-model-card hardware-notes swap-models
            notes license
        """,
        "skills/README.md": "available-skills setup",
        "tests/README.md": """
            xr-ai-integration-tests layout running gpu--docker--nvenc-tests
            test-taxonomy no-cross-talk-guarantee
        """,
    }

    anchor_pattern = re.compile(r'<a id="([^"]+)"></a>')
    for relative, required in expected.items():
        anchors = set(anchor_pattern.findall((_ROOT / relative).read_text()))
        assert set(required.split()) <= anchors, relative


def test_service_root_commands_declare_their_working_directory() -> None:
    for readme_path in sorted((_ROOT / "services").glob("*/README.md")):
        source = _visible_markdown(readme_path.read_text(encoding="utf-8"))
        if "uv run --project services/" not in source:
            continue
        first_fence = source.index("```bash")
        assert "repository root" in source[:first_fence], readme_path


def test_reworded_headings_keep_compatibility_anchors() -> None:
    clients = (_ROOT / "docs/source/getting_started/clients.md").read_text()
    hub = (_ROOT / "docs/source/reference/agent-sdk-hub.md").read_text()
    models = (_ROOT / "docs/source/reference/agent-sdk-models.md").read_text()
    tools = (_ROOT / "docs/source/reference/agent-sdk-tools.md").read_text()
    voice = (_ROOT / "docs/source/reference/agent-sdk-voice.md").read_text()
    xr_render = (_ROOT / "docs/source/reference/xr-render-demo.md").read_text()
    adding_cloudxr = (_ROOT / "docs/source/guides/adding-cloudxr.md").read_text()
    troubleshooting = (_ROOT / "docs/source/guides/troubleshooting.md").read_text()

    for anchor in (
        "which-clients-exist",
        "network-telemetry",
        "the-connect-flow",
        "self-signed-certificate-trust",
        "web-basic-sample",
        "android-xr",
        "create-the-xcode-project",
        "adding-a-client-for-a-new-platform",
    ):
        assert f"({anchor})=" in clients
    for page, anchors in (
        (
            hub,
            (
                "subscription-and-roster-contract",
                "return-path",
            ),
        ),
        (
            models,
            (
                "profile-contract",
                "deployment-profiles",
                "remote-and-hosted-nim-endpoints",
                "remote-hosted-nim-endpoints",
                "riva-grpc-speech-nim-stt-tts",
            ),
        ),
        (
            tools,
            (
                "native-tools-and-model-tool-calls",
                "typed-capability-services",
                "image-selection-and-vlm-query-tools",
                "magenta-polygon-image-editing",
            ),
        ),
        (
            voice,
            (
                "multiple-voice-producers",
                "voice-tuning-and-data-echo",
            ),
        ),
        (adding_cloudxr, ("add-cloudxr-runtime-yaml-to-the-sample-root",)),
        (
            troubleshooting,
            (
                "vllm-backend-docker-docker-run-fails-with-could-not-select-device-driver",
            ),
        ),
    ):
        for anchor in anchors:
            assert f"({anchor})=" in page
    for page, aliases in (
        (
            clients,
            (
                ("requirements", "clients-requirements"),
                ("build-and-run", "clients-build-and-run"),
                ("connect", "clients-connect"),
            ),
        ),
        (
            hub,
            (
                ("frames", "agent-sdk-hub-frames"),
                ("readiness", "agent-sdk-hub-readiness"),
            ),
        ),
        (
            models,
            (
                ("contract", "agent-sdk-models-contract"),
                ("quickstart", "agent-sdk-models-quickstart"),
                ("protocols", "agent-sdk-models-protocols"),
            ),
        ),
        (voice, (("usage", "agent-sdk-voice-usage"),)),
    ):
        for legacy, scoped in aliases:
            assert f'<a id="{legacy}"></a>' in page
            assert f"({scoped})=" in page
            assert f"({legacy})=" not in page
    assert '<a id="remote--hosted-nim-endpoints"></a>' in models
    assert "(worker-configuration)=" in xr_render
