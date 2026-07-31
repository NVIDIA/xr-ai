# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository contracts for the final reusable-service layout."""
import ast
import subprocess
import sys
from pathlib import Path

import tomllib
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_SERVICES = {
    "cloudxr-runtime",
    "llama-nemotron-llm",
    "magpie-tts",
    "nemotron-omni-llm",
    "nemotron3-nano-llm",
    "openxr-service",
    "piper-tts",
    "stt-server",
    "video-memory-service",
    "vlm-server",
    "xr-media-hub",
}
_MODEL_SERVICES = {
    "llama-nemotron-llm": (
        "llama-nemotron-llm-server",
        "llama_nemotron_llm_server",
        8106,
    ),
    "magpie-tts": ("magpie-tts-server", "magpie_tts_server", 8104),
    "nemotron-omni-llm": (
        "nemotron-omni-llm-server",
        "nemotron_omni_llm_server",
        8108,
    ),
    "nemotron3-nano-llm": (
        "nemotron3-nano-llm-server",
        "nemotron3_nano_llm_server",
        8107,
    ),
    "piper-tts": ("piper-tts-server", "piper_tts_server", 8105),
    "stt-server": ("stt-server", "stt_server", 8103),
    "vlm-server": ("vlm-server", "vlm_server", 8100),
}
_LEGACY_PROJECTS = (
    "ai-services/vlm-server",
    "ai-services/stt-server",
    "ai-services/llm/llama_nemotron",
    "ai-services/llm/nemotron3_nano",
    "ai-services/llm/nemotron_omni",
    "ai-services/tts/piper",
    "ai-services/tts/magpie",
)
_LEGACY_ROOTS = {"ai-services", "cloudxr-runtime", "server-runtime"}
_ALLOWED_LEGACY_REFERENCES = {
    Path("docs/changelog.md"),
    Path("tests/test_service_layout.py"),
}
_HUB_PROJECT = _ROOT / "services" / "xr-media-hub"
_SAMPLE_WEB_CLIENTS = {
    "simple-vlm-example": _ROOT / "client-samples" / "web",
    "xr-render-demo": _ROOT / "client-samples" / "web-xr",
}


def _tracked_paths() -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return [Path(path.decode()) for path in output.split(b"\0") if path]


def test_reusable_services_are_direct_children() -> None:
    services = _ROOT / "services"

    assert _REQUIRED_SERVICES <= {
        path.name for path in services.iterdir() if path.is_dir()
    }
    # Ignored model caches can leave empty legacy directories in local checkouts.
    tracked_roots = {path.parts[0] for path in _tracked_paths()}
    assert _LEGACY_ROOTS.isdisjoint(tracked_roots)


def test_every_service_editable_source_path_resolves() -> None:
    for directory in _REQUIRED_SERVICES:
        project = _ROOT / "services" / directory
        metadata = tomllib.loads((project / "pyproject.toml").read_text())

        sources = metadata.get("tool", {}).get("uv", {}).get("sources", {})
        for source in sources.values():
            if path := source.get("path"):
                assert (project / path).resolve().exists(), (
                    f"{directory}: missing editable source {path}"
                )


def test_model_service_projects_preserve_their_public_contracts() -> None:
    for directory, (package, command, port) in _MODEL_SERVICES.items():
        project = _ROOT / "services" / directory
        metadata = tomllib.loads((project / "pyproject.toml").read_text())
        config = yaml.safe_load((project / f"{command}.yaml").read_text())

        assert metadata["project"]["name"] == package
        assert command in metadata["project"]["scripts"]
        assert config["port"] == port
        assert (project / config["model_cache"]).resolve() == _ROOT / "models"


def test_xr_media_hub_preserves_its_package_and_command() -> None:
    metadata = tomllib.loads((_HUB_PROJECT / "pyproject.toml").read_text())

    assert metadata["project"]["name"] == "xr-media-hub"
    assert metadata["project"]["scripts"] == {
        "xr_media_hub": "xr_media_hub.__main__:run"
    }
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "xr_media_hub"
    ]
    assert (_HUB_PROJECT / "xr_media_hub" / "__main__.py").is_file()


def test_sample_hub_projects_resolve() -> None:
    sample_projects: dict[str, str] = {}
    for main_path in sorted((_ROOT / "agent-samples").glob("*/main.py")):
        tree = ast.parse(main_path.read_text(), filename=str(main_path))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not isinstance(call.func, ast.Name) or call.func.id != "Process":
                continue
            if len(call.args) < 2:
                continue
            if (
                not isinstance(call.args[0], ast.Constant)
                or call.args[0].value != "hub"
            ):
                continue
            project = call.args[1]
            assert isinstance(project, ast.Constant) and isinstance(project.value, str)
            sample_projects[main_path.parent.name] = project.value

    assert {"simple-vlm-example", "xr-render-demo"} <= sample_projects.keys()
    for sample, project in sample_projects.items():
        sample_root = _ROOT / "agent-samples" / sample
        assert (sample_root / project).resolve() == _HUB_PROJECT


def test_hub_configuration_web_client_paths_resolve(monkeypatch) -> None:
    from xr_media_hub._config_loader import load_config

    reference_path = _HUB_PROJECT / "xr_media_hub.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["xr_media_hub", "--config", str(reference_path)],
    )
    reference = load_config()
    assert Path(reference.web_client_dir) == _ROOT / "client-samples" / "web"

    config_paths = sorted(
        (_ROOT / "agent-samples").glob("*/yaml/xr_media_hub.yaml")
    )
    assert {path.parents[1].name for path in config_paths} == _SAMPLE_WEB_CLIENTS.keys()
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text())
        sample = config_path.parents[1].name
        assert (config_path.parent / config["web_client_dir"]).resolve() == (
            _SAMPLE_WEB_CLIENTS[sample]
        )


def test_tracked_text_has_no_retired_service_paths() -> None:
    stale: list[tuple[str, str]] = []

    for relative in _tracked_paths():
        if relative in _ALLOWED_LEGACY_REFERENCES:
            continue
        path = _ROOT / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode()
        except UnicodeDecodeError:
            continue
        stale.extend(
            (str(relative), legacy)
            for legacy in _LEGACY_PROJECTS
            if legacy in text
        )
        if "server-runtime/" in text:
            stale.append((str(relative), "server-runtime/"))

    assert not stale, f"retired service paths remain: {stale}"


def test_manifests_and_ci_have_no_legacy_service_roots() -> None:
    stale: list[tuple[str, str]] = []
    for relative in _tracked_paths():
        if not (
            relative.name == "pyproject.toml"
            or relative.parts[0] == ".github"
            or relative in {Path("ruff.toml"), Path("sonar-project.properties")}
        ):
            continue
        text = (_ROOT / relative).read_text()
        stale.extend(
            (str(relative), legacy)
            for legacy in ("server-runtime/", "ai-services/", "../cloudxr-runtime")
            if legacy in text
        )

    assert not stale, f"legacy service roots remain in manifests or CI: {stale}"
