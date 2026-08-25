# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository contracts for the final reusable-service layout."""
import ast
import importlib.util
import os.path
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_SERVICES = {
    "cloudxr-runtime",
    "embedding-server",
    "llama-nemotron-llm",
    "magpie-tts",
    "nemotron-omni-llm",
    "nemotron3-nano-llm",
    "openxr-service",
    "piper-tts",
    "rag-service",
    "stt-server",
    "video-memory-service",
    "vlm-server",
    "device-io-hub",
}
_MODEL_SERVICES = {
    "embedding-server": ("embedding-server", "embedding_server", 8109),
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
    "ai-services/embedding-server",
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
    Path("tests/test_service_layout.py"),
    Path("docs/source/reference/migrations.md"),
}
_HUB_PROJECT = _ROOT / "services" / "device-io-hub"
_SAMPLE_WEB_CLIENTS = {
    "lab-instrument-monitoring": _ROOT / "client-samples" / "web",
    "simple-vlm-example": _ROOT / "client-samples" / "web",
    "tea-making-sample": _ROOT / "client-samples" / "web",
    "xr-render-demo": _ROOT / "client-samples" / "web-xr",
}
_RETIRED_AGENT_SDK_PATHS = (
    "agent-sdk/xr-ai-hub-client",
    "agent-sdk/xr-ai-agent-runtime",
    "agent-sdk/xr-ai-pipecat",
    "agent-sdk/xr-ai-hub/xr_ai_agent",
    "agent-sdk/xr-ai-models/xr_ai_models/config.py",
    "agent-sdk/xr-ai-models/xr_ai_models/factory.py",
    "agent-sdk/xr-ai-models/xr_ai_models/openai_compat.py",
    "agent-sdk/xr-ai-models/xr_ai_models/protocols.py",
)
_RETIRED_AGENT_SDK_MODULES = (
    "xr_ai_agent",
    "xr_ai_pipecat",
    "xr_ai_models.config",
    "xr_ai_models.factory",
    "xr_ai_models.openai_compat",
    "xr_ai_models.protocols",
)
_RETIRED_AGENT_SDK_REFERENCES = (
    "agent-sdk/xr-ai-hub-client/",
    "agent-sdk/xr-ai-agent-runtime/",
)
def _tracked_paths() -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "-C", str(_ROOT), "ls-files", "-z"],
            check=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("service-layout tracked-path checks require a git checkout")
    return [Path(path.decode()) for path in output.split(b"\0") if path]


def _load_module(name: str, relative: str):
    path = _ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_cache_default(path: Path) -> str:
    defaults: list[str] = []

    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "resolve_model_cache":
            defaults.extend(
                ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg == "default"
            )
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "model_cache"
        ):
            defaults.append(ast.literal_eval(node.args[1]))

    assert len(defaults) == 1, f"expected one model_cache default in {path}"
    return defaults[0]


def test_retired_agent_sdk_surfaces_are_absent() -> None:
    for relative in _RETIRED_AGENT_SDK_PATHS:
        assert not (_ROOT / relative).exists(), relative


def test_retired_agent_sdk_modules_are_not_importable() -> None:
    for module in _RETIRED_AGENT_SDK_MODULES:
        assert importlib.util.find_spec(module) is None, module


def test_reusable_services_are_direct_children() -> None:
    services = _ROOT / "services"
    tracked_paths = _tracked_paths()

    discovered = {
        project.name
        for project in services.iterdir()
        if project.is_dir()
        and any(
            "model_cache" in (yaml.safe_load(config.read_text()) or {})
            for config in project.glob("*.yaml")
        )
    }
    assert _MODEL_SERVICES.keys() == discovered
    assert _REQUIRED_SERVICES <= {
        path.name for path in services.iterdir() if path.is_dir()
    }

    services_path = Path("services")
    nested_projects = sorted(
        path.relative_to(services_path)
        for path in tracked_paths
        if path.name == "pyproject.toml"
        and path.is_relative_to(services_path)
        and path.parent.parent != services_path
    )
    assert not nested_projects, f"nested service projects remain: {nested_projects}"

    legacy_projects = tuple(Path(project) for project in _LEGACY_PROJECTS)
    stale = [
        path
        for path in tracked_paths
        if any(path.is_relative_to(project) for project in legacy_projects)
    ]
    assert not stale, f"tracked legacy model-service files remain: {stale}"

    # Ignored model caches can leave empty legacy directories in local checkouts.
    tracked_roots = {path.parts[0] for path in tracked_paths}
    assert _LEGACY_ROOTS.isdisjoint(tracked_roots)


def test_xr_render_checks_the_web_xr_vendor_bundle() -> None:
    source = (_ROOT / "agent-samples/xr-render-demo/main.py").read_text()

    assert "client-samples/web-xr/vendor" in source
    assert "client-samples/web/vendor" not in source
    assert 'vendor_dir / "cloudxr-sdk.esm.mjs"' in source
    assert 'vendor_dir / "livekit-client.esm.mjs"' in source


def test_xr_render_repairs_an_incomplete_web_xr_vendor_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _load_module(
        "service_layout_render_vendor_bundle",
        "agent-samples/xr-render-demo/main.py",
    )
    sample_root = tmp_path / "agent-samples" / "xr-render-demo"
    sample_root.mkdir(parents=True)
    vendor_dir = (sample_root / "../../client-samples/web-xr/vendor").resolve()
    vendor_dir.mkdir(parents=True)
    (vendor_dir / "cloudxr-sdk.esm.mjs").write_text("cloudxr")
    build_script = (
        sample_root / "../../client-samples/web-xr-build/build.sh"
    ).resolve()
    build_script.parent.mkdir(parents=True)
    build_script.write_text("#!/bin/sh\n")
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: str) -> subprocess.CompletedProcess:
        calls.append(command)
        (vendor_dir / "livekit-client.esm.mjs").write_text("livekit")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sample, "_BASE", sample_root)
    monkeypatch.setattr(sample.shutil, "which", lambda _command: "/usr/bin/npm")
    monkeypatch.setattr(sample.subprocess, "run", fake_run)

    sample._ensure_web_vendor()
    assert calls == [[str(build_script)]]

    calls.clear()
    sample._ensure_web_vendor()
    assert calls == []


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
        assert Path(os.path.normpath(project / config["model_cache"])) == _ROOT / "models", (
            f"{directory}: model_cache must target the repo models directory"
        )
        default = _model_cache_default(project / command / "__main__.py")
        assert Path(os.path.normpath(project / default)) == _ROOT / "models", (
            f"{directory}: default model cache must target the repo models directory"
        )


def test_device_io_hub_preserves_its_package_and_command() -> None:
    metadata = tomllib.loads((_HUB_PROJECT / "pyproject.toml").read_text())

    assert metadata["project"]["name"] == "device-io-hub"
    assert metadata["project"]["scripts"] == {
        "device_io_hub": "device_io_hub.__main__:run"
    }
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "device_io_hub"
    ]
    assert (_HUB_PROJECT / "device_io_hub" / "__main__.py").is_file()


def test_sample_process_projects_resolve(monkeypatch) -> None:
    model_servers = _load_module(
        "service_layout_model_servers",
        "agent-samples/model-servers/main.py",
    )
    simple_vlm = _load_module(
        "service_layout_simple_vlm",
        "agent-samples/simple-vlm-example/main.py",
    )
    render_demo = _load_module(
        "service_layout_render_demo",
        "agent-samples/xr-render-demo/main.py",
    )
    monkeypatch.setattr(model_servers, "detect_gpu_config", lambda: "spark")

    declarations = [
        (
            _ROOT / "agent-samples/model-servers",
            model_servers._build_processes("default")[0]
            + model_servers._build_processes("vlm_llm_nim")[0],
        ),
        (
            _ROOT / "agent-samples/simple-vlm-example",
            simple_vlm.PROCESSES,
        ),
        (
            _ROOT / "agent-samples/xr-render-demo",
            render_demo._build_processes(),
        ),
    ]

    for sample_root, processes in declarations:
        for process in processes:
            project = (sample_root / process.project).resolve()
            assert (project / "pyproject.toml").is_file(), (
                f"{process.name}: process project does not resolve: {project}"
            )


def test_simple_vlm_reuses_every_model_process() -> None:
    sample = _load_module(
        "service_layout_simple_vlm_reuse",
        "agent-samples/simple-vlm-example/main.py",
    )

    assert {
        process.name: process.launch_mode
        for process in sample.PROCESSES
        if process.name in {"stt", "vlm", "tts"}
    } == {"stt": "reuse", "vlm": "reuse", "tts": "reuse"}


def test_render_demo_reuses_every_model_process() -> None:
    sample = _load_module(
        "service_layout_render_reuse",
        "agent-samples/xr-render-demo/main.py",
    )

    assert {
        process.name: process.launch_mode
        for process in sample._build_processes()
        if process.name in {"stt", "omni", "vlm", "tts"}
    } == {"stt": "reuse", "omni": "reuse", "vlm": "reuse", "tts": "reuse"}


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
    from device_io_hub._config_loader import load_config

    reference_path = _HUB_PROJECT / "device_io_hub.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["device_io_hub", "--config", str(reference_path)],
    )
    reference = load_config()
    assert Path(reference.web_client_dir) == _ROOT / "client-samples" / "web"

    config_paths = sorted(
        (_ROOT / "agent-samples").glob("*/yaml/device_io_hub.yaml")
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
            (f"{relative}:{line_number}", legacy)
            for line_number, line in enumerate(text.splitlines(), start=1)
            for legacy in _LEGACY_PROJECTS
            if legacy in line
        )
        stale.extend(
            (f"{relative}:{line_number}", "server-runtime/")
            for line_number, line in enumerate(text.splitlines(), start=1)
            if "server-runtime/" in line
        )
        stale.extend(
            (f"{relative}:{line_number}", legacy)
            for line_number, line in enumerate(text.splitlines(), start=1)
            for legacy in _RETIRED_AGENT_SDK_REFERENCES
            if legacy in line
        )
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
