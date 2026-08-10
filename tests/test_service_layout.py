# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository contract for direct-child model services."""
import ast
import importlib.util
import subprocess
from pathlib import Path

import pytest
import tomllib
import yaml

_ROOT = Path(__file__).resolve().parents[1]
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
_ALLOWED_LEGACY_REFERENCES = {
    Path("tests/test_service_layout.py"),
}
_ALLOWED_LEGACY_LINES = {
    Path("docs/changelog.md"): {
        "the 8B held is freed. The standalone `ai-services/llm/llama_nemotron` server",
    },
}


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


def test_model_services_are_direct_children() -> None:
    services = _ROOT / "services"

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
    nested_projects = sorted(
        path.relative_to(services)
        for path in services.rglob("pyproject.toml")
        if path.parent.parent != services
    )
    assert not nested_projects, f"nested service projects remain: {nested_projects}"

    legacy_projects = tuple(Path(project) for project in _LEGACY_PROJECTS)
    stale = [
        path
        for path in _tracked_paths()
        if any(path.is_relative_to(project) for project in legacy_projects)
    ]
    assert not stale, f"tracked legacy model-service files remain: {stale}"


def test_model_service_projects_preserve_their_public_contracts() -> None:
    for directory, (package, command, port) in _MODEL_SERVICES.items():
        project = _ROOT / "services" / directory
        metadata = tomllib.loads((project / "pyproject.toml").read_text())
        config = yaml.safe_load((project / f"{command}.yaml").read_text())

        assert metadata["project"]["name"] == package
        assert command in metadata["project"]["scripts"]
        assert config["port"] == port
        assert (project / config["model_cache"]).resolve() == _ROOT / "models"
        default = _model_cache_default(project / command / "__main__.py")
        assert (project / default).resolve() == _ROOT / "models"

        for source in metadata.get("tool", {}).get("uv", {}).get("sources", {}).values():
            if path := source.get("path"):
                assert (project / path).resolve().exists(), (
                    f"{directory}: missing editable source {path}"
                )


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
            model_servers._build_processes("vlm-llm")
            + model_servers._build_processes("omni"),
        ),
        (
            _ROOT / "agent-samples/simple-vlm-example",
            simple_vlm._MODEL_PROCESSES.values(),
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


def test_tracked_text_has_no_retired_model_service_paths() -> None:
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
        allowed_lines = _ALLOWED_LEGACY_LINES.get(relative, set())
        stale.extend(
            (f"{relative}:{line_number}", legacy)
            for line_number, line in enumerate(text.splitlines(), start=1)
            for legacy in _LEGACY_PROJECTS
            if legacy in line and line not in allowed_lines
        )

    assert not stale, f"retired model-service paths remain: {stale}"
