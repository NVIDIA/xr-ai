# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository contract for direct-child model services."""
import subprocess
from pathlib import Path

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
    Path("docs/changelog.md"),
    Path("tests/test_service_layout.py"),
}


def _tracked_paths() -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return [Path(path.decode()) for path in output.split(b"\0") if path]


def test_model_services_are_direct_children() -> None:
    services = _ROOT / "services"

    assert _MODEL_SERVICES.keys() <= {
        path.name for path in services.iterdir() if path.is_dir()
    }
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

        for source in metadata.get("tool", {}).get("uv", {}).get("sources", {}).values():
            if path := source.get("path"):
                assert (project / path).resolve().exists(), (
                    f"{directory}: missing editable source {path}"
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
        stale.extend(
            (str(relative), legacy)
            for legacy in _LEGACY_PROJECTS
            if legacy in text
        )

    assert not stale, f"retired model-service paths remain: {stale}"
