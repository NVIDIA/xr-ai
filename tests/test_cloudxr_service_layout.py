# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository contracts for the relocated CloudXR runtime service."""
import runpy
import subprocess
import tomllib
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_PROJECT = _ROOT / "services" / "cloudxr-runtime"
_RENDER_ROOT = _ROOT / "agent-samples" / "xr-render-demo"


def test_cloudxr_service_preserves_its_public_contract() -> None:
    metadata = tomllib.loads((_PROJECT / "pyproject.toml").read_text())

    assert metadata["project"]["name"] == "cloudxr-runtime"
    assert metadata["project"]["scripts"]["cloudxr_runtime"] == (
        "cloudxr_runtime.__main__:run"
    )
    legacy_files = subprocess.run(
        ["git", "-C", str(_ROOT), "ls-files", "--", "cloudxr-runtime"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not legacy_files, f"tracked legacy CloudXR files remain: {legacy_files}"

    for source in metadata["tool"]["uv"]["sources"].values():
        assert (_PROJECT / source["path"]).resolve().is_dir()


def test_render_cloudxr_project_and_config_paths_resolve() -> None:
    namespace = runpy.run_path(str(_RENDER_ROOT / "main.py"))
    cloudxr = next(
        process
        for process in namespace["_build_processes"]()[0]
        if process.name == "cloudxr"
    )

    assert (_RENDER_ROOT / cloudxr.project).resolve() == _PROJECT
    assert (_RENDER_ROOT / cloudxr.config).resolve().is_file()
    assert cloudxr.command == "cloudxr_runtime"


def test_cloudxr_environment_file_matches_the_runtime_install_dir() -> None:
    runtime_config = yaml.safe_load(
        (_RENDER_ROOT / "yaml" / "cloudxr_runtime.yaml").read_text()
    )
    expected_env = (
        Path(runtime_config["cloudxr_install_dir"]).expanduser()
        / "run"
        / "cloudxr.env"
    )

    for config_path in (
        _RENDER_ROOT / "scene" / "scene_service.yaml",
        _RENDER_ROOT / "yaml" / "openxr_service.yaml",
    ):
        consumer_config = yaml.safe_load(config_path.read_text())
        assert Path(consumer_config["cloudxr_env_file"]).expanduser() == expected_env
