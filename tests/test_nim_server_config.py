# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config parsing / dispatch coverage for the nim-server command.

The package is not an editable dependency of the test project, so it is
imported off the source tree directly; ``serve_nim`` is stubbed, so nothing
touches docker.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "services" / "nim-server"))

import nim_server.__main__ as nim_main  # noqa: E402

_IMAGE = "nvcr.io/nim/meta/llama-3.1-8b-instruct:latest"


@pytest.fixture
def launch(monkeypatch, tmp_path):
    """Run nim_main.run() with *cfg*, capturing the serve_nim kwargs."""
    captured: dict = {}
    monkeypatch.setattr(nim_main, "setup_logging", lambda name: None)
    monkeypatch.setattr(nim_main, "serve_nim", lambda **kw: captured.update(kw))

    def _run(cfg: dict, yaml_dir: Path | None = None, ready_file=None) -> dict:
        monkeypatch.setattr(
            nim_main, "load_config",
            lambda: (cfg, yaml_dir or tmp_path, ready_file),
        )
        nim_main.run()
        return captured

    return _run


def test_missing_image_exits(launch):
    with pytest.raises(SystemExit):
        launch({"http_port": 8106})


def test_missing_http_port_exits(launch):
    with pytest.raises(SystemExit):
        launch({"image": _IMAGE})


def test_default_container_name_sanitized_from_image(launch):
    kw = launch({"image": "nvcr.io/nim/org/some+model name:1.0", "http_port": 8106})
    assert kw["container_name"] == "xr-ai-nim-some-model-name"


def test_explicit_container_name_wins_and_prefixes_log(launch):
    kw = launch({"image": _IMAGE, "http_port": 8106,
                 "container_name": "xr-ai-nim-custom"})
    assert kw["container_name"] == "xr-ai-nim-custom"
    assert kw["log_prefix"] == "custom"


def test_relative_nim_cache_resolves_against_yaml_dir(launch, tmp_path):
    yaml_dir = tmp_path / "sample" / "yaml"
    yaml_dir.mkdir(parents=True)
    kw = launch(
        {"image": _IMAGE, "http_port": 8106, "nim_cache": "../cache/nim"},
        yaml_dir=yaml_dir,
    )
    assert kw["nim_cache"] == (tmp_path / "sample" / "cache" / "nim").resolve()


def test_absolute_nim_cache_used_verbatim(launch, tmp_path):
    kw = launch({"image": _IMAGE, "http_port": 8106,
                 "nim_cache": str(tmp_path / "abs")})
    assert kw["nim_cache"] == tmp_path / "abs"


def test_grpc_port_absent_is_none(launch):
    kw = launch({"image": _IMAGE, "http_port": 8106})
    assert kw["grpc_port"] is None
    assert kw["http_port"] == 8106


def test_grpc_port_parsed_as_int(launch):
    kw = launch({"image": _IMAGE, "http_port": 9010, "grpc_port": "50052"})
    assert kw["grpc_port"] == 50052


def test_env_map_forwarded_as_strings(launch):
    kw = launch({"image": _IMAGE, "http_port": 8106,
                 "env": {"NIM_LOG_LEVEL": "DEBUG", "NIM_MAX_BATCH": 4}})
    assert kw["extra_env"] == {"NIM_LOG_LEVEL": "DEBUG", "NIM_MAX_BATCH": "4"}


def test_ready_file_and_cuda_devices_forwarded(launch, tmp_path):
    ready = tmp_path / "ready"
    kw = launch({"image": _IMAGE, "http_port": 8106, "cuda_visible_devices": 0},
                ready_file=ready)
    assert kw["ready_file"] == ready
    assert kw["cuda_visible_devices"] == "0"
