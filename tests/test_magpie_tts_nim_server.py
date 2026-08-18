# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launcher-owned Magpie TTS NIM container configuration coverage."""

from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_PROJECT = _ROOT / "services" / "magpie-tts-nim"
_MODULE = _PROJECT / "magpie_tts_nim_server" / "__main__.py"
_SPEC = importlib.util.spec_from_file_location("magpie_tts_nim_server_main", _MODULE)
assert _SPEC and _SPEC.loader
_server = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_server)


def test_magpie_nim_docker_command_uses_pinned_streaming_image(tmp_path) -> None:
    config_path = (
        _ROOT
        / "agent-samples"
        / "simple-vlm-example"
        / "yaml"
        / "magpie_tts_nim_server.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    cache = tmp_path / "nim-cache"
    export = cache / "export"
    argv = _server._build_docker_argv(config, cache, export)

    assert argv[-1] == "nvcr.io/nim/nvidia/magpie-tts-multilingual:1.9.0"
    assert argv[argv.index("--gpus") + 1] == "device=0"
    assert argv[argv.index("-p") + 1] == "127.0.0.1:9000:9000"
    assert "--user" not in argv
    assert "NGC_API_KEY" in argv
    assert f"{export}:/opt/nim/export" in argv
    assert "NIM_EXPORT_PATH=/opt/nim/export" in argv
    assert "NIM_DISABLE_MODEL_DOWNLOAD=true" in argv
    assert "name=magpie-tts-multilingual,batch_size=8" in " ".join(argv)
    assert not any("nvapi-" in arg for arg in argv)


def test_magpie_nim_export_command_builds_without_disabling_download(tmp_path) -> None:
    config = {"image": "magpie:test"}
    argv = _server._build_docker_argv(
        config,
        tmp_path / "cache",
        tmp_path / "export",
        export_only=True,
    )

    assert argv[-1] == "magpie:test"
    assert "NIM_EXPORT_PATH=/opt/nim/export" in argv
    assert "NIM_DISABLE_MODEL_DOWNLOAD=true" not in argv


def test_magpie_nim_export_requires_complete_model_store(tmp_path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    archive = export / "magpie.tar.gz"

    with tarfile.open(archive, "w") as model_store:
        for name in ("model/encoder.plan", "model/config.pbtxt"):
            payload = b"model-data"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            model_store.addfile(info, io.BytesIO(payload))

    assert _server._export_is_ready(export)

    archive.write_bytes(b"incomplete tar")
    assert not _server._export_is_ready(export)
