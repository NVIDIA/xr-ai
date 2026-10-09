# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Clef's production load path with real Hub cache resolution."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest
from _helpers_hf import block_network, cache_file, use_hub_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/clef-server"))

from clef_server._config import DEFAULT_MODEL, DEFAULT_REVISION, ServerConfig  # noqa: E402
from clef_server._service import ClefBackend  # noqa: E402

_LOADER = """
def load_release_model(directory, *, device, dtype):
    return {"directory": directory, "device": device, "dtype": dtype}, object()
def systemone(*args, **kwargs):
    return {}
def encode_record(*args, **kwargs):
    return None
"""
_FILES = {
    "joint_schema_model.py": _LOADER,
    "joint_head_config.json": "{}",
    "joint_head.safetensors": "head",
    "config.json": "{}",
    "processor_config.json": "{}",
    "tokenizer.json": "{}",
    "tokenizer_config.json": "{}",
    "chat_template.jinja": "{{ messages }}",
    "model.safetensors.index.json": json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors",
        "b": "model-00002-of-00002.safetensors",
    }}),
    "model-00001-of-00002.safetensors": "a",
    "model-00002-of-00002.safetensors": "b",
}


def _snapshot(cache: Path, *, omit: str | None = None, commit=DEFAULT_REVISION) -> Path:
    for name, content in _FILES.items():
        if name != omit:
            path = cache_file(cache, DEFAULT_MODEL, name, content, revision=None, commit=commit)
    return path.parent


@pytest.fixture
def backend(tmp_path, monkeypatch):
    use_hub_cache(monkeypatch, tmp_path / "unused-default-cache")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16="bf16", float16="fp16", float32="fp32",
    ))
    config = ServerConfig(
        DEFAULT_MODEL, DEFAULT_REVISION, None, tmp_path / "configured-cache",
        "127.0.0.1", 8120, "cpu", "float32", 4096, 1_048_576,
    )
    instance = ClefBackend(config)
    yield instance
    instance.executor.shutdown(wait=True)
    monkeypatch.delitem(sys.modules, "_clef_release_model", raising=False)


@pytest.mark.parametrize("offline", [False, True])
def test_complete_cache_loads_without_network(backend, monkeypatch, offline):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    snapshot = _snapshot(backend.config.model_cache)
    attempts = block_network(monkeypatch)

    backend.load()

    assert backend.model["directory"] == snapshot
    assert backend.model["device"] == "cpu"
    assert attempts == []


@pytest.mark.parametrize("omit", [None, *_FILES])
def test_missing_artifact_downloads_before_loading(backend, monkeypatch, tmp_path, omit):
    if omit is not None:
        _snapshot(backend.config.model_cache, omit=omit)
    downloaded = _snapshot(tmp_path / "downloaded")
    real_lookup = huggingface_hub.snapshot_download
    online = Mock(return_value=str(downloaded))

    def fetch(**kwargs):
        if kwargs.get("local_files_only"):
            return real_lookup(**kwargs)
        return online(**kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    attempts = block_network(monkeypatch)

    backend.load()

    online.assert_called_once_with(
        repo_id=DEFAULT_MODEL, revision=DEFAULT_REVISION, cache_dir=backend.config.model_cache,
    )
    assert backend.model["directory"] == downloaded
    assert attempts == []


@pytest.mark.parametrize("omit", [None, *_FILES])
def test_offline_miss_names_model_revision_cache_and_missing_file(backend, monkeypatch, omit):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    if omit is not None:
        _snapshot(backend.config.model_cache, omit=omit)
    attempts = block_network(monkeypatch)

    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE") as raised:
        backend.load()

    message = str(raised.value)
    assert DEFAULT_MODEL in message and DEFAULT_REVISION in message
    assert str(backend.config.model_cache) in message
    assert (omit or "model snapshot") in message
    assert backend.model is None
    assert attempts == []


def test_configured_revision_is_required(backend, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    _snapshot(backend.config.model_cache, commit="a" * 40)
    block_network(monkeypatch)

    with pytest.raises(RuntimeError, match=DEFAULT_REVISION):
        backend.load()


@pytest.mark.parametrize("content", ["{}", "[]", "{", '{"weight_map": {"a": 1}}'])
def test_invalid_weight_index_is_an_offline_miss(backend, monkeypatch, content):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    directory = _snapshot(backend.config.model_cache)
    (directory / "model.safetensors.index.json").write_text(content)
    block_network(monkeypatch)

    with pytest.raises(RuntimeError, match="Missing: model.safetensors.index.json"):
        backend.load()


def test_unsharded_backbone_is_a_cache_hit(backend, monkeypatch):
    directory = _snapshot(backend.config.model_cache, omit="model.safetensors.index.json")
    (directory / "model.safetensors").write_text("unsharded")
    attempts = block_network(monkeypatch)

    backend.load()

    assert backend.model["directory"] == directory
    assert attempts == []


def test_explicit_model_path_bypasses_hub(backend, monkeypatch, tmp_path):
    directory = tmp_path / "extracted"
    directory.mkdir()
    (directory / "joint_schema_model.py").write_text(_LOADER)
    backend.config = replace(backend.config, model_path=directory)
    lookup = Mock(side_effect=AssertionError("model_path must bypass the Hub"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lookup)

    backend.load()

    assert backend.model["directory"] == directory
    lookup.assert_not_called()


def test_incomplete_download_fails_before_model_loading(backend, monkeypatch, tmp_path):
    partial = _snapshot(tmp_path / "downloaded", omit="joint_head.safetensors")
    real_lookup = huggingface_hub.snapshot_download

    def fetch(**kwargs):
        return real_lookup(**kwargs) if kwargs.get("local_files_only") else str(partial)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    block_network(monkeypatch)

    with pytest.raises(RuntimeError, match="downloaded snapshot.*joint_head.safetensors"):
        backend.load()
    assert backend.model is None
