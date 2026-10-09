# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Clef's production load path with real Hub cache resolution."""
from __future__ import annotations

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


def _snapshot(cache: Path, *, commit=DEFAULT_REVISION) -> Path:
    return cache_file(
        cache, DEFAULT_MODEL, "joint_schema_model.py", _LOADER,
        revision=None, commit=commit,
    ).parent


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


@pytest.mark.parametrize("cached,offline", [(True, False), (True, True), (False, False), (False, True)])
def test_cache_resolution_through_backend(backend, monkeypatch, tmp_path, cached, offline):
    use_hub_cache(monkeypatch, tmp_path / "unused-default-cache", offline=offline)
    directory = _snapshot(backend.config.model_cache if cached else tmp_path / "downloaded")
    online = Mock(return_value=str(directory))
    real_lookup = huggingface_hub.snapshot_download

    def fetch(**kwargs):
        return real_lookup(**kwargs) if kwargs.get("local_files_only") else online(**kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    attempts = block_network(monkeypatch)
    if offline and not cached:
        with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE") as error:
            backend.load()
        message = str(error.value)
        assert DEFAULT_MODEL in message and DEFAULT_REVISION in message
        assert str(backend.config.model_cache) in message
        assert backend.model is None
    else:
        backend.load()
        assert backend.model["directory"] == directory
        assert backend.model["device"] == "cpu"
    if cached or offline:
        online.assert_not_called()
    else:
        online.assert_called_once_with(
            repo_id=DEFAULT_MODEL, revision=DEFAULT_REVISION, cache_dir=backend.config.model_cache,
        )
    assert attempts == []


def test_configured_revision_is_required(backend, monkeypatch):
    use_hub_cache(monkeypatch, backend.config.model_cache, offline=True)
    _snapshot(backend.config.model_cache, commit="a" * 40)
    attempts = block_network(monkeypatch)
    with pytest.raises(RuntimeError, match=DEFAULT_REVISION):
        backend.load()
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
