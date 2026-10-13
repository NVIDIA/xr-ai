# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-first ``.nemo`` checkpoint resolution in the STT and Magpie TTS servers.

NeMo and torch are replaced with stand-ins, so these run without either.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _helpers_hf import COMMIT, block_network, cache_file, use_hub_cache
from stt_server import __main__ as stt_main

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
_TTS_MODEL = "nvidia/magpie_tts_multilingual_357m"


def _load_magpie():
    spec = importlib.util.spec_from_file_location(
        "magpie_tts_server_main",
        _REPO_ROOT / "services/magpie-tts/magpie_tts_server/__main__.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_nemo_module(monkeypatch, name: str, **attributes) -> None:
    """Install ``name`` and its parent packages as attribute-linked stand-ins."""
    parts = name.split(".")
    child = SimpleNamespace(**attributes)
    for depth in range(len(parts), 0, -1):
        module_name = ".".join(parts[:depth])
        monkeypatch.setitem(sys.modules, module_name, child)
        if depth > 1:
            child = SimpleNamespace(**{parts[depth - 1]: child})


@pytest.fixture(params=["stt", "magpie"])
def backend(request, tmp_path, monkeypatch):
    cache = use_hub_cache(monkeypatch, tmp_path / "hub")
    loader = Mock()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    ))
    if request.param == "stt":
        model_name = _STT_MODEL
        _fake_nemo_module(monkeypatch, "nemo.collections.asr", models=SimpleNamespace(ASRModel=loader))
        instance = stt_main._AsrBackend(model_name, "cpu", cache)
    else:
        model_name = _TTS_MODEL
        _fake_nemo_module(monkeypatch, "nemo.collections.tts.models.magpietts", MagpieTTSModel=loader)
        instance = _load_magpie()._TtsBackend(model_name, "cpu", 22050, cache)
    return instance, loader, model_name, cache


def test_cached_checkpoint_restores_without_network(backend, monkeypatch):
    instance, loader, model_name, cache = backend
    checkpoint = cache_file(cache, model_name, model_name.split("/")[-1] + ".nemo")
    attempts = block_network(monkeypatch)
    instance._ensure_loaded()
    loader.restore_from.assert_called_once_with(restore_path=str(checkpoint))
    loader.from_pretrained.assert_not_called()
    assert instance.ready and attempts == []


def test_cache_miss_uses_existing_loader(backend):
    instance, loader, model_name, _cache = backend
    instance._ensure_loaded()
    loader.from_pretrained.assert_called_once_with(model_name)
    loader.restore_from.assert_not_called()


def test_offline_miss_reports_model_and_cache(backend, monkeypatch):
    instance, loader, model_name, cache = backend
    use_hub_cache(monkeypatch, cache, offline=True)
    attempts = block_network(monkeypatch)
    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE") as error:
        instance._ensure_loaded()
    assert model_name in str(error.value) and str(cache) in str(error.value)
    assert "revision main" in str(error.value) and attempts == []
    loader.restore_from.assert_not_called()
    loader.from_pretrained.assert_not_called()


def test_magpie_lookup_uses_configured_revision(tmp_path, monkeypatch):
    cache = use_hub_cache(monkeypatch, tmp_path / "hub", offline=True)
    checkpoint = cache_file(cache, _TTS_MODEL, "magpie_tts_multilingual_357m.nemo", revision=None)
    module = _load_magpie()
    assert module._cached_nemo_checkpoint(_TTS_MODEL, COMMIT) == str(checkpoint)
    with pytest.raises(RuntimeError, match="revision other"):
        module._cached_nemo_checkpoint(_TTS_MODEL, "other")


@pytest.mark.parametrize("name", ["stt_en_conformer_ctc_large", "/models/custom/asr.nemo"])
def test_non_hub_models_are_left_to_nemo(tmp_path, monkeypatch, name):
    use_hub_cache(monkeypatch, tmp_path / "hub", offline=True)
    assert stt_main._cached_nemo_checkpoint(name) is None
