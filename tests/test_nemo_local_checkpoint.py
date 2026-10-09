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


def _fake_cpu_torch(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )


def _fake_nemo_module(monkeypatch, name: str, **attributes) -> None:
    """Install ``name`` and its parent packages as attribute-linked stand-ins."""
    parts = name.split(".")
    child = SimpleNamespace(**attributes)
    for depth in range(len(parts), 0, -1):
        module_name = ".".join(parts[:depth])
        monkeypatch.setitem(sys.modules, module_name, child)
        if depth > 1:
            child = SimpleNamespace(**{parts[depth - 1]: child})


def _fake_loader() -> Mock:
    model = Mock()
    model.eval.return_value = None
    return Mock(
        restore_from=Mock(return_value=model),
        from_pretrained=Mock(return_value=model),
    )


class TestSttCheckpoint:
    def test_cache_hit_makes_no_network_request(self, tmp_path, monkeypatch):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        checkpoint = cache_file(cache, _STT_MODEL, "parakeet-tdt-0.6b-v3.nemo")
        attempts = block_network(monkeypatch)

        assert stt_main._cached_nemo_checkpoint(_STT_MODEL) == str(checkpoint)
        assert attempts == []

    def test_cache_miss_returns_none(self, tmp_path, monkeypatch):
        use_hub_cache(monkeypatch, tmp_path / "hub")
        attempts = block_network(monkeypatch)

        assert stt_main._cached_nemo_checkpoint(_STT_MODEL) is None
        assert attempts == []

    def test_explicit_offline_miss_names_model(self, tmp_path, monkeypatch):
        use_hub_cache(monkeypatch, tmp_path / "hub", offline=True)

        with pytest.raises(RuntimeError, match=rf"{_STT_MODEL} \(revision main"):
            stt_main._cached_nemo_checkpoint(_STT_MODEL)

    def test_ngc_catalog_name_is_left_to_nemo(self, tmp_path, monkeypatch):
        use_hub_cache(monkeypatch, tmp_path / "hub", offline=True)

        assert stt_main._cached_nemo_checkpoint("stt_en_conformer_ctc_large") is None

    def test_non_hub_path_is_left_to_nemo(self, tmp_path, monkeypatch):
        use_hub_cache(monkeypatch, tmp_path / "hub", offline=True)

        assert stt_main._cached_nemo_checkpoint("/models/custom/asr.nemo") is None

    def _backend_with_fake_nemo(self, tmp_path, monkeypatch) -> tuple[object, Mock]:
        loader = _fake_loader()
        _fake_cpu_torch(monkeypatch)
        _fake_nemo_module(
            monkeypatch,
            "nemo.collections.asr",
            models=SimpleNamespace(ASRModel=loader),
        )
        return stt_main._AsrBackend(_STT_MODEL, "auto", tmp_path), loader

    def test_backend_restores_cached_checkpoint(self, tmp_path, monkeypatch):
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        checkpoint = cache_file(cache, _STT_MODEL, "parakeet-tdt-0.6b-v3.nemo")
        backend, loader = self._backend_with_fake_nemo(tmp_path, monkeypatch)
        attempts = block_network(monkeypatch)

        backend._ensure_loaded()

        loader.restore_from.assert_called_once_with(restore_path=str(checkpoint))
        loader.from_pretrained.assert_not_called()
        assert attempts == []

    def test_backend_downloads_on_cache_miss(self, tmp_path, monkeypatch):
        use_hub_cache(monkeypatch, tmp_path / "hub")
        backend, loader = self._backend_with_fake_nemo(tmp_path, monkeypatch)

        backend._ensure_loaded()

        loader.from_pretrained.assert_called_once_with(_STT_MODEL)
        loader.restore_from.assert_not_called()


class TestMagpieCheckpoint:
    def test_pinned_revision_hit_makes_no_network_request(self, tmp_path, monkeypatch):
        module = _load_magpie()
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        checkpoint = cache_file(
            cache, _TTS_MODEL, "magpie_tts_multilingual_357m.nemo", revision=None
        )
        attempts = block_network(monkeypatch)

        assert module._cached_nemo_checkpoint(_TTS_MODEL, COMMIT) == str(checkpoint)
        assert attempts == []

    def test_explicit_offline_miss_names_model_and_revision(self, tmp_path, monkeypatch):
        module = _load_magpie()
        use_hub_cache(monkeypatch, tmp_path / "hub", offline=True)

        with pytest.raises(RuntimeError, match=rf"{_TTS_MODEL} \(revision {COMMIT}"):
            module._cached_nemo_checkpoint(_TTS_MODEL, COMMIT)

    def test_backend_restores_cached_checkpoint(self, tmp_path, monkeypatch):
        module = _load_magpie()
        cache = use_hub_cache(monkeypatch, tmp_path / "hub")
        checkpoint = cache_file(cache, _TTS_MODEL, "magpie_tts_multilingual_357m.nemo")
        loader = _fake_loader()
        _fake_cpu_torch(monkeypatch)
        _fake_nemo_module(
            monkeypatch, "nemo.collections.tts.models.magpietts", MagpieTTSModel=loader
        )
        attempts = block_network(monkeypatch)

        module._TtsBackend(_TTS_MODEL, "auto", 22050, tmp_path)._ensure_loaded()

        loader.restore_from.assert_called_once_with(restore_path=str(checkpoint))
        loader.from_pretrained.assert_not_called()
        assert attempts == []
