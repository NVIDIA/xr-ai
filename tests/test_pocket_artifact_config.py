# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pocket TTS artifact resolution tests independent of the optional package."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_POCKET_MAIN = _ROOT / "services/pocket-tts/pocket_tts_server/__main__.py"
_POCKET_FIXTURE = _ROOT / "tests/fixtures/pocket_tts-3.0.2/pocket_tts"


def _load_pocket_module():
    name = "pocket_artifact_config_main"
    spec = importlib.util.spec_from_file_location(name, _POCKET_MAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture
def pocket_module():
    return _load_pocket_module()


def test_hf_cache_path_maps_nested_pinned_artifact(
    tmp_path: Path,
    pocket_module,
) -> None:
    uri = (
        "hf://kyutai/pocket-tts/languages/english/model.safetensors"
        "@39592ff23c9ef80098bb74895d104c26275fe2c9"
    )

    resolved = pocket_module._hf_cache_path(tmp_path, uri)

    assert resolved == (
        tmp_path
        / "models--kyutai--pocket-tts"
        / "snapshots"
        / "39592ff23c9ef80098bb74895d104c26275fe2c9"
        / "languages/english/model.safetensors"
    )


def test_pocket_artifact_paths_match_pinned_english_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pocket_module,
) -> None:
    monkeypatch.setattr(
        pocket_module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[_POCKET_FIXTURE]),
    )

    resolved = pocket_module._pocket_artifact_paths(
        tmp_path,
        "english",
        "bill_boerst",
    )

    assert resolved == (
        tmp_path
        / "models--kyutai--pocket-tts"
        / "snapshots"
        / "39592ff23c9ef80098bb74895d104c26275fe2c9"
        / "languages/english/model.safetensors",
        tmp_path
        / "models--kyutai--pocket-tts-without-voice-cloning"
        / "snapshots"
        / "d29db7978e464fb90cb3359ee0c69a273b9142cc"
        / "languages/english/model.safetensors",
        tmp_path
        / "models--kyutai--pocket-tts-without-voice-cloning"
        / "snapshots"
        / "d29db7978e464fb90cb3359ee0c69a273b9142cc"
        / "languages/english/tokenizer.model",
        tmp_path
        / "models--kyutai--pocket-tts-without-voice-cloning"
        / "snapshots"
        / "e81d79e8194ad4c7ce879c87a4258ef20cbf2487"
        / "languages/english/embeddings/bill_boerst.safetensors",
    )
