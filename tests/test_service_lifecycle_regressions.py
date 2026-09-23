# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for shared service preparation."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import device_io_hub.__main__ as hub_main
import pytest
from xr_ai_launcher import read_artifact_manifest, write_artifact_manifest

_ROOT = Path(__file__).resolve().parents[1]
_MISSING_MODULE = object()


def _load_pocket_module():
    path = _ROOT / "services/pocket-tts/pocket_tts_server/__main__.py"
    spec = importlib.util.spec_from_file_location("lifecycle_pocket_tts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name, _MISSING_MODULE)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is _MISSING_MODULE:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous


@pytest.fixture
def pocket_module():
    return _load_pocket_module()


def test_artifact_manifest_accepts_empty_files_and_rejects_unsafe_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    empty = root / "metadata.json"
    weights = root / "weights.bin"
    empty.write_bytes(b"")
    weights.write_bytes(b"weights")
    marker = tmp_path / "prepared.json"

    written = write_artifact_manifest(marker, root, (empty, weights))
    loaded = read_artifact_manifest(marker, expected_root=root)

    assert written.size == len(b"weights")
    assert loaded == written
    marker.write_text(
        json.dumps(
            {
                "artifact_root": str(root),
                "files": [["../outside", 1]],
                "version": 1,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "outside").write_bytes(b"x")
    assert read_artifact_manifest(marker) is None


def test_hub_main_prefixes_config_value_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hub_main,
        "load_config",
        lambda: (_ for _ in ()).throw(ValueError("invalid runtime config")),
    )
    monkeypatch.setattr(hub_main, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hub_main.sys, "argv", ["device_io_hub"])

    with pytest.raises(SystemExit) as error:
        hub_main.run()

    assert str(error.value) == "[device_io_hub] invalid runtime config"


def test_pocket_prepare_uses_hf_hub_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pocket_module,
) -> None:
    pocket = pocket_module
    hf_home = tmp_path / "hf-home"
    hub_cache = tmp_path / "hub-cache"
    model_cache = tmp_path / "models"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    selected_roots: list[Path] = []
    owned = tuple(hub_cache / name for name in ("model", "tokenizer", "voice"))

    def artifact_paths(
        root: Path,
        _language: str,
        _voice: str,
    ) -> tuple[Path, Path, Path, Path]:
        selected_roots.append(root)
        return (hub_cache / "unused", *owned)

    def load(backend) -> None:
        backend._model = SimpleNamespace(has_voice_cloning=False)
        for path in owned:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"artifact")

    monkeypatch.setattr(pocket, "_pocket_artifact_paths", artifact_paths)
    monkeypatch.setattr(pocket, "_pocket_version", lambda: "3.0.2")
    monkeypatch.setattr(pocket, "_pocket_auth_identity", lambda: "anonymous")
    monkeypatch.setattr(pocket._PocketTTSBackend, "_ensure_loaded", load)
    config = {
        "voice": "bill_boerst",
        "language": "english",
        "model_cache": str(model_cache),
    }

    pocket._prepare(config, tmp_path)
    pocket._prepare(config, tmp_path)

    assert selected_roots == [hub_cache, hub_cache]
    marker = next((model_cache / "pocket").glob(".xr-ai-prepare-*"))
    assert read_artifact_manifest(marker, expected_root=hub_cache) is not None


def test_pocket_artifact_uri_requires_a_revision(
    tmp_path: Path,
    pocket_module,
) -> None:
    with pytest.raises(RuntimeError, match="must pin a revision"):
        pocket_module._hf_cache_path(tmp_path, "hf://owner/repository/model.bin")
