# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Nano reasoning-parser preparation."""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = (
    _ROOT
    / "services/nemotron3-nano-llm/nemotron3_nano_llm_server/__main__.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("nano_parser_prepare", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reasoning_parser_is_readable_before_replace_and_when_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nano = _load_module()
    target = tmp_path / nano._PARSER_FILENAME
    replacement_modes: list[int] = []
    real_replace = nano.os.replace

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self) -> bytes:
            return b"parser source"

    def replace(source: Path, destination: Path) -> None:
        source = Path(source)
        replacement_modes.append(stat.S_IMODE(source.stat().st_mode))
        real_replace(source, destination)

    monkeypatch.setattr(nano.urllib.request, "urlopen", lambda _url: Response())
    monkeypatch.setattr(nano.os, "replace", replace)

    assert nano._ensure_reasoning_parser(tmp_path, "https://example/parser") == target
    assert replacement_modes == [0o644]
    assert stat.S_IMODE(target.stat().st_mode) == 0o644

    target.chmod(0o600)
    monkeypatch.setattr(
        nano.urllib.request,
        "urlopen",
        lambda _url: pytest.fail("a cached parser must remain offline"),
    )

    assert nano._ensure_reasoning_parser(tmp_path, "https://example/parser") == target
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_prepare_repairs_cached_reasoning_parser_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nano = _load_module()
    target = tmp_path / nano._PARSER_FILENAME
    target.write_bytes(b"parser source")
    target.chmod(0o600)
    prepared: list[dict] = []

    monkeypatch.setattr(nano, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        nano,
        "load_config",
        lambda: ({"model_ada": "example/model", "spark_uma": False}, tmp_path, None),
    )
    monkeypatch.setattr(nano, "resolve_model_cache", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(nano, "setup_hf_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nano, "gpu_compute_major", lambda: 8)
    monkeypatch.setattr(nano, "prepare_vllm", lambda **kwargs: prepared.append(kwargs))
    monkeypatch.setattr(
        nano.urllib.request,
        "urlopen",
        lambda _url: pytest.fail("a cached parser must remain offline"),
    )
    monkeypatch.setattr(
        nano,
        "serve",
        lambda **_kwargs: pytest.fail("--prepare must not start the service"),
    )
    monkeypatch.setattr(sys, "argv", ["nemotron3_nano_llm_server", "--prepare"])

    nano.run()

    assert len(prepared) == 1
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
