# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple VLM launcher TTS selection coverage."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MAIN = _ROOT / "agent-samples" / "simple-vlm-example" / "main.py"
_SPEC = importlib.util.spec_from_file_location("simple_vlm_tts_selection", _MAIN)
assert _SPEC and _SPEC.loader
_main = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_main)


def test_piper_selection_preserves_main_tts_path() -> None:
    processes, credentials = _main._build_processes("piper")

    tts = next(process for process in processes if process.name == "tts")
    worker = next(process for process in processes if process.name == "worker")
    assert tts.project == "../../services/piper-tts"
    assert tts.command == "piper_tts_server"
    assert worker.config == "yaml/simple_vlm_example_worker.yaml"
    assert "NGC_API_KEY" not in credentials


def test_magpie_selection_uses_streaming_nim_and_memory_profile() -> None:
    processes, credentials = _main._build_processes("magpie")

    tts = next(process for process in processes if process.name == "tts")
    vlm = next(process for process in processes if process.name == "vlm")
    worker = next(process for process in processes if process.name == "worker")
    assert tts.project == "../../services/magpie-tts-nim"
    assert tts.command == "magpie_tts_nim_server"
    assert tts.config == "yaml/magpie_tts_nim_server.yaml"
    assert tts.quiet_native_output is True
    assert vlm.config == "yaml/vlm_server.magpie.yaml"
    assert worker.config == "yaml/simple_vlm_example_worker.magpie.yaml"
    assert credentials == ("NGC_API_KEY",)
    assert not any(process.project == "../../services/piper-tts" for process in processes)


def test_unknown_tts_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported TTS backend"):
        _main._build_processes("other")


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--piper", "piper"), ("--magpie", "magpie")],
)
def test_cli_flags_select_tts_backend(monkeypatch, flag, expected) -> None:
    selected: list[str] = []
    monkeypatch.setattr(sys, "argv", [str(_MAIN), flag, "--allow-anonymous"])
    monkeypatch.setattr(_main, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        _main,
        "_build_processes",
        lambda backend: selected.append(backend) or ([], ()),
    )
    monkeypatch.setattr(_main, "require_credentials", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_main, "ensure_credentials", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_main, "run_stack", lambda *_args, **_kwargs: None)

    _main.run()

    assert selected == [expected]
