# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr-render-demo process selection."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAIN_PATH = _REPO_ROOT / "agent-samples/xr-render-demo/main.py"
_SPEC = importlib.util.spec_from_file_location("xr_render_demo_main", _MAIN_PATH)
assert _SPEC and _SPEC.loader
_render_demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_render_demo)


def _reused_ports(processes: list[object]) -> dict[str, int | None]:
    return {
        process.name: process.port
        for process in processes
        if process.launch_mode == "reuse"
    }


def test_local_reused_models_declare_health_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_render_demo, "_model_backend", lambda: "local")

    assert _reused_ports(_render_demo._build_processes()) == {
        "stt": 8103,
        "agent-llm": 8107,
        "vlm": 8100,
    }


def test_nim_omits_hosted_models_from_reuse_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_render_demo, "_model_backend", lambda: "nim")

    assert _reused_ports(_render_demo._build_processes()) == {"stt": 8103}
