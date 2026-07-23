# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cloudxr_runtime process launch behavior."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _install_cloudxr_runtime_fakes(monkeypatch):
    isaacteleop = types.ModuleType("isaacteleop")
    isaacteleop.__version__ = "test"

    cloudxr_pkg = types.ModuleType("isaacteleop.cloudxr")

    env_config = types.ModuleType("isaacteleop.cloudxr.env_config")

    class EnvConfig:
        @classmethod
        def from_args(cls, *_args, **_kwargs):
            return cls()

    env_config.EnvConfig = EnvConfig
    env_config.get_env_config = lambda: types.SimpleNamespace(openxr_run_dir=lambda: "/tmp")

    runtime = types.ModuleType("isaacteleop.cloudxr.runtime")

    def runtime_run():
        return None

    runtime.run = runtime_run
    runtime.check_eula = lambda *args, **kwargs: None
    runtime.latest_runtime_log = lambda: None
    runtime.runtime_version = lambda: "test"
    runtime.terminate_or_kill_runtime = lambda _proc: None
    runtime.wait_for_runtime_ready = lambda *args, **kwargs: True

    wss = types.ModuleType("isaacteleop.cloudxr.wss")
    wss.run = lambda *args, **kwargs: None

    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda _stream: {}

    loguru = types.ModuleType("loguru")
    loguru.logger = types.SimpleNamespace(
        debug=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )

    xr_ai_logging = types.ModuleType("xr_ai_logging")
    xr_ai_logging.setup_logging = lambda *_args, **_kwargs: None

    monkeypatch.setitem(sys.modules, "isaacteleop", isaacteleop)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", cloudxr_pkg)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr.env_config", env_config)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr.runtime", runtime)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr.wss", wss)
    monkeypatch.setitem(sys.modules, "yaml", yaml)
    monkeypatch.setitem(sys.modules, "loguru", loguru)
    monkeypatch.setitem(sys.modules, "xr_ai_logging", xr_ai_logging)


def _import_cloudxr_runtime(monkeypatch):
    _install_cloudxr_runtime_fakes(monkeypatch)
    package_dir = Path(__file__).resolve().parents[1] / "cloudxr-runtime"
    monkeypatch.syspath_prepend(str(package_dir))
    sys.modules.pop("cloudxr_runtime.__main__", None)
    sys.modules.pop("cloudxr_runtime", None)
    return importlib.import_module("cloudxr_runtime.__main__")


def test_start_runtime_process_uses_spawn_context(monkeypatch):
    module = _import_cloudxr_runtime(monkeypatch)
    requested_contexts = []
    started_targets = []

    class FakeProcess:
        def __init__(self, target):
            self.target = target

        def start(self):
            started_targets.append(self.target)

    class FakeContext:
        def Process(self, target):  # pylint: disable=invalid-name
            return FakeProcess(target)

    def fake_get_context(method):
        requested_contexts.append(method)
        return FakeContext()

    monkeypatch.setattr(module.multiprocessing, "get_context", fake_get_context)

    try:
        proc = module._start_runtime_process()

        assert requested_contexts == ["spawn"]
        assert started_targets == [module.runtime_run]
        assert proc.target is module.runtime_run
    finally:
        sys.modules.pop("cloudxr_runtime.__main__", None)
        sys.modules.pop("cloudxr_runtime", None)
