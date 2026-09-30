# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import xr_ai_launcher._stack as _stack

_PASS = [{"name": "python", "ok": True, "detected": "3.12",
          "required": ">=3.11", "remediation": ""}]
_FAIL = [{"name": "vulkan", "ok": False, "detected": "not found",
          "required": "Vulkan loader", "remediation": "Install the Vulkan loader (libvulkan1 on Ubuntu)."}]


def _options(*, check: bool = False, prepare: bool = False, json_: bool = False):
    return argparse.Namespace(check=check, prepare=prepare, json=json_)


@pytest.fixture
def phases(monkeypatch):
    events: list[str] = []
    report = {"rows": _PASS, "runtime": [], "shutdowns": []}
    preparation_codes: dict[str, object] = {}
    class Child:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait(self) -> int:
            result = preparation_codes.get(self.name, 0)
            return result() if callable(result) else result
    def fake_preflight(*_args, **kwargs):
        events.append("preflight")
        report["runtime"].append(kwargs["runtime"])
        return report["rows"]
    def fake_spawn(process, _base, _ready_file, *, ready_process_may_exit=False,
                   prepare=False):
        del ready_process_may_exit
        events.append(f"{'prepare' if prepare else 'launch'}:{process.name}")
        return Child(process.name)
    monkeypatch.setattr(_stack, "load_credentials", lambda: None)
    monkeypatch.setattr(_stack, "preflight", fake_preflight)
    monkeypatch.setattr(_stack, "_spawn", fake_spawn)
    monkeypatch.setattr(_stack, "_wait_ready", lambda *_args: None)
    monkeypatch.setattr(_stack, "_shutdown", lambda procs, **_kwargs:
                        report["shutdowns"].append(tuple(procs)))
    monkeypatch.setattr(_stack, "_print_ready_banner", lambda _names: None)
    return events, report, preparation_codes


def test_failed_preflight_stops_before_preparation_and_launch(
    phases, tmp_path, capsys,
):
    events, report, _codes = phases
    report["rows"] = _FAIL
    process = _stack.Process("model", ".", "model", prepare=True)
    with pytest.raises(SystemExit) as error:
        _stack.run_stack(
            [process],
            tmp_path,
            prepare_sample=lambda: pytest.fail("sample preparation must not run"),
            before_launch=lambda: pytest.fail("launch hook must not run"),
        )
    assert error.value.code == 1
    assert events == ["preflight"]
    assert "FAIL" in capsys.readouterr().out


def test_check_json_is_clean_and_has_no_side_effects(phases, tmp_path, capsys):
    events, report, _codes = phases
    process = _stack.Process("model", ".", "model", prepare=True)
    _stack.run_stack(
        [process],
        tmp_path,
        options=_options(check=True, json_=True),
        prepare_sample=lambda: pytest.fail("sample preparation must not run"),
        before_launch=lambda: pytest.fail("launch hook must not run"),
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == report["rows"]
    assert captured.err == ""
    assert events == ["preflight"]
    assert report["runtime"] == [True]


@pytest.mark.parametrize(("options", "expected"), [
    (_options(prepare=True), ["preflight", "prepare:model", "prepare:sample"]),
    (None, ["preflight", "before_launch", "prepare:model", "prepare:sample",
            "launch:model", "launch:worker"]),
])
def test_check_prepare_and_launch_phase_order(phases, tmp_path, options, expected):
    events, report, _codes = phases
    processes = [
        _stack.Process("model", ".", "model", prepare=True),
        _stack.Process("reused", ".", "reused", prepare=True, launch_mode="reuse"),
        _stack.Process("worker", ".", "worker"),
    ]
    _stack.run_stack(
        processes,
        tmp_path,
        options=options,
        exit_after_ready=True,
        prepare_sample=lambda: events.append("prepare:sample"),
        before_launch=lambda: events.append("before_launch"),
    )
    assert events == expected
    assert report["runtime"] == [not bool(options and options.prepare)]


def test_failed_preparation_stops_before_sample_or_launch(phases, tmp_path):
    events, report, codes = phases
    codes["model"] = 7
    with pytest.raises(SystemExit, match="model: preparation failed"):
        _stack.run_stack(
            [_stack.Process("model", ".", "model", prepare=True)], tmp_path,
            prepare_sample=lambda: pytest.fail("sample preparation ran"),
        )
    assert events == ["preflight", "prepare:model"]
    assert report["shutdowns"] == [("model",)]


def test_sigterm_during_preparation_restores_handler_and_stops_child(
    phases, tmp_path, monkeypatch,
):
    _events, report, codes = phases
    original = object()
    handlers = {_stack.signal.SIGTERM: original}
    def install_handler(sig, handler):
        previous = handlers[sig]
        handlers[sig] = handler
        return previous
    monkeypatch.setattr(_stack.signal, "signal", install_handler)
    codes["model"] = lambda: handlers[_stack.signal.SIGTERM](_stack.signal.SIGTERM, None)
    with pytest.raises(SystemExit) as error:
        _stack.run_stack(
            [_stack.Process("model", ".", "model", prepare=True)], tmp_path,
        )
    assert error.value.code == 130
    assert handlers[_stack.signal.SIGTERM] is original
    assert report["shutdowns"] == [("model",)]


def test_json_without_check_is_usage_error(phases, tmp_path):
    with pytest.raises(SystemExit) as error:
        _stack.run_stack([], tmp_path, options=_options(json_=True))
    assert error.value.code == 2


@pytest.mark.parametrize(("relative_main", "arguments"), [
    ("model-server-samples/model-servers/main.py",
     ["--check", "--json", "--gpu-profile", "spark"]),
    ("model-server-samples/model-servers-nim/main.py",
     ["--check", "--json", "--gpu-profile", "spark"]),
])
def test_model_server_check_json_is_clean_and_does_not_launch(
    relative_main,
    arguments,
    monkeypatch,
    capsys,
):
    root = Path(__file__).resolve().parents[1]
    path = root / relative_main
    spec = importlib.util.spec_from_file_location(
        f"launcher_check_{path.parent.name.replace('-', '_')}", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_stack, "load_credentials", lambda: None)
    monkeypatch.setattr(_stack, "preflight", lambda *_args, **_kwargs: _PASS)
    monkeypatch.setattr(sys, "argv", [path.stem, *arguments])
    module.run()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == _PASS
