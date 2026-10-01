# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import xr_ai_launcher._stack as _stack

_PASS = [{
    "name": "python", "status": "passed", "detected": "3.12",
    "required": ">=3.11", "remediation": "",
}]
_FAIL = [{
    "name": "driver", "status": "failed", "detected": "570",
    "required": ">=580", "remediation": "Upgrade the driver.",
}]
_SKIP = [{
    "name": "video_codecs", "status": "skipped",
    "detected": "checked by DeviceIOHub", "required": "NVDEC and NVENC",
    "remediation": "",
}]


def _options(*, check: bool = False, prepare: bool = False, json_: bool = False):
    return argparse.Namespace(check=check, prepare=prepare, json=json_)


@pytest.fixture
def phases(monkeypatch):
    events: list[str] = []
    report = {"rows": list(_PASS), "runtime": [], "shutdowns": []}
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
    monkeypatch.setattr(
        _stack,
        "_shutdown",
        lambda procs, **_kwargs: report["shutdowns"].append(tuple(procs)),
    )
    monkeypatch.setattr(_stack, "_print_ready_banner", lambda _names: None)
    return events, report, preparation_codes


def test_failed_preflight_stops_before_preparation_and_launch(
    phases, tmp_path, capsys,
) -> None:
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

    captured = capsys.readouterr()
    assert error.value.code == 1
    assert events == ["preflight"]
    assert captured.out == ""
    assert "[failed] driver" in captured.err


@pytest.mark.parametrize(("sample_rows", "code"), [(_SKIP, None), (_FAIL, 1)])
def test_check_json_is_one_stdout_document_without_side_effects(
    phases, tmp_path, monkeypatch, capsys, sample_rows, code,
) -> None:
    events, report, _codes = phases
    monkeypatch.setattr(
        _stack,
        "_spawn",
        lambda *_args, **_kwargs: pytest.fail("--check must not spawn a process"),
    )
    with pytest.raises(SystemExit) if code else nullcontext() as error:
        _stack.run_stack(
            [_stack.Process("model", ".", "model", prepare=True)],
            tmp_path,
            options=_options(check=True, json_=True),
            check_sample=lambda: sample_rows,
            prepare_sample=lambda: pytest.fail("sample preparation must not run"),
            before_launch=lambda: pytest.fail("launch hook must not run"),
        )

    captured = capsys.readouterr()
    assert (error.value.code if code else None) == code
    assert json.loads(captured.out) == [*_PASS, *sample_rows]
    assert captured.err == ""
    assert events == ["preflight"]
    assert report["runtime"] == [True]


def test_human_check_writes_statuses_to_stderr_and_skip_is_nonfatal(
    phases, tmp_path, capsys,
) -> None:
    events, report, _codes = phases
    report["rows"] = [*_PASS, *_SKIP]

    _stack.run_stack([], tmp_path, options=_options(check=True))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[passed] python" in captured.err
    assert "[skipped] video_codecs" in captured.err
    assert events == ["preflight"]


@pytest.mark.parametrize(("options", "expected"), [
    (_options(prepare=True), ["preflight", "prepare:model", "prepare:sample"]),
    (None, [
        "preflight", "before_launch", "prepare:model", "prepare:sample",
        "launch:model", "launch:worker",
    ]),
])
def test_check_prepare_and_launch_phase_order(
    phases, tmp_path, options, expected, capsys,
) -> None:
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
    output = capsys.readouterr().out
    assert "[reused] Skipping preparation: service is reused" in output
    assert "[model] Preparation complete" in output


def test_prepare_reports_external_endpoint_ownership(phases, tmp_path, capsys) -> None:
    events, _report, _codes = phases

    _stack.run_stack(
        [],
        tmp_path,
        options=_options(prepare=True),
        model_profile=tmp_path / "models.json",
    )

    assert events == ["preflight"]
    note = capsys.readouterr().err
    assert "For external or reused endpoints" in note
    assert "model_servers --prepare" in note


def test_failed_preparation_stops_before_sample_or_launch(phases, tmp_path) -> None:
    events, report, codes = phases
    codes["model"] = 7

    with pytest.raises(SystemExit, match="model: preparation failed"):
        _stack.run_stack(
            [_stack.Process("model", ".", "model", prepare=True)],
            tmp_path,
            prepare_sample=lambda: pytest.fail("sample preparation ran"),
        )

    assert events == ["preflight", "prepare:model"]
    assert report["shutdowns"] == [("model",)]


@pytest.mark.parametrize("checking", [False, True])
def test_sigterm_restores_handler_and_stops_preparing_child(
    phases, tmp_path, monkeypatch, checking,
) -> None:
    events, report, codes = phases
    original = object()
    handlers = {_stack.signal.SIGTERM: original}

    def install_handler(sig, handler):
        previous = handlers[sig]
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(_stack.signal, "signal", install_handler)
    def interrupt():
        handlers[_stack.signal.SIGTERM](_stack.signal.SIGTERM, None)

    if checking:
        def interrupting_preflight(*_args, **_kwargs):
            events.append("preflight")
            interrupt()
        monkeypatch.setattr(_stack, "preflight", interrupting_preflight)
    else:
        codes["model"] = interrupt

    with pytest.raises(SystemExit) as error:
        _stack.run_stack(
            [_stack.Process("model", ".", "model", prepare=True)],
            tmp_path,
            options=_options(check=checking),
        )

    assert error.value.code == 130
    assert handlers[_stack.signal.SIGTERM] is original
    assert report["shutdowns"] == ([] if checking else [("model",)])


def test_json_without_check_is_usage_error(phases, tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        _stack.run_stack([], tmp_path, options=_options(json_=True))
    assert error.value.code == 2


@pytest.mark.parametrize(("relative_main", "arguments"), [
    ("model-server-samples/model-servers/main.py", [
        "--check", "--json", "--gpu-profile", "spark",
    ]),
    ("model-server-samples/model-servers-nim/main.py", [
        "--check", "--json", "--gpu-profile", "spark",
    ]),
    ("agent-samples/lab-instrument-monitoring/main.py", ["--check", "--json"]),
    ("agent-samples/simple-vlm-example/main.py", ["--check", "--json"]),
    ("agent-samples/tea-making-sample/main.py", ["--check", "--json"]),
])
def test_sample_check_json_wires_profile_without_launching(
    relative_main,
    arguments,
    monkeypatch,
    capsys,
) -> None:
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
    monkeypatch.setattr(
        _stack,
        "_spawn",
        lambda *_args, **_kwargs: pytest.fail("--check must not launch a process"),
    )
    seen = {}
    monkeypatch.setattr(
        _stack,
        "preflight",
        lambda *_args, **kwargs: (
            seen.update(kwargs, profile_found=kwargs["model_profile"].is_file()) or _PASS
        ),
    )
    monkeypatch.setattr(sys, "argv", [path.stem, *arguments])

    module.run()

    captured = capsys.readouterr()
    assert seen["profile_found"]
    assert json.loads(captured.out) == _PASS
    assert captured.err == ""
