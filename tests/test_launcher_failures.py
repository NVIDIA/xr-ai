# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure-summary tests for the stack launcher."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
import xr_ai_launcher._failure as _failure
import xr_ai_launcher._stack as _stack


def _set_log_dir(monkeypatch, root: Path, *, create: bool = True) -> Path:
    monkeypatch.setenv("XR_AI_LOG_ROOT", str(root))
    monkeypatch.setenv("XR_AI_LOG_NAMESPACE", "failure-test")
    monkeypatch.setenv("XR_AI_LOG_TIMESTAMP", "2026-10-05_12-00-00")
    log_dir = root / "log_failure-test_2026-10-05_12-00-00"
    if create:
        log_dir.mkdir(parents=True)
    return log_dir


def _clear_log_dir(monkeypatch) -> None:
    monkeypatch.delenv("XR_AI_LOG_ROOT", raising=False)
    monkeypatch.delenv("XR_AI_LOG_NAMESPACE", raising=False)
    monkeypatch.delenv("XR_AI_LOG_TIMESTAMP", raising=False)


def _write_module(tmp_path: Path, name: str, source: str) -> None:
    (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")


def _run_python_module(monkeypatch, tmp_path: Path, module: str) -> None:
    monkeypatch.setattr(_stack, "load_credentials", lambda: None)
    monkeypatch.setattr(_stack.shutil, "which", lambda _command: None)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    _stack.run_stack([_stack.Process("test-service", tmp_path, module)], tmp_path)


@pytest.fixture
def gated_output_forwarders(monkeypatch):
    forward = _stack._forward
    drain_output_readers = _stack._drain_output_readers
    release_forwarders = threading.Event()
    drain_started = threading.Event()

    def gated_forward(*args, **kwargs):
        release_forwarders.wait()
        forward(*args, **kwargs)

    def releasing_drain(*args, **kwargs):
        drain_started.set()
        release_forwarders.set()
        return drain_output_readers(*args, **kwargs)

    monkeypatch.setattr(_stack, "_forward", gated_forward)
    monkeypatch.setattr(_stack, "_drain_output_readers", releasing_drain)
    try:
        yield drain_started
    finally:
        release_forwarders.set()


@pytest.mark.integration
def test_real_readiness_failure_forwards_output_and_prints_summary(
    monkeypatch, tmp_path, capsys, gated_output_forwarders
):
    log_dir = _set_log_dir(monkeypatch, tmp_path / "logs")
    _write_module(
        tmp_path,
        "readiness_failure",
        "import sys\n"
        "print('opaque stdout detail', flush=True)\n"
        "print('opaque stderr detail', file=sys.stderr, flush=True)\n"
        "raise SystemExit(7)\n",
    )

    with pytest.raises(SystemExit) as excinfo:
        _run_python_module(monkeypatch, tmp_path, "readiness_failure")

    assert excinfo.value.code == 7
    assert gated_output_forwarders.is_set()
    captured = capsys.readouterr()
    assert "[test-service] opaque stdout detail" in captured.out
    assert "[test-service] opaque stderr detail" in captured.out
    assert "Process failure: test-service (readiness)" in captured.err
    assert "Command:" in captured.err
    assert "-m readiness_failure --ready-file" in captured.err
    assert f"Project: {tmp_path}" in captured.err
    assert "Exit code: 7" in captured.err
    assert f"Run logs: {log_dir}" in captured.err


@pytest.mark.integration
def test_real_foreground_runtime_exit_prints_summary_and_stays_nonzero(
    monkeypatch, tmp_path, capsys, gated_output_forwarders
):
    _clear_log_dir(monkeypatch)
    _write_module(
        tmp_path,
        "runtime_failure",
        "import sys\n"
        "from pathlib import Path\n"
        "ready = Path(sys.argv[sys.argv.index('--ready-file') + 1])\n"
        "ready.touch()\n"
        "print('runtime stopped unexpectedly', file=sys.stderr, flush=True)\n"
        "raise SystemExit(9)\n",
    )

    with pytest.raises(SystemExit) as excinfo:
        _run_python_module(monkeypatch, tmp_path, "runtime_failure")

    assert excinfo.value.code == 9
    assert gated_output_forwarders.is_set()
    captured = capsys.readouterr()
    assert "[test-service] runtime stopped unexpectedly" in captured.out
    assert "Process failure: test-service (runtime)" in captured.err
    assert "Exit code: 9" in captured.err
    assert "Run logs unavailable; review the terminal output." in captured.err


def test_signal_summary_and_exit_status(monkeypatch, tmp_path, capsys):
    _clear_log_dir(monkeypatch)
    context = _failure.FailureContext(
        "worker",
        ("worker", "--ready-file", "/tmp/worker.ready"),
        tmp_path,
        None,
    )

    _failure.emit_failure_summary(context, "runtime", returncode=-15)

    terminal = capsys.readouterr().err
    assert "Process failure: worker (runtime)" in terminal
    assert "Terminated by signal: SIGTERM (15)" in terminal


@pytest.mark.parametrize("log_state", ["missing", "unset"])
def test_summary_does_not_point_at_unavailable_log_directory(
    monkeypatch, tmp_path, capsys, log_state
):
    if log_state == "missing":
        missing = _set_log_dir(monkeypatch, tmp_path / "logs", create=False)
    else:
        _clear_log_dir(monkeypatch)
        missing = None
    context = _failure.FailureContext("worker", ("worker",), tmp_path, None)

    _failure.emit_failure_summary(context, "readiness", returncode=2)

    terminal = capsys.readouterr().err
    assert "Run logs unavailable; review the terminal output." in terminal
    if missing is not None:
        assert str(missing) not in terminal


def test_spawn_error_summary_preserves_original_exception(monkeypatch, tmp_path, capsys):
    log_dir = _set_log_dir(monkeypatch, tmp_path / "logs")
    original = OSError("executable format error")
    monkeypatch.setattr(_stack.shutil, "which", lambda _command: None)
    monkeypatch.setattr(_stack.subprocess, "Popen", Mock(side_effect=original))

    with pytest.raises(OSError) as excinfo:
        _stack._spawn(
            _stack.Process("worker", tmp_path, "worker", config="service.yaml"),
            tmp_path,
            tmp_path / "ready",
        )

    assert excinfo.value is original
    terminal = capsys.readouterr().err
    assert "Process failure: worker (spawn)" in terminal
    assert f"Project: {tmp_path}" in terminal
    assert f"Config: {tmp_path / 'service.yaml'}" in terminal
    assert "-m worker --config" in terminal
    assert "Error: OSError: executable format error" in terminal
    assert f"Run logs: {log_dir}" in terminal


def test_summary_error_is_debug_logged_without_escaping(monkeypatch, tmp_path):
    context = _failure.FailureContext("worker", ("worker",), tmp_path, None)
    debug = Mock()
    monkeypatch.setattr(
        _stack,
        "emit_failure_summary",
        Mock(side_effect=RuntimeError("broken")),
    )
    monkeypatch.setattr(_stack.log, "debug", debug)

    _stack._emit_failure_summary(context, "runtime", returncode=5)

    debug.assert_called_once_with("Failure summary generation failed", exc_info=True)


@pytest.mark.parametrize("ready_before_exit", [False, True])
@pytest.mark.parametrize("returncode", [0, 5])
def test_parallel_readiness_detects_member_exit_before_group_ready(
    monkeypatch, tmp_path, ready_before_exit, returncode
):
    failed = Mock()
    failed.poll.side_effect = (
        [None, returncode] if ready_before_exit else [returncode]
    )
    failed.returncode = returncode
    dead_ready = tmp_path / "dead.ready"
    if ready_before_exit:
        dead_ready.touch()
    sibling = Mock()
    sibling.poll.return_value = None
    sibling_ready = tmp_path / "sibling.ready"
    waits = []

    def advance_sibling(_seconds):
        waits.append(True)
        sibling_ready.touch()

    monkeypatch.setattr(_stack.time, "sleep", advance_sibling)

    with pytest.raises(_stack._ReadinessFailure) as excinfo:
        _stack._wait_ready_parallel(
            [
                ("failed", dead_ready, failed),
                ("sibling", sibling_ready, sibling),
            ]
        )

    assert excinfo.value.name == "failed"
    assert excinfo.value.code == (1 if returncode == 0 else returncode)
    assert len(waits) == int(ready_before_exit)


def test_parallel_readiness_allows_ready_persist_wrapper_zero_exit(
    monkeypatch, tmp_path
):
    persist = Mock()
    persist.poll.return_value = 0
    persist.returncode = 0
    persist_ready = tmp_path / "persist.ready"
    persist_ready.touch()
    sibling = Mock()
    sibling.poll.return_value = None
    sibling_ready = tmp_path / "sibling.ready"
    monkeypatch.setattr(_stack.time, "sleep", lambda _seconds: sibling_ready.touch())

    _stack._wait_ready_parallel(
        [
            ("persist", persist_ready, persist),
            ("sibling", sibling_ready, sibling),
        ],
        {"persist"},
    )


def test_parallel_readiness_rejects_ready_persist_wrapper_nonzero_exit(tmp_path):
    persist = Mock()
    persist.poll.return_value = 6
    persist.returncode = 6
    persist_ready = tmp_path / "persist.ready"
    persist_ready.touch()

    with pytest.raises(_stack._ReadinessFailure) as excinfo:
        _stack._wait_ready_parallel(
            [("persist", persist_ready, persist)],
            {"persist"},
        )

    assert excinfo.value.name == "persist"
    assert excinfo.value.code == 6


def test_output_reader_drain_has_one_deadline_when_inherited_pipes_stay_open():
    pipes = [os.pipe(), os.pipe()]
    streams = [os.fdopen(read_fd, "rb") for read_fd, _ in pipes]
    readers = [
        threading.Thread(
            target=_stack._forward,
            args=(stream, "[test]"),
            daemon=True,
        )
        for stream in streams
    ]
    for reader in readers:
        reader.start()

    started = time.monotonic()
    try:
        _stack._drain_output_readers(readers, timeout=0.05)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert all(reader.is_alive() for reader in readers)
    finally:
        for _, write_fd in pipes:
            os.close(write_fd)
        for reader in readers:
            reader.join(timeout=1.0)
        for stream in streams:
            stream.close()


def test_output_reader_drain_shares_timeout_budget(monkeypatch):
    now = [10.0]
    join_budgets = []

    def join_first(timeout):
        join_budgets.append(timeout)
        now[0] += 0.04

    def join_second(timeout):
        join_budgets.append(timeout)

    readers = [Mock(join=Mock(side_effect=join_first)),
               Mock(join=Mock(side_effect=join_second))]
    monkeypatch.setattr(_stack.time, "monotonic", lambda: now[0])

    _stack._drain_output_readers(readers, timeout=0.05)

    assert join_budgets == pytest.approx([0.05, 0.01])


@pytest.mark.parametrize("error", [RuntimeError("join failed"), KeyboardInterrupt()])
def test_output_reader_drain_errors_do_not_escape(error):
    reader = Mock()
    reader.join.side_effect = error

    _stack._drain_output_readers([reader], timeout=0.05)


@pytest.mark.parametrize("returncode,status", [(0, 1), (-15, 143), (-9, 137)])
def test_failure_exit_status_is_nonzero_and_maps_signals(returncode, status):
    assert _failure.failure_exit_status(returncode) == status
