# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure-summary tests for the stack launcher."""
from __future__ import annotations

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


@pytest.mark.integration
def test_real_readiness_failure_forwards_output_and_prints_summary(
    monkeypatch, tmp_path, capsys
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
    monkeypatch, tmp_path, capsys
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


def test_parallel_readiness_failure_releases_other_waiters(tmp_path):
    dead = Mock()
    dead.poll.return_value = 5
    dead.returncode = 5
    alive = Mock()
    alive.poll.return_value = None
    alive_ready = tmp_path / "alive.ready"
    failures: list[BaseException] = []

    def wait() -> None:
        try:
            _stack._wait_ready_parallel(
                [
                    ("alive", alive_ready, alive),
                    ("dead", tmp_path / "dead.ready", dead),
                ]
            )
        except BaseException as exc:
            failures.append(exc)

    started = time.monotonic()
    waiter = threading.Thread(target=wait, daemon=True)
    waiter.start()
    waiter.join(timeout=2.0)
    released = not waiter.is_alive()
    if not released:
        alive_ready.touch()
        waiter.join(timeout=2.0)

    assert not waiter.is_alive(), "readiness waiter did not stop during test cleanup"
    assert released, "parallel readiness did not release the live sibling"
    assert len(failures) == 1
    assert isinstance(failures[0], SystemExit)
    assert failures[0].code == 5
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize("returncode,status", [(0, 1), (-15, 143), (-9, 137)])
def test_failure_exit_status_is_nonzero_and_maps_signals(returncode, status):
    assert _failure.failure_exit_status(returncode) == status
