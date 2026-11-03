# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_launcher._stack data structures and pure helpers."""
from __future__ import annotations

import os
import signal
import socket
import subprocess
from unittest.mock import Mock

import pytest

import xr_ai_launcher._stack as _stack


def test_spawn_marks_the_launcher_owned_process_group(monkeypatch, tmp_path):
    spawned = Mock()
    spawned.stdout = None
    spawned.stderr = None
    popen = Mock(return_value=spawned)
    thread = Mock()
    monkeypatch.setattr(_stack.shutil, "which", lambda _command: "/usr/bin/uv")
    monkeypatch.setattr(_stack.subprocess, "Popen", popen)
    monkeypatch.setattr(_stack.threading, "Thread", Mock(return_value=thread))

    process = _stack.Process("tts", tmp_path, "pocket_tts_server")
    failure_contexts = {}
    result = _stack._spawn(
        process,
        tmp_path,
        tmp_path / "ready",
        failure_contexts=failure_contexts,
    )

    assert result is spawned
    assert failure_contexts["tts"].service == "tts"
    kwargs = popen.call_args.kwargs
    assert kwargs["start_new_session"] is True
    assert (
        kwargs["env"][_stack._PROCESS_GROUP_OWNER_ENV]
        == "pocket_tts_server"
    )
    assert _stack._READY_PROCESS_MAY_EXIT_ENV not in kwargs["env"]

    _stack._spawn(
        process,
        tmp_path,
        tmp_path / "ready",
        ready_process_may_exit=True,
    )
    assert (
        popen.call_args.kwargs["env"][_stack._READY_PROCESS_MAY_EXIT_ENV]
        == "1"
    )


class TestProcessDataclass:
    def test_defaults(self):
        p = _stack.Process("hub", "../../services/device-io-hub", "device_io_hub")
        assert p.name == "hub"
        assert p.project == "../../services/device-io-hub"
        assert p.command == "device_io_hub"
        assert p.config is None
        assert p.gpu is None
        assert p.launch_mode == "own"
        assert p.port is None

    def test_all_fields(self):
        p = _stack.Process(
            "vlm", "../../services/vlm-server", "vlm_server",
            config="yaml/vlm.yaml",
            gpu="0",
            launch_mode="persist",
            port=8100,
        )
        assert p.config == "yaml/vlm.yaml"
        assert p.gpu == "0"
        assert p.launch_mode == "persist"
        assert p.port == 8100

    def test_frozen_immutability(self):
        p = _stack.Process("hub", "../../services/device-io-hub", "device_io_hub")
        with pytest.raises((AttributeError, TypeError)):
            p.name = "other"  # type: ignore[misc]

    def test_reuse_launch_mode(self):
        p = _stack.Process("stt", "../../services/stt-server", "stt_server",
                    launch_mode="reuse")
        assert p.launch_mode == "reuse"


class TestParallelDataclass:
    def test_stores_processes_as_tuple(self):
        p1 = _stack.Process("stt", "../../services/stt-server", "stt_server")
        p2 = _stack.Process("tts", "../../services/pocket-tts", "pocket_tts_server")
        group = _stack.Parallel([p1, p2])
        assert isinstance(group.processes, tuple)
        assert group.processes == (p1, p2)

    def test_accepts_single_process(self):
        p = _stack.Process("stt", "../../services/stt-server", "stt_server")
        group = _stack.Parallel([p])
        assert len(group.processes) == 1

    def test_empty_parallel_ok(self):
        group = _stack.Parallel([])
        assert group.processes == ()

    def test_frozen_immutability(self):
        group = _stack.Parallel([])
        with pytest.raises((AttributeError, TypeError)):
            group.processes = ()  # type: ignore[misc]


@pytest.mark.parametrize("reap_delay", [0.25, None])
def test_shutdown_escalates_all_children_with_shared_budgets(monkeypatch, reap_delay):
    now = 0.0
    signals = []
    waits = []
    events = []

    class Child:
        def __init__(self, name, exit_at=None):
            self.name = name
            self.exit_at = exit_at
            self.reaped = False

        def poll(self):
            if self.exit_at is not None and now >= self.exit_at:
                self.reaped = True
                return 0
            return None

        def wait(self, timeout):
            nonlocal now
            waits.append((self.name, now, timeout))
            events.append(("wait", self.name))
            if self.exit_at is not None and self.exit_at <= now + timeout:
                now = max(now, self.exit_at)
                self.reaped = True
                return 0
            now += timeout
            raise subprocess.TimeoutExpired(self.name, timeout)

    def send_signal(child, sig):
        signals.append((child.name, sig, now))
        if sig == signal.SIGKILL:
            events.append(("kill", child.name))
            if reap_delay is not None:
                child.exit_at = now + reap_delay

    children = {"ordinary": Child("ordinary", exit_at=1.0)}
    children.update({name: Child(name) for name in ("first", "second", "third", "retained")})
    children["exited"] = Child("exited", exit_at=0.0)
    monkeypatch.setattr(_stack.time, "monotonic", lambda: now)
    monkeypatch.setattr(_stack, "_killpg", send_signal)

    _stack._shutdown(children, no_kill={"retained"})

    assert signals == [
        (name, sig, when)
        for sig, when, names in (
            (signal.SIGTERM, 0.0, ("ordinary", "first", "second", "third")),
            (signal.SIGKILL, _stack._STOP_TIMEOUT, ("first", "second", "third")),
        )
        for name in names
    ]
    assert events[:6] == [
        ("wait", "ordinary"), ("wait", "first"),
        ("kill", "first"), ("kill", "second"), ("kill", "third"),
        ("wait", "first"),
    ]
    assert children["ordinary"].reaped
    assert not children["retained"].reaped
    assert not any(name == "retained" for name, _, _ in waits)
    assert waits[:2] == [
        ("ordinary", 0.0, _stack._STOP_TIMEOUT),
        ("first", 1.0, _stack._STOP_TIMEOUT - 1.0),
    ]
    assert now == _stack._STOP_TIMEOUT + (reap_delay if reap_delay is not None else _stack._KILL_TIMEOUT)
    assert all(children[name].reaped == (reap_delay is not None) for name in ("first", "second", "third"))


class _FakePopen:
    """Minimal Popen stand-in: alive (poll()->None), spawned without subprocess."""
    def __init__(self, name: str, returncode: int | None = None) -> None:
        self.name = name
        self.returncode = returncode

    def poll(self):
        return None


class TestRunStackShutdownContract:
    """run_stack abort vs clean-exit shutdown semantics.

    Hermetic: no real subprocess, no docker, no credential I/O. We monkeypatch
    _spawn, _wait_ready*, _shutdown, and load_credentials so only the
    abort/clean-exit branch logic is exercised.
    """

    @pytest.fixture()
    def stub_stack(self, monkeypatch, tmp_path):
        original_handlers = {
            sig: _stack.signal.getsignal(sig)
            for sig in (_stack.signal.SIGINT, _stack.signal.SIGTERM)
        }
        # load_credentials reads real ~/.config/~/.cache — neutralize it.
        monkeypatch.setattr(_stack, "load_credentials", lambda: None)
        calls: dict[str, object] = {}
        spawn_options: dict[str, bool] = {}

        def _fake_spawn(
            proc,
            base,
            ready_file,
            *,
            ready_process_may_exit=False,
            failure_contexts=None,
            output_readers=None,
        ):
            spawn_options[proc.name] = ready_process_may_exit
            if failure_contexts is not None:
                failure_contexts[proc.name] = Mock(service=proc.name)
            if output_readers is not None:
                output_readers[proc.name] = ()
            return _FakePopen(proc.name)

        monkeypatch.setattr(_stack, "_spawn", _fake_spawn)
        monkeypatch.setattr(_stack, "_print_ready_banner", lambda names: None)
        calls["spawn_options"] = spawn_options

        def _spy_shutdown(procs, no_kill=None):
            calls["procs"] = procs
            calls["no_kill"] = no_kill

        monkeypatch.setattr(_stack, "_shutdown", _spy_shutdown)
        yield calls
        assert {
            sig: _stack.signal.getsignal(sig) for sig in original_handlers
        } == original_handlers

    def test_abort_during_startup_kills_unready_persist(
        self, stub_stack, tmp_path, monkeypatch, capsys,
    ):
        # Simulate Ctrl-C while waiting for the (persist) server to come up.
        def _boom(name, ready_file, proc, cancellation):
            raise KeyboardInterrupt

        monkeypatch.setattr(_stack, "_wait_ready", _boom)

        processes = [_stack.Process("vlm", "../../vlm", "vlm_server",
                             launch_mode="persist", port=8100)]

        with pytest.raises(SystemExit) as excinfo:
            _stack.run_stack(processes, tmp_path)

        assert excinfo.value.code == 130
        # Persistent processes that have not reached readiness are cleaned up.
        assert stub_stack["no_kill"] == set()
        assert "vlm" in stub_stack["procs"]
        assert stub_stack["spawn_options"] == {"vlm": False}
        assert "Process failure:" not in capsys.readouterr().err

    @pytest.mark.parametrize("parallel", [False, True])
    def test_spawn_failure_preserves_ready_persist(
        self, stub_stack, tmp_path, monkeypatch, parallel,
    ):
        spawn = _stack._spawn

        def _spawn(proc, base, ready_file, **kwargs):
            if proc.name == "later":
                raise OSError("could not spawn later service")
            child = spawn(proc, base, ready_file, **kwargs)
            if proc.name != "pending":
                ready_file.touch()
            return child

        monkeypatch.setattr(_stack, "_spawn", _spawn)
        ready = [
            _stack.Process("llm", ".", "llm", launch_mode="persist"),
            _stack.Process("vlm", ".", "vlm", launch_mode="persist"),
            _stack.Process("owned", ".", "owned"),
        ]
        pending = _stack.Process("pending", ".", "pending", launch_mode="persist")
        later = _stack.Process("later", ".", "later")
        processes = (
            [_stack.Parallel([*ready, pending, later])] if parallel
            else [*ready, _stack.Parallel([pending, later])]
        )
        with pytest.raises(OSError, match="could not spawn later service"):
            _stack.run_stack(processes, tmp_path, exit_after_ready=True)
        assert stub_stack["no_kill"] == {"llm", "vlm"}
        assert set(stub_stack["procs"]) == {"llm", "vlm", "owned", "pending"}

    def test_clean_exit_after_ready_keeps_persist_alive(
        self, stub_stack, tmp_path, monkeypatch, capsys,
    ):
        # Ready file appears immediately; no interruption.
        monkeypatch.setattr(_stack, "_wait_ready", lambda *args: None)

        processes = [
            _stack.Process(
                "vlm",
                "../../vlm",
                "vlm_server",
                launch_mode="persist",
                port=8100,
            ),
            _stack.Process("worker", "../../worker", "worker"),
        ]

        # exit_after_ready returns normally (no SystemExit) after readiness.
        _stack.run_stack(processes, tmp_path, exit_after_ready=True)

        # Clean exit preserves the persist set so the container outlives us.
        assert stub_stack["no_kill"] == {"vlm"}
        assert stub_stack["spawn_options"] == {"vlm": True, "worker": False}
        assert "Process failure:" not in capsys.readouterr().err

    def test_leaves_hugging_face_policy_to_model_services(
        self, stub_stack, tmp_path, monkeypatch,
    ):
        # Stacks that own no model must not probe the Hub or change the
        # Hugging Face environment their children inherit.
        monkeypatch.setattr(_stack, "_wait_ready", lambda *args: None)
        connect = Mock(side_effect=AssertionError("run_stack opened a socket"))
        monkeypatch.setattr(socket, "create_connection", connect)
        monkeypatch.setattr(socket.socket, "connect", connect)

        def hf_environment() -> dict[str, str]:
            return {
                key: value
                for key, value in os.environ.items()
                if key.startswith(("HF_", "HUGGINGFACE_", "TRANSFORMERS_"))
            }

        before = hf_environment()
        _stack.run_stack(
            [_stack.Process("worker", "../../worker", "worker")],
            tmp_path,
            exit_after_ready=True,
        )

        assert hf_environment() == before
        connect.assert_not_called()

    def test_parallel_exit_after_ready_allows_only_persist_zero_exit(
        self, stub_stack, tmp_path, monkeypatch,
    ):
        allowed_exits = []

        def _wait(group, ready_processes_may_exit, cancellation):
            allowed_exits.append(ready_processes_may_exit)

        monkeypatch.setattr(_stack, "_wait_ready_parallel", _wait)

        _stack.run_stack(
            [
                _stack.Parallel(
                    [
                        _stack.Process(
                            "vlm",
                            "../../vlm",
                            "vlm_server",
                            launch_mode="persist",
                        ),
                        _stack.Process("worker", "../../worker", "worker"),
                    ]
                )
            ],
            tmp_path,
            exit_after_ready=True,
        )

        assert allowed_exits == [{"vlm"}]
        assert stub_stack["spawn_options"] == {"vlm": True, "worker": False}

    def test_runtime_summary_interrupt_preserves_exit_and_persist_ownership(
        self, stub_stack, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(_stack, "_wait_ready", lambda *args: None)

        def _monitor(procs, cancellation):
            proc = procs["vlm"]
            proc.returncode = 7
            return "vlm", proc

        monkeypatch.setattr(_stack, "_monitor", _monitor)
        monkeypatch.setattr(
            _stack,
            "emit_failure_summary",
            Mock(side_effect=KeyboardInterrupt),
        )

        with pytest.raises(SystemExit) as excinfo:
            _stack.run_stack(
                [
                    _stack.Process(
                        "vlm",
                        "../../vlm",
                        "vlm_server",
                        launch_mode="persist",
                    )
                ],
                tmp_path,
            )

        assert excinfo.value.code == 7
        assert stub_stack["no_kill"] == {"vlm"}

    def test_unexpected_zero_runtime_exit_is_failure(
        self, stub_stack, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(_stack, "_wait_ready", lambda *args: None)

        def _monitor(procs, cancellation):
            proc = procs["worker"]
            proc.returncode = 0
            return "worker", proc

        monkeypatch.setattr(_stack, "_monitor", _monitor)
        monkeypatch.setattr(
            _stack,
            "emit_failure_summary",
            Mock(side_effect=RuntimeError("summary failed")),
        )
        monkeypatch.setattr(
            _stack.log,
            "debug",
            Mock(side_effect=RuntimeError("logging failed")),
        )

        with pytest.raises(SystemExit) as excinfo:
            _stack.run_stack(
                [_stack.Process("worker", "../../worker", "worker")],
                tmp_path,
            )

        assert excinfo.value.code == 1

    def test_readiness_summary_interrupt_preserves_signal_exit(
        self, stub_stack, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(
            _stack,
            "_wait_ready",
            lambda name, rf, proc, cancellation: (_ for _ in ()).throw(
                _stack._ReadinessFailure(name, -15)
            ),
        )
        monkeypatch.setattr(
            _stack,
            "emit_failure_summary",
            Mock(side_effect=KeyboardInterrupt),
        )

        with pytest.raises(SystemExit) as excinfo:
            _stack.run_stack(
                [
                    _stack.Process(
                        "vlm",
                        "../../vlm",
                        "vlm_server",
                        launch_mode="persist",
                    )
                ],
                tmp_path,
            )

        assert excinfo.value.code == 143
        assert stub_stack["no_kill"] == set()


def test_monitor_prefers_cancellation_over_same_tick_child_exit(monkeypatch):
    handlers = {}
    monkeypatch.setattr(_stack.signal, "getsignal", lambda _sig: None)
    monkeypatch.setattr(
        _stack.signal,
        "signal",
        lambda sig, handler: handlers.__setitem__(sig, handler),
    )

    class _ExitedDuringSignal:
        returncode = 9

        def poll(self):
            handlers[_stack.signal.SIGINT](_stack.signal.SIGINT, None)
            return self.returncode

    with _stack._Cancellation() as cancellation:
        assert _stack._monitor({"worker": _ExitedDuringSignal()}, cancellation) is None


class TestStripConflictingCudnn:
    """LD_LIBRARY_PATH sanitization so a host cuDNN can't shadow the venv one."""

    def _make_cudnn_dir(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        (d / "libcudnn.so.9").touch()
        return str(d)

    def test_none_and_empty_pass_through(self):
        assert _stack._strip_conflicting_cudnn(None) == (None, [])
        assert _stack._strip_conflicting_cudnn("") == ("", [])

    def test_no_cudnn_dirs_unchanged(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        value = os.pathsep.join([str(plain), "/usr/lib"])
        assert _stack._strip_conflicting_cudnn(value) == (value, [])

    def test_drops_only_the_cudnn_dir(self, tmp_path):
        cudnn = self._make_cudnn_dir(tmp_path, "cudnn")
        keep = tmp_path / "keep"
        keep.mkdir()
        value = os.pathsep.join([cudnn, str(keep)])
        cleaned, dropped = _stack._strip_conflicting_cudnn(value)
        assert cleaned == str(keep)
        assert dropped == [cudnn]

    def test_returns_none_when_only_cudnn_dir(self, tmp_path):
        cudnn = self._make_cudnn_dir(tmp_path, "cudnn")
        cleaned, dropped = _stack._strip_conflicting_cudnn(cudnn)
        assert cleaned is None
        assert dropped == [cudnn]

    def test_preserves_empty_cwd_entry(self, tmp_path):
        cudnn = self._make_cudnn_dir(tmp_path, "cudnn")
        # Leading "" => current-directory entry; must survive untouched.
        value = os.pathsep.join(["", cudnn, "/usr/lib"])
        cleaned, dropped = _stack._strip_conflicting_cudnn(value)
        assert cleaned == os.pathsep.join(["", "/usr/lib"])
        assert dropped == [cudnn]

    def test_matches_versioned_soname(self, tmp_path):
        # glob libcudnn.so* must catch libcudnn.so.9.13.1 etc.
        d = tmp_path / "lib"
        d.mkdir()
        (d / "libcudnn.so.9.13.1").touch()
        cleaned, dropped = _stack._strip_conflicting_cudnn(str(d))
        assert cleaned is None
        assert dropped == [str(d)]
