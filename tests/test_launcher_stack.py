# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_launcher._stack data structures and pure helpers."""
from __future__ import annotations

import os

import pytest

import xr_ai_launcher._stack as _stack


class TestProcessDataclass:
    def test_defaults(self):
        p = _stack.Process("hub", "../../server-runtime", "xr_media_hub")
        assert p.name == "hub"
        assert p.project == "../../server-runtime"
        assert p.command == "xr_media_hub"
        assert p.config is None
        assert p.gpu is None
        assert p.launch_mode == "own"
        assert p.port is None

    def test_all_fields(self):
        p = _stack.Process(
            "vlm", "../../ai-services/vlm-server", "vlm_server",
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
        p = _stack.Process("hub", "../../server-runtime", "xr_media_hub")
        with pytest.raises((AttributeError, TypeError)):
            p.name = "other"  # type: ignore[misc]

    def test_reuse_launch_mode(self):
        p = _stack.Process("stt", "../../ai-services/stt-server", "stt_server",
                    launch_mode="reuse")
        assert p.launch_mode == "reuse"


class TestParallelDataclass:
    def test_stores_processes_as_tuple(self):
        p1 = _stack.Process("stt", "../../ai-services/stt-server", "stt_server")
        p2 = _stack.Process("tts", "../../ai-services/tts/piper", "piper_tts_server")
        group = _stack.Parallel([p1, p2])
        assert isinstance(group.processes, tuple)
        assert group.processes == (p1, p2)

    def test_accepts_single_process(self):
        p = _stack.Process("stt", "../../ai-services/stt-server", "stt_server")
        group = _stack.Parallel([p])
        assert len(group.processes) == 1

    def test_empty_parallel_ok(self):
        group = _stack.Parallel([])
        assert group.processes == ()

    def test_frozen_immutability(self):
        group = _stack.Parallel([])
        with pytest.raises((AttributeError, TypeError)):
            group.processes = ()  # type: ignore[misc]


class _FakePopen:
    """Minimal Popen stand-in: alive (poll()->None), spawned without subprocess."""
    def __init__(self, name: str) -> None:
        self.name = name

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
        # load_credentials reads real ~/.config/~/.cache — neutralize it.
        monkeypatch.setattr(_stack, "load_credentials", lambda: None)
        monkeypatch.setattr(
            _stack, "_spawn",
            lambda proc, base, ready_file: _FakePopen(proc.name),
        )
        monkeypatch.setattr(_stack, "_print_ready_banner", lambda names: None)

        calls: dict[str, object] = {}

        def _spy_shutdown(procs, no_kill=None):
            calls["procs"] = procs
            calls["no_kill"] = no_kill

        monkeypatch.setattr(_stack, "_shutdown", _spy_shutdown)
        return calls

    def test_abort_during_startup_kills_everything_including_persist(
        self, stub_stack, tmp_path, monkeypatch,
    ):
        # Simulate Ctrl-C while waiting for the (persist) server to come up.
        def _boom(name, ready_file, proc):
            raise KeyboardInterrupt

        monkeypatch.setattr(_stack, "_wait_ready", _boom)

        processes = [_stack.Process("vlm", "../../vlm", "vlm_server",
                             launch_mode="persist", port=8100)]

        with pytest.raises(SystemExit) as excinfo:
            _stack.run_stack(processes, tmp_path)

        assert excinfo.value.code == 130
        # Despite a persist process, abort must tear down EVERYTHING.
        assert stub_stack["no_kill"] == set()
        assert "vlm" in stub_stack["procs"]

    def test_clean_exit_after_ready_keeps_persist_alive(
        self, stub_stack, tmp_path, monkeypatch,
    ):
        # Ready file appears immediately; no interruption.
        monkeypatch.setattr(_stack, "_wait_ready", lambda name, rf, proc: None)

        processes = [_stack.Process("vlm", "../../vlm", "vlm_server",
                             launch_mode="persist", port=8100)]

        # exit_after_ready returns normally (no SystemExit) after readiness.
        _stack.run_stack(processes, tmp_path, exit_after_ready=True)

        # Clean exit preserves the persist set so the container outlives us.
        assert stub_stack["no_kill"] == {"vlm"}
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
