# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xr_ai_launcher._stack data structures and pure helpers."""
from __future__ import annotations

import pytest

import xr_ai_launcher._stack as _stack
from xr_ai_launcher._stack import Parallel, Process, run_stack


class TestProcessDataclass:
    def test_defaults(self):
        p = Process("hub", "../../server-runtime", "xr_media_hub")
        assert p.name == "hub"
        assert p.project == "../../server-runtime"
        assert p.command == "xr_media_hub"
        assert p.config is None
        assert p.gpu is None
        assert p.launch_mode == "own"
        assert p.port is None

    def test_all_fields(self):
        p = Process(
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
        p = Process("hub", "../../server-runtime", "xr_media_hub")
        with pytest.raises((AttributeError, TypeError)):
            p.name = "other"  # type: ignore[misc]

    def test_reuse_launch_mode(self):
        p = Process("stt", "../../ai-services/stt-server", "stt_server",
                    launch_mode="reuse")
        assert p.launch_mode == "reuse"


class TestParallelDataclass:
    def test_stores_processes_as_tuple(self):
        p1 = Process("stt", "../../ai-services/stt-server", "stt_server")
        p2 = Process("tts", "../../ai-services/tts/piper", "piper_tts_server")
        group = Parallel([p1, p2])
        assert isinstance(group.processes, tuple)
        assert group.processes == (p1, p2)

    def test_accepts_single_process(self):
        p = Process("stt", "../../ai-services/stt-server", "stt_server")
        group = Parallel([p])
        assert len(group.processes) == 1

    def test_empty_parallel_ok(self):
        group = Parallel([])
        assert group.processes == ()

    def test_frozen_immutability(self):
        group = Parallel([])
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

        processes = [Process("vlm", "../../vlm", "vlm_server",
                             launch_mode="persist", port=8100)]

        with pytest.raises(SystemExit) as excinfo:
            run_stack(processes, tmp_path)

        assert excinfo.value.code == 130
        # Despite a persist process, abort must tear down EVERYTHING.
        assert stub_stack["no_kill"] == set()
        assert "vlm" in stub_stack["procs"]

    def test_clean_exit_after_ready_keeps_persist_alive(
        self, stub_stack, tmp_path, monkeypatch,
    ):
        # Ready file appears immediately; no interruption.
        monkeypatch.setattr(_stack, "_wait_ready", lambda name, rf, proc: None)

        processes = [Process("vlm", "../../vlm", "vlm_server",
                             launch_mode="persist", port=8100)]

        # exit_after_ready returns normally (no SystemExit) after readiness.
        run_stack(processes, tmp_path, exit_after_ready=True)

        # Clean exit preserves the persist set so the container outlives us.
        assert stub_stack["no_kill"] == {"vlm"}
