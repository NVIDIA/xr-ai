# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety tests for persistent-server cleanup discovery."""
from __future__ import annotations

import signal
import time

import xr_ai_vllm


def test_stop_fails_closed_when_listener_discovery_fails(monkeypatch) -> None:
    monkeypatch.setattr(xr_ai_vllm._docker, "container_on_port_checked", lambda _port: (None, True))
    monkeypatch.setattr(xr_ai_vllm._docker, "pid_on_port_checked", lambda _port: (None, False, False))

    assert not xr_ai_vllm.stop_persistent_servers([("omni", 8108)])


def test_stop_does_not_signal_unidentified_listener(monkeypatch) -> None:
    monkeypatch.setattr(xr_ai_vllm._docker, "container_on_port_checked", lambda _port: (None, True))
    monkeypatch.setattr(xr_ai_vllm._docker, "pid_on_port_checked", lambda _port: (1234, True, True))
    monkeypatch.setattr(xr_ai_vllm._docker, "is_xr_ai_server_process", lambda *_args: False)

    assert not xr_ai_vllm.stop_persistent_servers([("omni", 8108)])


def test_stop_container_without_visible_host_pid(monkeypatch) -> None:
    stopped: list[str] = []
    monkeypatch.setattr(xr_ai_vllm._docker, "container_on_port_checked", lambda _port: ("omni", True))
    monkeypatch.setattr(xr_ai_vllm._docker, "pid_on_port_checked", lambda _port: (None, True, False))
    monkeypatch.setattr(xr_ai_vllm._docker, "stop_container", lambda name: stopped.append(name) or True)
    monkeypatch.setattr(xr_ai_vllm._docker, "remove_container", lambda _name: True)

    assert xr_ai_vllm.stop_persistent_servers([("omni", 8108)])
    assert stopped == ["omni"]


def test_stop_fails_closed_for_listener_without_visible_pid(monkeypatch) -> None:
    monkeypatch.setattr(xr_ai_vllm._docker, "container_on_port_checked", lambda _port: (None, True))
    monkeypatch.setattr(xr_ai_vllm._docker, "pid_on_port_checked", lambda _port: (None, True, True))

    assert not xr_ai_vllm.stop_persistent_servers([("omni", 8108)])


def test_stop_does_not_signal_external_vllm_process(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "cmdline").write_text("vllm\0serve\0external-model")
    (proc_root / "environ").write_bytes(b"PATH=/bin\0")
    monkeypatch.setattr(xr_ai_vllm._docker, "Path", lambda _path: proc_root / _path.rsplit("/", 1)[-1])
    monkeypatch.setattr(xr_ai_vllm._docker, "container_on_port_checked", lambda _port: (None, True))
    monkeypatch.setattr(xr_ai_vllm._docker, "pid_on_port_checked", lambda _port: (1234, True, True))
    monkeypatch.setattr(xr_ai_vllm.os, "kill", lambda *_args: (_ for _ in ()).throw(AssertionError()))

    assert not xr_ai_vllm.stop_persistent_servers([("omni", 8108)])


def test_pip_ownership_marker_matches_service_port(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "cmdline").write_text("vllm\0serve\0model")
    (proc_root / "environ").write_bytes(
        b"XR_AI_VLLM_MANAGED=1\0XR_AI_VLLM_PORT=8108\0"
    )
    monkeypatch.setattr(xr_ai_vllm._docker, "Path", lambda _path: proc_root / _path.rsplit("/", 1)[-1])

    assert xr_ai_vllm._docker.is_xr_ai_server_process(1234, "omni", 8108)
    assert not xr_ai_vllm._docker.is_xr_ai_server_process(1234, "omni", 8107)


def test_unmarked_piper_process_is_not_owned(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "cmdline").write_text(
        "uv\0run\0--project\0services/piper-tts\0piper_tts_server"
    )
    (proc_root / "environ").write_bytes(b"PATH=/bin\0")
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "Path",
        lambda _path: proc_root / _path.rsplit("/", 1)[-1],
    )

    assert not xr_ai_vllm._docker.is_xr_ai_server_process(1234, "tts", 8105)


def test_piper_ownership_marker_matches_service_port(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "cmdline").write_text("python\0-m\0piper_tts_server\0--_serve")
    (proc_root / "environ").write_bytes(
        b"XR_AI_VLLM_MANAGED=1\0XR_AI_VLLM_PORT=8105\0"
    )
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "Path",
        lambda _path: proc_root / _path.rsplit("/", 1)[-1],
    )

    assert xr_ai_vllm._docker.is_xr_ai_server_process(1234, "tts", 8105)
    assert not xr_ai_vllm._docker.is_xr_ai_server_process(1234, "tts", 8104)


def test_stop_signals_complete_managed_process_group(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "container_on_port_checked",
        lambda _port: (None, True),
    )
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "pid_on_port_checked",
        lambda _port: (1234, True, True),
    )
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "is_xr_ai_server_process",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "has_xr_ai_ownership_marker",
        lambda *_args: True,
    )
    monkeypatch.setattr(xr_ai_vllm.os, "getpgid", lambda _pid: 4321)
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "process_group_alive",
        lambda _pgid: False,
    )
    monkeypatch.setattr(
        xr_ai_vllm.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must signal group")),
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def killpg(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))

    monkeypatch.setattr(xr_ai_vllm.os, "killpg", killpg)

    assert xr_ai_vllm.stop_persistent_servers([("tts", 8105)])
    assert calls == [(4321, signal.SIGTERM)]


def test_stop_keeps_non_piper_managed_process_pid_scoped(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "container_on_port_checked",
        lambda _port: (None, True),
    )
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "pid_on_port_checked",
        lambda _port: (1234, True, True),
    )
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "is_xr_ai_server_process",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        xr_ai_vllm.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(
            AssertionError("non-Piper cleanup must remain PID-scoped")
        ),
    )
    monkeypatch.setattr(
        xr_ai_vllm.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("non-Piper cleanup must not signal a process group")
        ),
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(xr_ai_vllm.os, "kill", kill)

    assert xr_ai_vllm.stop_persistent_servers([("omni", 8108)])
    assert calls == [(1234, signal.SIGTERM), (1234, 0)]


def test_process_group_liveness_ignores_zombies(tmp_path) -> None:
    proc_root = tmp_path / "proc"
    zombie = proc_root / "1234"
    zombie.mkdir(parents=True)
    (zombie / "stat").write_text("1234 (piper worker) Z 1 4321 4321 0 0\n")

    assert not xr_ai_vllm._docker.process_group_alive(4321, proc_root)

    live = proc_root / "1235"
    live.mkdir()
    (live / "stat").write_text("1235 (piper worker) S 1 4321 4321 0 0\n")

    assert xr_ai_vllm._docker.process_group_alive(4321, proc_root)
