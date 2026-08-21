# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety tests for persistent-server cleanup discovery."""
from __future__ import annotations

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


def test_in_process_service_command_identifies_shared_piper(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "cmdline").write_text(
        "uv\0run\0--project\0services/piper-tts\0piper_tts_server"
    )
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "Path",
        lambda _path: proc_root / _path.rsplit("/", 1)[-1],
    )

    assert xr_ai_vllm._docker.is_xr_ai_server_process(1234, "tts", 8105)
    assert not xr_ai_vllm._docker.is_xr_ai_server_process(1234, "stt", 8105)
