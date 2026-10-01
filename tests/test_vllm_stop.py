# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety tests for persistent-server cleanup discovery."""
from __future__ import annotations

import signal
import time
from unittest.mock import MagicMock

import pytest
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


@pytest.mark.parametrize("denied_name", ["cmdline", "environ"])
def test_public_listener_helper_reports_unconfirmed_identity(
    monkeypatch, denied_name,
) -> None:
    process_file = MagicMock()
    process_file.read_text.return_value = "vllm\0serve\0model"
    process_file.read_bytes.return_value = (
        b"XR_AI_VLLM_MANAGED=1\0XR_AI_VLLM_PORT=8108\0"
    )
    getattr(process_file, "read_text" if denied_name == "cmdline" else "read_bytes").side_effect = PermissionError

    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "pid_on_port_checked",
        lambda _port: (1234, True, True),
    )
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "Path",
        lambda _path: process_file,
    )

    with pytest.raises(
        RuntimeError,
        match="port 8108.*pid 1234.*identity unconfirmed.*uninspectable.*inspect pid 1234",
    ):
        xr_ai_vllm.owned_listener_pid(8108, "vllm")


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


@pytest.mark.parametrize(
    ("command", "label", "port", "expected"),
    [
        ("/venv/bin/vllm\0serve\0model\0", "vllm", 8108, True),
        ("VLLM::APIServer\0", "vllm", 8108, True),
        ("VLLM::EngineCore\0", "vllm", 8108, False),
        ("/venv/bin/python\0-m\0pocket_tts_server\0--_serve\0", "tts", 8108, True),
        ("/venv/bin/python\0-m\0pocket_tts_server\0--_serve\0", "vllm", 8108, False),
        ("/venv/bin/vllm\0serve\0model\0", "tts", 8108, False),
        ("/venv/bin/vllm\0serve\0model\0", "vllm", 8107, False),
    ],
)
def test_service_identity_uses_command_and_managed_port(
    tmp_path, monkeypatch, command, label, port, expected,
) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "Path",
        lambda _path: proc_root / _path.rsplit("/", 1)[-1],
    )
    (proc_root / "cmdline").write_text(command)
    (proc_root / "environ").write_bytes(
        b"XR_AI_VLLM_MANAGED=1\0XR_AI_VLLM_PORT=8108\0"
    )
    assert xr_ai_vllm._docker.is_xr_ai_server_process(1234, label, port) is expected


def test_stt_ownership_requires_private_serve_mode(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "environ").write_bytes(b"PATH=/bin\0")
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "Path",
        lambda _path: proc_root / _path.rsplit("/", 1)[-1],
    )

    (proc_root / "cmdline").write_text("/venv/bin/stt_server\0")
    assert not xr_ai_vllm._docker.is_xr_ai_server_process(1234, "stt", 8106)

    (proc_root / "cmdline").write_text("/venv/bin/stt_server\0--_serve\0")
    assert xr_ai_vllm._docker.is_xr_ai_server_process(1234, "stt", 8106)


def test_unmarked_pocket_process_is_not_owned(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "cmdline").write_text(
        "uv\0run\0--project\0services/pocket-tts\0pocket_tts_server"
    )
    (proc_root / "environ").write_bytes(b"PATH=/bin\0")
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "Path",
        lambda _path: proc_root / _path.rsplit("/", 1)[-1],
    )

    assert not xr_ai_vllm._docker.is_xr_ai_server_process(1234, "tts", 8105)


def test_pocket_ownership_marker_matches_service_port(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "cmdline").write_text("python\0-m\0pocket_tts_server\0--_serve")
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


def test_pocket_process_group_requires_matching_owned_session(
    tmp_path, monkeypatch,
) -> None:
    proc_root = tmp_path / "proc" / "1234"
    proc_root.mkdir(parents=True)
    (proc_root / "environ").write_bytes(
        b"XR_AI_VLLM_MANAGED=1\0"
        b"XR_AI_VLLM_PORT=8105\0"
        b"_XR_AI_POCKET_PROCESS_GROUP=1234\0"
    )
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "Path",
        lambda _path: proc_root / _path.rsplit("/", 1)[-1],
    )
    monkeypatch.setattr(xr_ai_vllm._docker.os, "getpgid", lambda _pid: 1234)
    monkeypatch.setattr(xr_ai_vllm._docker.os, "getsid", lambda _pid: 1234)

    assert xr_ai_vllm._docker._pocket_owned_process_group(1234, 8105) == 1234

    monkeypatch.setattr(xr_ai_vllm._docker.os, "getsid", lambda _pid: 4321)
    assert xr_ai_vllm._docker._pocket_owned_process_group(1234, 8105) is None


def test_pocket_process_group_accepts_verified_launcher_session(
    tmp_path, monkeypatch,
) -> None:
    listener = tmp_path / "proc" / "1234"
    leader = tmp_path / "proc" / "4321"
    listener.mkdir(parents=True)
    leader.mkdir(parents=True)
    (listener / "environ").write_bytes(
        b"XR_AI_VLLM_MANAGED=1\0"
        b"XR_AI_VLLM_PORT=8105\0"
        b"_XR_AI_POCKET_PROCESS_GROUP=4321\0"
    )
    (leader / "environ").write_bytes(
        b"_XR_AI_LAUNCHER_PROCESS_GROUP_OWNER=pocket_tts_server\0"
    )
    monkeypatch.setattr(
        xr_ai_vllm._docker,
        "Path",
        lambda raw: tmp_path / str(raw).removeprefix("/"),
    )
    monkeypatch.setattr(xr_ai_vllm._docker.os, "getpgid", lambda _pid: 4321)
    monkeypatch.setattr(xr_ai_vllm._docker.os, "getsid", lambda _pid: 4321)

    assert xr_ai_vllm._docker._pocket_owned_process_group(1234, 8105) == 4321

    (leader / "environ").write_bytes(
        b"_XR_AI_LAUNCHER_PROCESS_GROUP_OWNER=other_server\0"
    )
    assert xr_ai_vllm._docker._pocket_owned_process_group(1234, 8105) is None


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
        "_pocket_owned_process_group",
        lambda *_args: 4321,
    )
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


def test_stop_keeps_unverified_pocket_process_pid_scoped(monkeypatch) -> None:
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
        "_pocket_owned_process_group",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        xr_ai_vllm.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("unverified Pocket TTS cleanup must remain PID-scoped")
        ),
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(xr_ai_vllm.os, "kill", kill)

    assert xr_ai_vllm.stop_persistent_servers([("tts", 8105)])
    assert calls == [(1234, signal.SIGTERM), (1234, 0)]


def test_pid_cleanup_waits_for_exit_after_sigkill(monkeypatch) -> None:
    signals: list[int] = []
    probes_after_sigkill = 0
    force_killed = False
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
        "_pocket_owned_process_group",
        lambda *_args: None,
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def kill(_pid: int, sig: int) -> None:
        nonlocal force_killed, probes_after_sigkill
        if sig == signal.SIGKILL:
            force_killed = True
            signals.append(sig)
            return
        if sig == signal.SIGTERM:
            signals.append(sig)
            return
        assert sig == 0
        if force_killed:
            probes_after_sigkill += 1
            if probes_after_sigkill == 3:
                raise ProcessLookupError

    monkeypatch.setattr(xr_ai_vllm.os, "kill", kill)

    assert xr_ai_vllm.stop_persistent_servers([("tts", 8105)])
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert probes_after_sigkill == 3


def test_stop_keeps_non_pocket_managed_process_pid_scoped(monkeypatch) -> None:
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
            AssertionError("non-Pocket TTS cleanup must remain PID-scoped")
        ),
    )
    monkeypatch.setattr(
        xr_ai_vllm.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("non-Pocket TTS cleanup must not signal a process group")
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
    (zombie / "stat").write_text("1234 (pocket worker) Z 1 4321 4321 0 0\n")

    assert not xr_ai_vllm._docker.process_group_alive(4321, proc_root)

    live = proc_root / "1235"
    live.mkdir()
    (live / "stat").write_text("1235 (pocket worker) S 1 4321 4321 0 0\n")

    assert xr_ai_vllm._docker.process_group_alive(4321, proc_root)
