# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety tests for persistent-server cleanup discovery."""
from __future__ import annotations

import xr_ai_vllm


def test_stop_fails_closed_when_listener_discovery_fails(monkeypatch) -> None:
    monkeypatch.setattr(xr_ai_vllm._docker, "container_on_port_checked", lambda _port: (None, True))
    monkeypatch.setattr(xr_ai_vllm._docker, "pid_on_port_checked", lambda _port: (None, False))

    assert not xr_ai_vllm.stop_persistent_servers([("omni", 8108)])


def test_stop_does_not_signal_unidentified_listener(monkeypatch) -> None:
    monkeypatch.setattr(xr_ai_vllm._docker, "container_on_port_checked", lambda _port: (None, True))
    monkeypatch.setattr(xr_ai_vllm._docker, "pid_on_port_checked", lambda _port: (1234, True))
    monkeypatch.setattr(xr_ai_vllm._docker, "is_xr_ai_server_process", lambda *_args: False)

    assert not xr_ai_vllm.stop_persistent_servers([("omni", 8108)])
