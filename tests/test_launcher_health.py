# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live orchestrator health checks, without models, Docker, or readiness files."""

import os
import socket
import subprocess
import sys
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import Mock

import pytest
import xr_ai_launcher._health as health
import xr_ai_launcher._stack as stack


def response(port, path="/health"):
    connection = HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", path)
        result = connection.getresponse()
        return result.status, result.getheader("Cache-Control"), result.read()
    finally:
        connection.close()


@pytest.fixture
def listener(monkeypatch):
    servers = []
    original = health.ThreadingHTTPServer

    def create(address, handler):
        server = original(address, handler)
        servers.append(server)
        return server

    monkeypatch.setattr(health, "ThreadingHTTPServer", create)
    return servers


def test_health_is_live_and_bound_to_loopback(listener):
    process = Mock()
    process.poll.return_value = None
    with health.stack_health({"worker": process}, 0) as initialized:
        address, port = listener[0].server_address
        assert address == "127.0.0.1"
        assert response(port) == (503, "no-store", b"not ready\n")
        initialized.set()
        assert response(port) == (200, "no-store", b"ready\n")
        process.poll.return_value = 1
        assert response(port)[0] == 503
        assert response(port, "/metrics")[0] == 404
        initialized.clear()
        assert response(port)[0] == 503
    with pytest.raises(OSError):
        response(port)


def test_disabled_health_does_not_open_listener(listener):
    with health.stack_health({}, None) as initialized:
        initialized.set()
    assert listener == []


@pytest.fixture
def foreground(monkeypatch):
    monkeypatch.setattr(stack, "load_credentials", lambda: None)
    process = Mock()
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr(stack, "_spawn", spawn)
    monkeypatch.setattr(stack, "_print_ready_banner", lambda *_args: None)
    monkeypatch.setenv("XR_AI_STACK_HEALTH_PORT", "8093")
    return spawn, process


def test_stack_exposes_startup_ready_failure_and_shutdown_states(foreground, monkeypatch, tmp_path, listener):
    # Use an OS-assigned port while exercising the actual run_stack integration.
    original = stack.stack_health
    monkeypatch.setattr(stack, "stack_health", lambda processes, _port: original(processes, 0))
    spawn, process = foreground

    def status():
        return response(listener[0].server_address[1])[0]

    def wait_ready(*_args):
        assert status() == 503

    def monitor(_processes):
        assert status() == 200
        process.poll.return_value = 1
        assert status() == 503

    def shutdown(*_args, **_kwargs):
        process.poll.return_value = None
        assert status() == 503

    monkeypatch.setattr(stack, "_wait_ready", wait_ready)
    monkeypatch.setattr(stack, "_monitor", monitor)
    monkeypatch.setattr(stack, "_shutdown", shutdown)
    stack.run_stack([stack.Process("worker", ".", "worker")], tmp_path)
    spawn.assert_called_once()
    with pytest.raises(OSError):
        response(listener[0].server_address[1])


@pytest.mark.parametrize("port", ["0", "65536", "-1", "invalid"])
def test_invalid_health_port_fails_before_spawn(foreground, monkeypatch, tmp_path, port):
    monkeypatch.setenv("XR_AI_STACK_HEALTH_PORT", port)
    with pytest.raises(ValueError, match="XR_AI_STACK_HEALTH_PORT"):
        stack.run_stack([stack.Process("worker", ".", "worker")], tmp_path)
    foreground[0].assert_not_called()


def test_startup_failure_closes_health_listener(foreground, monkeypatch, tmp_path, listener):
    original = stack.stack_health
    monkeypatch.setattr(stack, "stack_health", lambda processes, _port: original(processes, 0))

    def fail(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(stack, "_wait_ready", fail)
    monkeypatch.setattr(stack, "_shutdown", lambda *_args, **_kwargs: None)
    with pytest.raises(SystemExit) as failure:
        stack.run_stack([stack.Process("worker", ".", "worker")], tmp_path)
    assert failure.value.code == 130
    with pytest.raises(OSError):
        response(listener[0].server_address[1])


def test_occupied_health_port_fails_before_spawn(foreground, monkeypatch, tmp_path):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        monkeypatch.setenv("XR_AI_STACK_HEALTH_PORT", str(occupied.getsockname()[1]))
        with pytest.raises(OSError):
            stack.run_stack([stack.Process("worker", ".", "worker")], tmp_path)
    foreground[0].assert_not_called()


def test_exit_after_ready_does_not_start_health(foreground, monkeypatch, tmp_path, listener):
    monkeypatch.setattr(stack, "_wait_ready", lambda *_args: None)
    monkeypatch.setattr(stack, "_shutdown", lambda *_args, **_kwargs: None)
    stack.run_stack([stack.Process("model", ".", "model", launch_mode="persist")],
                    tmp_path, exit_after_ready=True)
    assert listener == []


def test_forced_orchestrator_death_closes_endpoint():
    # Print the actual bound port without a shared file or a free-port reservation.
    code = '''import time
import xr_ai_launcher._health as health
original = health.ThreadingHTTPServer
def server(address, handler):
    result = original(address, handler)
    print(result.server_address[1], flush=True)
    return result
health.ThreadingHTTPServer = server
with health.stack_health({}, 0) as initialized:
    initialized.set()
    print("ready", flush=True)
    time.sleep(30)
'''
    package = Path(stack.__file__).resolve().parent.parent
    with subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                          env=os.environ | {"PYTHONPATH": str(package)}) as child:
        try:
            port = int(child.stdout.readline())
            assert child.stdout.readline().strip() == "ready"
            assert response(port)[0] == 200
            child.kill()
            child.wait(timeout=5)
            with pytest.raises(OSError):
                response(port)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
