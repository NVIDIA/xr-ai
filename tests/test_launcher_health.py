# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live process-health reporting for applications using run_stack."""

import json
import socket
from http.client import HTTPConnection
from unittest.mock import Mock

import pytest
import xr_ai_launcher._health as health
import xr_ai_launcher._stack as stack


def response(port, path="/healthz", method="GET", headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request(method, path, headers=headers or {})
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
    with health.stack_health({"worker": process}, 0, ("model",)) as initialized:
        address, port = listener[0].server_address
        assert address == "127.0.0.1"
        assert response(port)[:2] == (503, "no-store")
        assert json.loads(response(port)[2]) == {
            "status": "not_ready", "processes": {"worker": "running"}, "unchecked_reuse": ["model"],
        }
        initialized.set()
        assert response(port)[:2] == (200, "no-store")
        assert json.loads(response(port)[2])["status"] == "ready"
        process.poll.return_value = 1
        assert response(port)[0] == 503
        assert json.loads(response(port)[2])["processes"] == {"worker": "exited"}
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
    monkeypatch.setattr(stack, "_wait_ready", Mock(side_effect=AssertionError("unexpected startup")))
    monkeypatch.setenv("XR_AI_STACK_HEALTH_PORT", "8093")
    return spawn, process


def test_stack_exposes_startup_ready_failure_and_shutdown_states(foreground, monkeypatch, tmp_path, listener):
    # Use an OS-assigned port while exercising the actual run_stack integration.
    original = stack.stack_health

    def bind(processes, port, unchecked_reuse):
        assert port == 8093
        assert unchecked_reuse == ("model",)
        return original(processes, 0, unchecked_reuse)

    monkeypatch.setattr(stack, "stack_health", bind)
    spawn, process = foreground

    def status():
        return response(listener[0].server_address[1])[0]

    def wait_ready(*_args):
        assert status() == 503

    def monitor(_processes, _initialized):
        assert status() == 200
        process.poll.return_value = 1
        assert status() == 503

    def shutdown(*_args, **_kwargs):
        process.poll.return_value = None
        assert status() == 503

    monkeypatch.setattr(stack, "_wait_ready", wait_ready)
    monkeypatch.setattr(stack, "_monitor", monitor)
    monkeypatch.setattr(stack, "_shutdown", shutdown)
    stack.run_stack([stack.Process("model", ".", "model", launch_mode="reuse"),
                     stack.Process("worker", ".", "worker")], tmp_path)
    spawn.assert_called_once()
    with pytest.raises(OSError):
        response(listener[0].server_address[1])


@pytest.mark.parametrize("port", ["0", "65536", "-1", "invalid", "+8093", " 8093", "8_093", "８０９３"])
@pytest.mark.parametrize("exit_after_ready", [False, True])
def test_invalid_health_port_fails_before_spawn(foreground, monkeypatch, tmp_path, port, exit_after_ready):
    monkeypatch.setenv("XR_AI_STACK_HEALTH_PORT", port)
    with pytest.raises(ValueError, match="XR_AI_STACK_HEALTH_PORT"):
        stack.run_stack([stack.Process("worker", ".", "worker")], tmp_path, exit_after_ready=exit_after_ready)
    foreground[0].assert_not_called()


def test_startup_failure_closes_health_listener(foreground, monkeypatch, tmp_path, listener):
    original = stack.stack_health

    def bind(processes, port, unchecked_reuse):
        assert port == 8093
        return original(processes, 0, unchecked_reuse)

    monkeypatch.setattr(stack, "stack_health", bind)

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
        with pytest.raises(OSError, match=f"XR_AI_STACK_HEALTH_PORT={occupied.getsockname()[1]}"):
            stack.run_stack([stack.Process("worker", ".", "worker")], tmp_path)
    foreground[0].assert_not_called()


def test_exit_after_ready_does_not_start_health(foreground, monkeypatch, tmp_path, listener):
    monkeypatch.setattr(stack, "_wait_ready", lambda *_args: None)
    monkeypatch.setattr(stack, "_shutdown", lambda *_args, **_kwargs: None)
    stack.run_stack([stack.Process("model", ".", "model", launch_mode="persist")],
                    tmp_path, exit_after_ready=True)
    assert listener == []


def test_idle_health_connection_is_bounded(listener):
    with health.stack_health({}, 0):
        server = listener[0]
        assert server.RequestHandlerClass.timeout == 2.0
        server.RequestHandlerClass.timeout = 0.05
        with socket.create_connection(server.server_address, timeout=1) as client:
            client.sendall(b"GET /healthz HTTP/1.1\r\n")
            assert client.recv(1) == b""


@pytest.mark.parametrize("error", [BrokenPipeError(), ConnectionResetError(), ValueError("unexpected")])
def test_disconnected_health_client_is_quiet(listener, monkeypatch, error):
    def disconnected(_self):
        raise error

    with health.stack_health({}, 0):
        monkeypatch.setattr(health.BaseHTTPRequestHandler, "handle", disconnected)
        handler = object.__new__(listener[0].RequestHandlerClass)
        if isinstance(error, ConnectionError):
            handler.handle()
        else:
            with pytest.raises(ValueError, match="unexpected"):
                handler.handle()


def test_health_routes_head_and_host_validation(listener, caplog):
    with caplog.at_level("INFO"), health.stack_health({}, 0) as initialized:
        port = listener[0].server_address[1]
        assert f"http://127.0.0.1:{port}/healthz" in caplog.text
        initialized.set()
        for path in ("/healthz", "/healthz/", "/healthz?probe=1"):
            assert response(port, path)[0] == 200
            assert response(port, path, method="HEAD") == (200, "no-store", b"")
        for host in ("attacker.example", "localhost:invalid", "user@localhost", "localhost/path"):
            assert response(port, headers={"Host": host})[0] == 400
        assert response(port, headers={"Host": "localhost"})[0] == 200
        initialized.clear()
        assert response(port, method="HEAD")[0] == 503


@pytest.mark.parametrize("sig", [stack.signal.SIGINT, stack.signal.SIGTERM])
def test_monitor_marks_unready_in_signal_handler(listener, monkeypatch, sig):
    handlers = {}
    monkeypatch.setattr(stack.signal, "getsignal", lambda _sig: None)
    monkeypatch.setattr(stack.signal, "signal", lambda number, handler: handlers.__setitem__(number, handler))
    process = Mock()
    process.poll.return_value = None
    with health.stack_health({"worker": process}, 0) as initialized:
        initialized.set()
        port = listener[0].server_address[1]

        def interrupt(_seconds):
            assert response(port)[0] == 200
            handlers[sig](sig, None)
            assert not initialized.is_set()
            assert response(port)[0] == 503

        monkeypatch.setattr(stack.time, "sleep", interrupt)
        with stack._Cancellation(initialized) as cancellation:
            assert stack._monitor({"worker": process}, cancellation) is None


@pytest.mark.parametrize("value", [None, ""])
def test_empty_health_port_disables_listener(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(health.HEALTH_PORT_ENV, raising=False)
    else:
        monkeypatch.setenv(health.HEALTH_PORT_ENV, value)
    assert health.health_port_from_env() is None
