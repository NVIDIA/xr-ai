# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real adapter processes, port ownership, reuse, and --stop without GPUs."""
from __future__ import annotations

import importlib.util
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import yaml
from xr_ai_vllm._docker import has_xr_ai_ownership_marker, pid_on_port_checked

BASE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("nim_lifecycle_sample", BASE / "main.py")
assert SPEC and SPEC.loader
sample = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sample)


@pytest.fixture
def servers(tmp_path, monkeypatch):
    if sys.platform != "linux" or not (ss := shutil.which("ss")):
        pytest.skip("Linux /proc and ss are required for real process ownership checks")
    # Exercise the real no-Docker cleanup path; never inspect users' containers.
    commands = tmp_path / "bin"
    commands.mkdir()
    (commands / "ss").symlink_to(ss)
    monkeypatch.setenv("PATH", str(commands))
    state = {"healthy": True}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if state["healthy"] else 503)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    config = {"host": "127.0.0.1", "port": port, "model": "nvidia/test",
              "base_url": f"http://127.0.0.1:{upstream.server_port}"}
    processes = []

    def launch(overrides=None, *, ready=True):
        number = len(processes)
        path = tmp_path / f"config-{number}.yaml"
        path.write_text(yaml.safe_dump(config | (overrides or {})))
        ready_file = tmp_path / f"ready-{number}"
        log = (tmp_path / f"process-{number}.log").open("w+")
        process = subprocess.Popen(
            [sys.executable, "-m", "nim_embedding_adapter", "--config", str(path),
             "--ready-file", str(ready_file)], stdout=log, stderr=log, start_new_session=True,
        )
        # Reap independently while --stop waits for /proc/<pid> to disappear.
        reaper = threading.Thread(target=process.wait, daemon=True)
        reaper.start()
        processes.append((process, reaper, log))
        deadline = time.monotonic() + 15
        while ready and not ready_file.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if ready:
            log.seek(0)
            assert ready_file.exists(), log.read()
            assert process.poll() is None
        else:
            reaper.join(15)
            assert not reaper.is_alive(), "conflicting adapter did not fail promptly"
            assert process.returncode != 0
            assert not ready_file.exists()
        log.seek(0)
        return process, log.read()

    try:
        yield config, state, launch
    finally:
        for process, reaper, log in reversed(processes):
            if process.poll() is None:
                process.terminate()
            reaper.join(5)
            if reaper.is_alive():
                process.kill()
                reaper.join(5)
            log.close()
        upstream.shutdown()
        upstream.server_close()
        worker.join(5)


def test_two_launches_reuse_one_listener_and_real_stop_allows_restart(servers, monkeypatch):
    config, _, launch = servers
    first, _ = launch()
    port = config["port"]
    assert pid_on_port_checked(port) == (first.pid, True, True)
    assert has_xr_ai_ownership_marker(first.pid, port)
    second, output = launch()
    assert "reusing managed listener" in output
    assert pid_on_port_checked(port) == (first.pid, True, True)
    assert second.poll() is None

    # Only restrict the CLI's target list; discovery, /proc verification, signals,
    # readiness, and process exit are real. No shared model ports are touched.
    monkeypatch.setattr(sample, "_known_ports", lambda: [("embedding", port)])
    monkeypatch.setattr(sys, "argv", ["model_servers_nim", "--stop"])
    sample.run()
    second.wait(timeout=5)
    assert first.poll() in (0, -signal.SIGTERM)
    assert second.returncode == 0
    assert pid_on_port_checked(port) == (None, True, False)

    restarted, _ = launch()
    assert restarted.pid != first.pid
    assert pid_on_port_checked(port) == (restarted.pid, True, True)
    sample.run()
    assert restarted.poll() in (0, -signal.SIGTERM)


@pytest.mark.parametrize("change", ["model", "base_url", "unhealthy"])
def test_existing_adapter_must_be_healthy_and_match_configuration(servers, change):
    config, state, launch = servers
    first, _ = launch()
    overrides = {}
    if change == "unhealthy":
        state["healthy"] = False
    else:
        overrides[change] = config[change] + "-changed"
    _, output = launch(overrides, ready=False)
    assert ("unhealthy" if change == "unhealthy" else "different adapter configuration") in output
    assert pid_on_port_checked(config["port"]) == (first.pid, True, True)
    assert first.poll() is None


def test_unmanaged_healthy_listener_is_neither_reused_nor_stopped(servers):
    config, _, launch = servers
    upstream_port = int(config["base_url"].rsplit(":", 1)[1])
    _, output = launch({"port": upstream_port}, ready=False)
    assert "unmanaged listener" in output
    with httpx.Client(trust_env=False) as client:
        assert client.get(config["base_url"]).status_code == 200


def test_unhealthy_backend_does_not_leave_a_listener_after_failed_start(servers):
    config, state, launch = servers
    state["healthy"] = False
    _, output = launch(ready=False)
    assert "unhealthy" in output
    assert pid_on_port_checked(config["port"]) == (None, True, False)
