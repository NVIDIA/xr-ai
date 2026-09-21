# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real adapter processes, port ownership, reuse, and --stop without GPUs."""
from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import grpc
import httpx
import nim_model_adapter.__main__ as launcher
import pytest
import yaml
from xr_ai_vllm._docker import has_xr_ai_ownership_marker, pid_on_port_checked

BASE = Path(__file__).resolve().parents[1] / "model-server-samples/model-servers-nim"
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

    def launch(overrides=None, *, ready=True, may_exit=False):
        number = len(processes)
        path = tmp_path / f"config-{number}.yaml"
        path.write_text(yaml.safe_dump(config | (overrides or {})))
        ready_file = tmp_path / f"ready-{number}"
        log = (tmp_path / f"process-{number}.log").open("w+")
        module = "magpie_nim_tts" if config.get("kind") == "tts" else "nim_model_adapter"
        env = os.environ.copy()
        if may_exit:
            env["_XR_AI_LAUNCHER_READY_PROCESS_MAY_EXIT"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-m", module, "--config", str(path),
             "--ready-file", str(ready_file)], stdout=log, stderr=log, start_new_session=True, env=env,
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
            if not may_exit:
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


@pytest.fixture
def grpc_backend():
    with ThreadPoolExecutor(max_workers=1) as pool:
        server = grpc.server(pool)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            yield f"127.0.0.1:{port}"
        finally:
            server.stop(0).wait()


@pytest.mark.parametrize("kind", ["embedding", "chat", "stt", "tts"])
def test_two_launches_reuse_one_listener_and_real_stop_allows_restart(
    servers, grpc_backend, monkeypatch, tmp_path, kind,
):
    config, _, launch = servers
    config.update(kind=kind, alias="llm" if kind == "chat" else "embed")
    if kind in ("stt", "tts"):
        config.update(health_url=config["base_url"], base_url=grpc_backend, language="en-US",
                      voice="Magpie-Multilingual.EN-US.Aria", sample_rate=44100)
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
    if kind == "stt":
        # Exercise the actual cross-profile target selection on the shared STT
        # port, while keeping all test listeners isolated from user services.
        for hardware, filename in (("spark", "stt_server.yaml"), ("96G_blackwell", "stt_adapter.yaml")):
            directory = tmp_path / "yaml" / hardware
            directory.mkdir(parents=True)
            (directory / filename).write_text(yaml.safe_dump({"port": port}))
        monkeypatch.setattr(sample, "_BASE", tmp_path)
        assert sample._known_ports() == [("stt-adapter", port)]
    else:
        label = "embedding" if kind == "embedding" else f"{kind}-adapter"
        monkeypatch.setattr(sample, "_known_ports", lambda: [(label, port)])
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



def test_persistent_reuse_exits_wrapper_and_preserves_listener(servers):
    config, _, launch = servers
    first, _ = launch()
    second, output = launch(may_exit=True)
    assert second.wait(timeout=5) == 0
    assert "reusing managed listener" in output
    assert pid_on_port_checked(config["port"]) == (first.pid, True, True)


@pytest.mark.parametrize("result", [(None, False, False), (None, True, True)])
async def test_reuse_fails_closed_when_ownership_cannot_be_inspected(monkeypatch, result):
    monkeypatch.setattr(launcher, "pid_on_port_checked", lambda port: result)
    with pytest.raises(RuntimeError, match="cannot inspect ownership"):
        await launcher._reusable_listener({"port": 8109})


async def test_reuse_rechecks_listener_after_health_probe(monkeypatch):
    config = {"port": 8109}
    probes = iter([(123, True, True), (456, True, True)])
    monkeypatch.setattr(launcher, "pid_on_port_checked", lambda port: next(probes))
    monkeypatch.setattr(launcher, "has_xr_ai_ownership_marker", lambda pid, port: True)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(launcher.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=launcher.identity(config))), **kwargs))
    with pytest.raises(RuntimeError, match="changed during inspection"):
        await launcher._reusable_listener(config)


async def test_reuse_monitor_survives_inconclusive_inspection(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    monkeypatch.delenv("_XR_AI_LAUNCHER_READY_PROCESS_MAY_EXIT", raising=False)
    monkeypatch.setattr(launcher, "_reusable_listener", AsyncMock(return_value=123))
    probes = iter([(None, False, False), (123, True, True), (None, True, False)])
    observed = []

    def inspect(port):
        result = next(probes)
        observed.append(result)
        return result

    monkeypatch.setattr(launcher, "pid_on_port_checked", inspect)
    ready = tmp_path / "ready"
    await asyncio.wait_for(launcher._serve({"port": 8109}, ready), 2)
    assert ready.exists() and len(observed) == 3
