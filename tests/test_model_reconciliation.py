# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile only recorded, compatible model instances before starting replacements."""

import json
import os
import select
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest
from xr_ai_model_server_stack import _reconcile, _stack, run_model_stack
from xr_ai_vllm import _identity, persistent_server_identity


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    directory = tmp_path / "deployment"
    directory.mkdir(mode=0o700)
    path = directory / "models.json"
    live, stopped = {}, []
    monkeypatch.setattr(_reconcile, "persistent_server_identity", lambda name, port: live.get(port))

    def stop(targets):
        for name, port in targets:
            stopped.append((name, port))
            live.pop(port, None)
        return True

    monkeypatch.setattr(_reconcile, "stop_persistent_servers", stop)
    monkeypatch.setattr(_reconcile.subprocess, "check_output", lambda *_a, **_k: "revision")

    @contextmanager
    def healthy(*_a, **_kw):
        yield Mock(status=200)

    monkeypatch.setattr(_reconcile, "build_opener", lambda *_a: Mock(open=healthy))
    return path, live, stopped


def launch(path, live, configs, *, ready=True):
    with _reconcile.model_state(path) as state:
        state.prepare(configs, Path("/checkout"))
        for name, cfg in configs.items():
            live.setdefault(cfg["port"], f"instance-{name}-{len(live)}")
        if ready:
            state.ready()


def test_switch_keeps_shared_models_and_prunes_unused_before_replacement(deployment):
    path, live, stopped = deployment
    launch(path, live, {"tts": {"port": 8105}, "vlm": {"port": 8100}})
    tts = live[8105]
    launch(path, live, {"tts": {"port": 8105}, "omni": {"port": 8108}})
    assert live[8105] == tts
    assert stopped == [("vlm", 8100)]
    assert 8100 not in live
    with _reconcile.model_state(path) as state:
        state.stop()
    assert live == {}
    assert json.loads(path.read_text()) == {}


@pytest.mark.parametrize(
    "change", ["model", "cuda_visible_devices", "port", "token", "revision", "unhealthy", "missing"]
)
def test_changed_or_unhealthy_models_are_replaced(deployment, monkeypatch, change):
    path, live, stopped = deployment
    original = {"vlm": {"port": 8100, "model": "first", "cuda_visible_devices": "0"}}
    launch(path, live, original)
    desired = {name: dict(cfg) for name, cfg in original.items()}
    if change == "model":
        desired["vlm"]["model"] = "second"
    elif change == "cuda_visible_devices":
        desired["vlm"]["cuda_visible_devices"] = "1"
    elif change == "port":
        desired["vlm"]["port"] = 9000
    elif change == "token":
        monkeypatch.setenv("HF_TOKEN", "hf_private_new_value")
    elif change == "revision":
        monkeypatch.setattr(_reconcile.subprocess, "check_output", lambda *_a, **_kw: "new-revision")
    elif change == "unhealthy":
        monkeypatch.setattr(_reconcile, "build_opener", lambda *_a: Mock(open=Mock(side_effect=URLError("offline"))))
    else:
        live.clear()
    launch(path, live, desired)
    assert stopped == [("vlm", 8100)]
    assert "hf_private_new_value" not in path.read_text()
    assert "first" not in path.read_text()


def test_changed_instance_and_untracked_servers_are_never_adopted_or_stopped(deployment):
    path, live, stopped = deployment
    live[8100] = "foreign"
    with pytest.raises(RuntimeError, match="untracked"):
        launch(path, live, {"vlm": {"port": 8100}})
    live.clear()
    launch(path, live, {"vlm": {"port": 8100}})
    live[8100] = "replacement"
    for operation in ("prepare", "stop"):
        with _reconcile.model_state(path) as state, pytest.raises(RuntimeError, match="instance changed"):
            if operation == "prepare":
                state.prepare({"vlm": {"port": 8100}}, Path("/checkout"))
            else:
                state.stop()
    assert stopped == []


def test_partial_launch_requires_explicit_cleanup_and_remains_stoppable(deployment):
    path, live, stopped = deployment
    configs = {"tts": {"port": 8105}, "vlm": {"port": 8100}}
    launch(path, live, configs, ready=False)
    with pytest.raises(RuntimeError, match="unfinished launch"):
        launch(path, live, configs)
    with _reconcile.model_state(path) as state:
        state.stop()
    assert stopped == [("tts", 8105), ("vlm", 8100)]
    assert not live


def test_replacement_failure_keeps_record_and_never_launches(deployment, monkeypatch):
    path, live, _ = deployment
    launch(path, live, {"vlm": {"port": 8100}})
    before = path.read_text()
    monkeypatch.setattr(_reconcile, "stop_persistent_servers", lambda *_a: False)
    with pytest.raises(RuntimeError, match="replacement failed"):
        launch(path, live, {"omni": {"port": 8108}})
    assert path.read_text() == before
    assert 8108 not in live


def test_private_state_lock_and_validation_fail_closed(deployment, tmp_path):
    path, _, _ = deployment
    with _reconcile.model_state(path):
        with pytest.raises(RuntimeError, match="in progress"):
            with _reconcile.model_state(path):
                pytest.fail("lock acquired twice")
    for text in ("not json", '{"vlm":{"port":8100}}'):
        path.write_text(text)
        with pytest.raises(RuntimeError, match="invalid model state"):
            with _reconcile.model_state(path):
                pytest.fail("invalid state accepted")
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="private regular file"):
        with _reconcile.model_state(path):
            pytest.fail("public state accepted")
    path.unlink()
    target = tmp_path / "untouched"
    target.write_text("secret")
    path.symlink_to(target)
    with pytest.raises(RuntimeError, match="cannot open private"):
        with _reconcile.model_state(path):
            pytest.fail("symlink followed")
    assert target.read_text() == "secret"


def test_stop_with_state_needs_no_gpu_or_sample_config(deployment, monkeypatch):
    path, live, _ = deployment
    launch(path, live, {"tts": {"port": 9000}})
    monkeypatch.setattr(_stack, "setup_logging", lambda *_a, **_kw: path.parent / "orchestrator.log")
    monkeypatch.setattr(_stack, "detect_gpu_config", lambda: pytest.fail("stop queried GPU"))
    monkeypatch.setattr(_stack, "_configs", lambda *_a: pytest.fail("stop read changed configs"))
    monkeypatch.setattr(_stack, "require_credentials", lambda *_a: pytest.fail("stop requested token"))
    run_model_stack(path.parent, (), stop=True, state_file=path)
    assert not live


def test_public_identity_distinguishes_container_restarts_and_inspection_failures(monkeypatch):
    monkeypatch.setattr(_identity._docker, "container_on_port_checked", lambda *_a: ("owned", True))
    monkeypatch.setattr(_identity.subprocess, "check_output", lambda *_a, **_kw: "id:started-once")
    first = persistent_server_identity("vlm", 8100)
    monkeypatch.setattr(_identity.subprocess, "check_output", lambda *_a, **_kw: "id:started-again")
    assert persistent_server_identity("vlm", 8100) != first
    monkeypatch.setattr(_identity._docker, "container_on_port_checked", lambda *_a: (None, False))
    with pytest.raises(RuntimeError, match="ownership"):
        persistent_server_identity("vlm", 8100)


@pytest.mark.parametrize(
    "label,port,error",
    [(None, 8100, TypeError), ("", 8100, ValueError), ("vlm", True, TypeError), ("vlm", 0, ValueError)],
)
def test_public_identity_validates_arguments_before_discovery(monkeypatch, label, port, error):
    monkeypatch.setattr(_identity._docker, "container_on_port_checked", lambda *_a: pytest.fail("queried ownership"))
    with pytest.raises(error):
        persistent_server_identity(label, port)


def test_model_state_argument_is_validated_before_logging(monkeypatch):
    monkeypatch.setattr(_stack, "setup_logging", lambda *_a, **_kw: pytest.fail("created logs"))
    with pytest.raises(TypeError, match="state_file"):
        run_model_stack(Path("/checkout"), (), state_file="not a Path")


def test_pip_backend_is_rejected_before_recording_or_starting_models(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    monkeypatch.setattr(_stack, "setup_logging", lambda *_a, **_kw: tmp_path / "orchestrator.log")
    monkeypatch.setattr(_stack, "_configs", lambda *_a: {"vlm": {"vllm_backend": "pip", "port": 8100}})
    monkeypatch.setattr(_stack, "require_credentials", lambda *_a: pytest.fail("asked for token"))
    monkeypatch.setattr(_stack, "_launch", lambda *_a: pytest.fail("launched models"))
    with pytest.raises(ValueError, match="requires Docker"):
        run_model_stack(tmp_path, ("vlm",), gpu_profile="spark", state_file=path)
    assert not path.exists()


def test_public_identity_handles_absent_unowned_and_recycled_pids(monkeypatch):
    monkeypatch.setattr(_identity._docker, "container_on_port_checked", lambda *_a: (None, True))
    monkeypatch.setattr(_identity._docker, "pid_on_port_checked", lambda *_a: (None, True, False))
    assert persistent_server_identity("tts", 8105) is None
    monkeypatch.setattr(_identity._docker, "pid_on_port_checked", lambda *_a: (123, True, True))
    monkeypatch.setattr(_identity._docker, "is_xr_ai_server_process", lambda *_a: False)
    with pytest.raises(RuntimeError, match="unmanaged"):
        persistent_server_identity("tts", 8105)
    monkeypatch.setattr(_identity._docker, "is_xr_ai_server_process", lambda *_a: True)

    def contents(path):
        return (
            "boot"
            if path.name == "boot_id"
            else "123 (process with ) spaces) " + " ".join(["S"] + ["0"] * 18 + ["456"])
        )

    monkeypatch.setattr(Path, "read_text", contents)
    assert persistent_server_identity("tts", 8105) == "process:boot:123:456"
    monkeypatch.setattr(
        Path, "read_text", lambda path: "new-boot" if path.name == "boot_id" else contents(path).replace("456", "789")
    )
    assert persistent_server_identity("tts", 8105) == "process:new-boot:123:789"


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-instance validation")
def test_switch_with_real_http_servers_keeps_pid_and_stops_unused_models(tmp_path, monkeypatch):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    path = directory / "models.json"
    processes = {}
    monkeypatch.setattr(_reconcile.subprocess, "check_output", lambda *_a, **_kw: "test-revision")
    monkeypatch.setattr(_identity._docker, "container_on_port_checked", lambda *_a: (None, True))
    monkeypatch.setattr(
        _identity._docker,
        "pid_on_port_checked",
        lambda port: (
            (processes[port].pid, True, True)
            if port in processes and processes[port].poll() is None
            else (None, True, False)
        ),
    )

    def stop(targets):
        for _, port in targets:
            if port in processes:
                processes[port].terminate()
                processes[port].wait(timeout=5)
        return True

    monkeypatch.setattr(_reconcile, "stop_persistent_servers", stop)

    def free_port():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return listener.getsockname()[1]

    ports = dict.fromkeys(("tts", "vlm", "omni"))
    for name in ports:
        ports[name] = free_port()
    assert len(set(ports.values())) == 3
    script = """
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
    def log_message(self, *args): pass
server = HTTPServer(('127.0.0.1', int(os.environ['XR_AI_VLLM_PORT'])), Handler)
print('ready', flush=True)
server.serve_forever()
"""

    def start(*names):
        with _reconcile.model_state(path) as state:
            state.prepare({name: {"port": ports[name]} for name in names}, Path(__file__).resolve().parents[1])
            for name in names:
                port = ports[name]
                if port in processes and processes[port].poll() is None:
                    continue
                proc = subprocess.Popen(
                    [sys.executable, "-c", script],
                    stdout=subprocess.PIPE,
                    env=os.environ | {"XR_AI_VLLM_MANAGED": "1", "XR_AI_VLLM_PORT": str(port)},
                )
                processes[port] = proc
                assert select.select([proc.stdout], [], [], 5)[0]
                assert proc.stdout.readline() == b"ready\n"
            state.ready()

    try:
        start("tts", "vlm")
        tts, vlm = processes[ports["tts"]], processes[ports["vlm"]]
        start("tts", "omni")
        assert processes[ports["tts"]] is tts and tts.poll() is None
        assert vlm.poll() is not None
        assert processes[ports["omni"]].poll() is None
        with _reconcile.model_state(path) as state:
            state.stop()
        assert all(proc.poll() is not None for proc in processes.values())
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            proc.stdout.close()
