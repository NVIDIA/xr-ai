# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker-STT stack selection and lifecycle; no models or recorded audio."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import msgpack
import pytest
import yaml
import zmq
import zmq.asyncio
from speaker_stt import __main__ as service
from speaker_stt import _inference
from xr_ai_voice._speaker_client import _SpeakerClient
from xr_ai_voicegate._speaker import _SpeakerConfig

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "speaker_stack_main", _ROOT / "model-server-samples/model-servers/main.py",
)
assert _SPEC and _SPEC.loader
stack = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stack)


@pytest.mark.parametrize("key", ["port", "http_port"])
def test_speaker_ipc_config_cannot_add_http_cleanup_targets(tmp_path, monkeypatch, key):
    config = tmp_path / "yaml" / "custom" / "speaker_stt.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(f"endpoint: ipc:///tmp/test-speaker.sock\n{key}: 8110\n")
    profile = tmp_path / "models.speaker.json"
    profile.write_text(json.dumps({
        "models": {}, "services": {"speaker-stt": {"ownership": "managed"}},
    }))
    monkeypatch.setattr(stack, "_BASE", tmp_path)
    with pytest.raises(ValueError, match="must not declare an HTTP port"):
        stack._build_processes(str(profile), "custom")
    assert not stack._known_service_ports()




@pytest.mark.parametrize("hardware", ["dual_48G_ada", "96G_blackwell", "spark"])
def test_normal_profile_selects_speaker_asr_and_uses_hardware_profile(hardware):
    processes, credentials = stack._build_processes("default", hardware)
    names = [p.name for p in processes]
    assert "stt" not in names and "stt-nim" not in names
    assert names.count("speaker-stt") == 1
    speaker = next(p for p in processes if p.name == "speaker-stt")
    assert speaker.port is None and speaker.launch_mode == "persist"
    assert speaker.command == "speaker_stt"
    cfg = yaml.safe_load(Path(speaker.config).read_text())
    assert cfg["cuda_visible_devices"] == ("1" if hardware == "dual_48G_ada" else "0")
    assert (Path(speaker.config).parent / cfg["model_cache"]).resolve() == _ROOT / "models"
    assert credentials == ()


def test_custom_batch_profile_remains_supported(tmp_path):
    profile = tmp_path / "models.speech.json"
    profile.write_text(json.dumps({"models": {"stt": {
        "adapter": {"kind": "riva_grpc"}, "endpoint": {"base_url": "localhost:50051"},
        "deployment": {"ownership": "managed", "service": "stt-nim", "credentials": ["NGC_API_KEY"]},
    }}}))
    processes, credentials = stack._build_processes(str(profile), "dual_48G_ada")
    assert [p.name for p in processes] == ["stt-nim"]
    assert credentials == ("NGC_API_KEY",)


@pytest.mark.parametrize("declaration", [
    [], {"unknown": {"ownership": "managed"}}, {"speaker-stt": {"ownership": "reused"}},
    {"speaker-stt": {"ownership": "managed", "endpoint": "tcp://*:8103"}},
])
def test_invalid_private_service_declaration_fails_before_launch(tmp_path, declaration):
    profile = tmp_path / "models.invalid.json"
    profile.write_text(json.dumps({"models": {}, "services": declaration}))
    with pytest.raises(ValueError, match="services"):
        stack._build_processes(str(profile), "dual_48G_ada")


def test_normal_cli_passes_only_profile_selection(monkeypatch):
    selected = []
    monkeypatch.setattr(stack, "setup_logging", lambda *_a, **_kw: None)
    monkeypatch.setattr(stack, "require_credentials", lambda *_a, **_kw: None)
    monkeypatch.setattr(stack, "_stop_unselected_services", lambda _p: None)
    monkeypatch.setattr(stack, "run_stack", lambda *_a, **_kw: None)
    monkeypatch.setattr(stack, "_build_processes", lambda *a, **kw: (selected.append((a, kw)) or [], ()))
    monkeypatch.setattr(sys, "argv", ["model_servers", "--models", "vlm_llm_nim"])
    stack.run()
    assert selected == [(("vlm_llm_nim", None), {})]


def test_profile_switches_stop_only_unselected_services(tmp_path, monkeypatch):
    endpoints = []
    ports = []
    monkeypatch.setattr(stack, "_stop_speaker_endpoint", endpoints.append)
    monkeypatch.setattr(stack, "stop_persistent_servers", lambda services: ports.extend(services) or True)
    processes, _ = stack._build_processes("default", "dual_48G_ada")
    stack._stop_unselected_services(processes)
    assert not endpoints
    assert {8103, 9010} <= {port for _, port in ports}
    profile = tmp_path / "models.batch.json"
    body = json.loads(stack._profile_path("default").read_text())
    body.pop("services")
    body["models"]["stt"]["deployment"]["ownership"] = "managed"
    profile.write_text(json.dumps(body))
    processes, _ = stack._build_processes(str(profile), "dual_48G_ada")
    stack._stop_unselected_services(processes)
    assert endpoints == ["ipc:///tmp/xr-ai-speaker-stt.sock"]
    endpoints.clear()
    stack._stop_models()
    assert endpoints == ["ipc:///tmp/xr-ai-speaker-stt.sock"]


async def _wait_ready(path):
    async with asyncio.timeout(2):
        while not path.exists():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_ipc_readiness_warm_reuse_changed_config_and_stop(tmp_path, monkeypatch):
    loads = []
    monkeypatch.setattr(
        _inference, "_Models",
        lambda cfg: loads.append(cfg) or SimpleNamespace(
            _session=lambda cfg: SimpleNamespace(_feed=lambda audio, pts_us: []),
        ),
    )
    monkeypatch.setenv("_XR_AI_LAUNCHER_READY_PROCESS_MAY_EXIT", "1")
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *_a: None)
    cfg = {"endpoint": f"ipc://{tmp_path}/speaker.sock", "model_cache": str(tmp_path / "models")}
    ready = tmp_path / "first.ready"
    task = asyncio.create_task(service._serve(cfg, ready))
    try:
        await _wait_ready(ready)
        assert stack._speaker_lock_held(tmp_path / "speaker.sock")
        status = await service._status(cfg["endpoint"])
        assert status["fingerprint"] == service._fingerprint(cfg)
        reused_ready = tmp_path / "second.ready"
        await service._serve(cfg, reused_ready)
        assert reused_ready.exists() and len(loads) == 1
        client = _SpeakerClient(_SpeakerConfig(endpoint=cfg["endpoint"]))
        assert await client._available()
        assert await client._feed("wearer", bytes(640), 1_000_000) == []
        await client._close()
        changed_ready = tmp_path / "changed.ready"
        with pytest.raises(RuntimeError, match="configuration changed"):
            await service._serve({**cfg, "precision": "float32"}, changed_ready)
        assert not changed_ready.exists() and len(loads) == 1
        # A stale control request must not stop a replacement configuration.
        client = zmq.asyncio.Context.instance().socket(zmq.REQ)
        client.connect(cfg["endpoint"])
        try:
            await client.send(msgpack.packb({"op": "shutdown", "fingerprint": "stale"}))
            result = msgpack.unpackb(await client.recv(), raw=False)
            assert "identity changed" in result["error"]
        finally:
            client.close(linger=0)
        assert not task.done()
        await asyncio.to_thread(stack._stop_speaker_endpoint, cfg["endpoint"])
        await asyncio.wait_for(task, timeout=2)
        assert not ready.exists() and not (tmp_path / "speaker.sock").exists()
        assert not stack._speaker_lock_held(tmp_path / "speaker.sock")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cleanup_refuses_unrelated_ipc_service(tmp_path):
    endpoint = f"ipc://{tmp_path}/unrelated.sock"
    socket = zmq.asyncio.Context.instance().socket(zmq.REP)
    socket.bind(endpoint)
    requests = []
    async def reply():
        request = msgpack.unpackb(await socket.recv(), raw=False)
        requests.append(request)
        await socket.send(msgpack.packb({"service": "unrelated", "protocol": 1}))
    task = asyncio.create_task(reply())
    try:
        with pytest.raises(RuntimeError, match="incompatible"):
            await asyncio.to_thread(stack._stop_speaker_endpoint, endpoint)
        await task
        assert requests == [{"op": "status"}]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        socket.close(linger=0)


@pytest.mark.parametrize("owned", [False, True])
def test_unresponsive_socket_is_only_ignored_without_live_owner(tmp_path, monkeypatch, owned):
    path = tmp_path / "speaker.sock"
    path.touch()
    monkeypatch.setattr(stack, "_speaker_lock_held", lambda _p: owned)
    def timeout(*_a):
        raise TimeoutError("unresponsive")
    monkeypatch.setattr(stack, "_speaker_exchange", timeout)
    if owned:
        with pytest.raises(TimeoutError, match="unresponsive"):
            stack._stop_speaker_endpoint(f"ipc://{path}")
    else:
        stack._stop_speaker_endpoint(f"ipc://{path}")
