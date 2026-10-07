# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private speaker-service protocol and lifecycle with stub inference."""

from __future__ import annotations

import asyncio
import os
import stat
from dataclasses import asdict
from types import SimpleNamespace

import msgpack
import pytest
import zmq
import zmq.asyncio
from speaker_stt import __main__ as service
from speaker_stt import _inference
from xr_ai_voicegate._speaker import _SpeakerConfig


class _Session:
    def __init__(self):
        self.count = 0

    def _feed(self, audio):
        self.count += 1
        return [{"kind": "transcript", "text": str(self.count)}]


def _open(server, identity, *, audio_origin_us=1):
    return server._request(
        {
            "op": "open",
            "session": identity,
            "audio_origin_us": audio_origin_us,
            "config": asdict(_SpeakerConfig()),
        }
    )


def _audio(server, identity, **overrides):
    return server._request({"op": "audio", "session": identity, "audio": bytes(640), **overrides})


def test_default_endpoint_uses_an_owner_only_runtime_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    endpoint = _SpeakerConfig().endpoint
    assert endpoint == f"ipc://{tmp_path}/xr-ai/speaker-stt.sock"
    path = service._socket_path(endpoint, create_parent=True)
    assert path.parent.stat().st_uid == os.getuid()
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_ipc_endpoint_rejects_an_untrusted_parent_or_file(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    with pytest.raises(RuntimeError, match="parent directory"):
        service._socket_path(f"ipc://{public}/speaker.sock")

    endpoint = f"ipc://{tmp_path}/speaker.sock"
    path = service._socket_path(endpoint)
    path.write_bytes(b"not a socket")
    with pytest.raises(RuntimeError, match="owner-only socket"):
        service._validate_socket(path)
    path.unlink()

    uid = os.getuid()
    monkeypatch.setattr(service.os, "getuid", lambda: uid + 1)
    with pytest.raises(RuntimeError, match="parent directory"):
        service._socket_path(endpoint)


def test_sessions_isolate_decoder_state_and_close_idempotently():
    server = service._Server(SimpleNamespace(_session=lambda cfg, *, audio_origin_us: _Session()))
    _open(server, "a")
    _open(server, "b")
    assert _audio(server, "a")["events"][0]["text"] == "1"
    assert _audio(server, "b")["events"][0]["text"] == "1"
    assert _audio(server, "a")["events"][0]["text"] == "2"
    for _ in range(2):
        assert server._request({"op": "close", "session": "a"}) == {}
    assert list(server.sessions) == ["b"]


def test_session_expiry_and_capacity_are_bounded(monkeypatch):
    server = service._Server(
        SimpleNamespace(_session=lambda cfg, *, audio_origin_us: _Session()),
        max_sessions=1,
        idle_timeout_s=1,
    )
    monkeypatch.setattr(service.time, "monotonic", lambda: 0)
    _open(server, "a")
    with pytest.raises(ValueError, match="already exists"):
        _open(server, "a")
    with pytest.raises(ValueError, match="capacity"):
        _open(server, "b")
    monkeypatch.setattr(service.time, "monotonic", lambda: 2)
    with pytest.raises(ValueError, match="unknown"):
        _audio(server, "a")
    assert not server.sessions
    _open(server, "b")


@pytest.mark.parametrize("overrides", [{"audio": "pcm"}, {"audio": bytes(32002)}])
def test_invalid_audio_requests_fail_before_inference(overrides):
    server = service._Server(SimpleNamespace(_session=lambda cfg, *, audio_origin_us: _Session()))
    _open(server, "a")
    with pytest.raises(ValueError):
        _audio(server, "a", **overrides)
    assert server.sessions["a"][0].count == 0


@pytest.mark.parametrize("audio_origin_us", [None, -1, True, 1.5, "1"])
def test_open_requires_nonnegative_integer_audio_origin(audio_origin_us):
    server = service._Server(SimpleNamespace(_session=lambda cfg, *, audio_origin_us: _Session()))
    with pytest.raises(ValueError, match="origin timestamp"):
        _open(server, "a", audio_origin_us=audio_origin_us)
    assert not server.sessions


def test_inference_failure_revokes_session():
    def fail(*_args):
        raise ValueError("model failure")

    server = service._Server(
        SimpleNamespace(
            _session=lambda cfg, *, audio_origin_us: SimpleNamespace(_feed=fail)
        )
    )
    _open(server, "a")
    with pytest.raises(ValueError, match="model failure"):
        _audio(server, "a")
    assert not server.sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "16"])
async def test_invalid_capacity_fails_before_model_loading_or_readiness(tmp_path, monkeypatch, capacity):
    def fail(_cfg):
        pytest.fail("invalid capacity reached model loading")

    monkeypatch.setattr(_inference, "_Models", fail)
    ready = tmp_path / "ready"
    cfg = {"endpoint": f"ipc://{tmp_path}/speaker.sock", "max_sessions": capacity}
    with pytest.raises(ValueError, match="positive integer"):
        await service._serve(cfg, ready)
    assert not ready.exists()
    with pytest.raises(ValueError, match="positive integer"):
        service._Server(SimpleNamespace(), max_sessions=capacity)


async def _exchange(endpoint, body):
    socket = zmq.asyncio.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    try:
        async with asyncio.timeout(2):
            await socket.send(msgpack.packb(body, use_bin_type=True))
            return msgpack.unpackb(await socket.recv(), raw=False)
    finally:
        socket.close()


@pytest.mark.asyncio
async def test_private_ipc_readiness_reuse_and_identity_checked_shutdown(tmp_path, monkeypatch):
    loads = []
    monkeypatch.setenv("HF_HOME", "operator-hf")
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")
    monkeypatch.delenv("NEMO_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        _inference,
        "_Models",
        lambda cfg: loads.append(cfg)
        or SimpleNamespace(_session=lambda cfg, *, audio_origin_us: _Session()),
    )
    monkeypatch.setenv("_XR_AI_LAUNCHER_READY_PROCESS_MAY_EXIT", "1")
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *_args: None)
    endpoint = f"ipc://{tmp_path}/speaker.sock"
    cfg = {"endpoint": endpoint, "model_cache": str(tmp_path / "models")}
    ready = tmp_path / "first.ready"
    task = asyncio.create_task(service._serve(cfg, ready))
    try:
        async with asyncio.timeout(2):
            while not ready.exists():
                await asyncio.sleep(0.01)
        assert (tmp_path / "speaker.sock").stat().st_mode & 0o777 == 0o600
        assert os.environ["HF_HOME"] == "operator-hf"
        assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"
        assert os.environ["NEMO_CACHE_DIR"] == str((tmp_path / "models" / "nemo").resolve())
        status = await service._status(endpoint)
        assert status["fingerprint"] == service._fingerprint(cfg)
        reused_ready = tmp_path / "reused.ready"
        await service._serve(cfg, reused_ready)
        assert reused_ready.exists() and len(loads) == 1
        for identity in ("a", "b"):
            assert (
                await _exchange(
                    endpoint,
                    {
                        "op": "open",
                        "session": identity,
                        "audio_origin_us": 1,
                        "config": asdict(_SpeakerConfig()),
                    },
                )
                == {}
            )
        assert (await _exchange(endpoint, {"op": "audio", "session": "a", "audio": bytes(640)}))["events"][
            0
        ]["text"] == "1"
        assert (await _exchange(endpoint, {"op": "audio", "session": "b", "audio": bytes(640)}))["events"][
            0
        ]["text"] == "1"
        assert await _exchange(endpoint, {"op": "close", "session": "a"}) == {}
        changed_ready = tmp_path / "changed.ready"
        with pytest.raises(RuntimeError, match="configuration changed"):
            await service._serve({**cfg, "precision": "float32"}, changed_ready)
        assert not changed_ready.exists() and len(loads) == 1
        result = await _exchange(endpoint, {"op": "shutdown", "fingerprint": "stale"})
        assert "identity changed" in result["error"] and not task.done()
        assert await _exchange(endpoint, {"op": "shutdown", "fingerprint": status["fingerprint"]}) == status
        await asyncio.wait_for(task, timeout=2)
        assert not ready.exists() and not (tmp_path / "speaker.sock").exists()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity",
    [
        {"service": "foreign", "protocol": service._PROTOCOL},
        {"service": service._SERVICE, "protocol": service._PROTOCOL - 1},
    ],
)
async def test_status_refuses_foreign_or_old_protocol_endpoint(tmp_path, identity):
    endpoint = f"ipc://{tmp_path}/foreign.sock"
    socket = zmq.asyncio.Context.instance().socket(zmq.REP)
    socket.bind(endpoint)
    (tmp_path / "foreign.sock").chmod(0o600)

    async def reply():
        await socket.recv()
        await socket.send(msgpack.packb(identity))

    task = asyncio.create_task(reply())
    try:
        with pytest.raises(RuntimeError, match="incompatible"):
            await service._status(endpoint)
        await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        socket.close(linger=0)


@pytest.mark.asyncio
async def test_reuse_watcher_requires_three_consecutive_status_failures(monkeypatch):
    expected = {"service": service._SERVICE, "protocol": service._PROTOCOL, "fingerprint": "same"}
    responses = iter([None, None, expected, None, None, None])
    calls = []

    async def status(endpoint):
        calls.append(endpoint)
        return next(responses)

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(service, "_status", status)
    monkeypatch.setattr(service.asyncio, "sleep", no_sleep)
    with pytest.raises(RuntimeError, match="no longer available"):
        await service._watch_reused("ipc:///unused", "same")
    assert len(calls) == 6
