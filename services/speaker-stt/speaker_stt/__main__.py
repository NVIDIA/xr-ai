# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the private local speaker-ASR IPC process."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import msgpack
import yaml
import zmq
import zmq.asyncio
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_voicegate._speaker import _SpeakerConfig

_SERVICE = "xr-ai-speaker-stt"
_PROTOCOL = 1
_READY_PROCESS_MAY_EXIT_ENV = "_XR_AI_LAUNCHER_READY_PROCESS_MAY_EXIT"


def _validate_capacity(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_sessions must be a positive integer")
    return value


def _fingerprint(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


async def _status(endpoint: str) -> dict | None:
    socket = zmq.asyncio.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.IMMEDIATE, 1)
    socket.connect(endpoint)
    try:
        async with asyncio.timeout(10):
            await socket.send(msgpack.packb({"op": "status"}, use_bin_type=True))
            result = msgpack.unpackb(await socket.recv(), raw=False)
        if not isinstance(result, dict) or result.get("service") != _SERVICE or result.get("protocol") != _PROTOCOL:
            raise RuntimeError("speaker IPC endpoint belongs to an incompatible service")
        return result
    except TimeoutError:
        return None
    finally:
        socket.close()


async def _reuse(cfg: dict, ready_file: Path | None) -> bool:
    endpoint = cfg.get("endpoint", _SpeakerConfig().endpoint)
    path = Path(endpoint.removeprefix("ipc://"))
    if not path.exists():
        return False
    status = await _status(endpoint)
    if status is None:
        return False
    if status["fingerprint"] != _fingerprint(cfg):
        raise RuntimeError("speaker STT configuration changed; stop the existing speaker service before restarting")
    logger.info("speaker ASR already running at {}; reusing", endpoint)
    if ready_file is not None:
        ready_file.touch()
    # Persistent-only stacks permit this readiness proxy to exit. Monitored
    # stacks need it to remain alive until the reused service is lost.
    if os.environ.get(_READY_PROCESS_MAY_EXIT_ENV) != "1":
        while True:
            await asyncio.sleep(5)
            current = await _status(endpoint)
            if current is None or current["fingerprint"] != status["fingerprint"]:
                raise RuntimeError("reused speaker STT service is no longer available")
    return True


class _Server:
    def __init__(self, models, *, max_sessions: int = 16, idle_timeout_s: float = 60.0) -> None:
        self.models = models
        self.max_sessions = _validate_capacity(max_sessions)
        self.idle_timeout_s = idle_timeout_s
        self.sessions: dict[str, tuple[object, float]] = {}

    def _request(self, body: dict) -> dict:
        now = time.monotonic()
        for key, (_, touched) in list(self.sessions.items()):
            if now - touched > self.idle_timeout_s:
                del self.sessions[key]
        session_id = body.get("session")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("invalid session identity")
        op = body.get("op")
        if op == "close":
            self.sessions.pop(session_id, None)
            return {}
        if op == "open":
            if session_id in self.sessions:
                raise ValueError("session already exists")
            if len(self.sessions) >= self.max_sessions:
                raise ValueError("speaker session capacity reached")
            raw = body["config"]
            cfg = _SpeakerConfig._from_yaml({**raw, "enabled": True})
            self.sessions[session_id] = (self.models._session(cfg), now)
            return {}
        if op != "audio" or session_id not in self.sessions:
            raise ValueError("unknown speaker session; enrollment required")
        audio = body.get("audio")
        pts_us = body.get("pts_us")
        if not isinstance(audio, bytes) or len(audio) > 32000:
            raise ValueError("audio must be at most one second of PCM")
        if not isinstance(pts_us, int) or isinstance(pts_us, bool) or pts_us < 0:
            raise ValueError("invalid audio timestamp")
        session, _ = self.sessions[session_id]
        try:
            events = session._feed(audio, pts_us)
        except Exception:
            self.sessions.pop(session_id, None)
            raise
        self.sessions[session_id] = (session, now)
        return {"events": events}


async def _serve(cfg: dict, ready_file: Path | None) -> None:
    endpoint = cfg.get("endpoint", _SpeakerConfig().endpoint)
    _SpeakerConfig._from_yaml({"enabled": True, "endpoint": endpoint})
    max_sessions = _validate_capacity(cfg.get("max_sessions", 16))
    if await _reuse(cfg, ready_file):
        return
    from ._inference import _Models

    socket_path = Path(endpoint.removeprefix("ipc://"))
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    # ZMQ allows a second binder to replace an IPC socket. Keep the lock for
    # the lifetime of the service so an existing server cannot be displaced.
    with socket_path.with_suffix(socket_path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = Path(cfg.get("model_cache", "../../models"))
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache.resolve())
        os.environ.setdefault("NEMO_LOGGING_LEVEL", "ERROR")
        models = await asyncio.to_thread(_Models, cfg)
        server = _Server(models, max_sessions=max_sessions)
        socket = zmq.asyncio.Context.instance().socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.MAXMSGSIZE, 65536)
        socket.bind(endpoint)
        socket_path.chmod(0o600)
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, task.cancel)
        if ready_file is not None:
            ready_file.touch()
        logger.info("speaker ASR ready at {}", endpoint)
        identity = {"service": _SERVICE, "protocol": _PROTOCOL, "fingerprint": _fingerprint(cfg)}
        try:
            while True:
                message = await socket.recv()
                try:
                    body = msgpack.unpackb(message, raw=False)
                    if not isinstance(body, dict):
                        raise ValueError("request must be a mapping")
                    op = body.get("op")
                    if op == "status":
                        response = identity
                    elif op == "shutdown":
                        if body.get("fingerprint") != identity["fingerprint"]:
                            raise ValueError("speaker service identity changed; refusing shutdown")
                        await socket.send(msgpack.packb(identity, use_bin_type=True))
                        return
                    else:
                        # Both models have mutable inference state. Serializing
                        # requests protects conditioning and buffers across clients.
                        response = await asyncio.to_thread(server._request, body)
                except Exception as exc:
                    logger.exception("speaker ASR request failed")
                    response = {"error": str(exc)}
                await socket.send(msgpack.packb(response, use_bin_type=True))
        finally:
            socket.close()
            socket_path.unlink(missing_ok=True)
            if ready_file is not None:
                ready_file.unlink(missing_ok=True)


def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "speaker_stt.yaml")
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("speaker STT config must be a mapping")
    cache = Path(cfg.get("model_cache", "../../models"))
    cfg["model_cache"] = str((args.config.parent / cache).resolve())
    if "cuda_visible_devices" in cfg:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])
    setup_logging("speaker-stt")
    try:
        asyncio.run(_serve(cfg, args.ready_file))
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    _run()
