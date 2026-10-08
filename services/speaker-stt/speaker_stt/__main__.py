# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the private local speaker-ASR HTTP and WebSocket service."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from xr_ai_logging import setup_logging
from xr_ai_voicegate._speaker import _SpeakerConfig

_SERVICE = "xr-ai-speaker-stt"
_PROTOCOL = 3
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8102
_STREAM_PATH = "/v1/audio/transcriptions/stream"
_REUSE_STATUS_FAILURE_LIMIT = 3
_READY_PROCESS_MAY_EXIT_ENV = "_XR_AI_LAUNCHER_READY_PROCESS_MAY_EXIT"


class _ProtocolError(ValueError):
    pass


def _validate_capacity(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_sessions must be a positive integer")
    return value


def _validate_host(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("host must be a non-empty hostname or address")
    return value


def _validate_port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("port must be an integer from 1 through 65535")
    return value


def _base_url(cfg: dict) -> str:
    host = _validate_host(cfg.get("host", _DEFAULT_HOST))
    port = _validate_port(cfg.get("port", _DEFAULT_PORT))
    # Wildcard addresses are valid listener targets, but clients must probe a
    # concrete local address when deciding whether an owned service is reusable.
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    display_host = f"[{host}]" if ":" in host else host
    return f"http://{display_host}:{port}"


def _fingerprint(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _identity(cfg: dict) -> dict:
    return {
        "status": "ok",
        "service": _SERVICE,
        "protocol": _PROTOCOL,
        "fingerprint": _fingerprint(cfg),
    }


def _blocking_status(base_url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            result = json.load(response)
    except (OSError, TimeoutError, urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(result, dict)
        or result.get("service") != _SERVICE
        or result.get("protocol") != _PROTOCOL
    ):
        raise RuntimeError("speaker HTTP endpoint belongs to an incompatible service")
    return result


async def _status(base_url: str) -> dict | None:
    return await asyncio.to_thread(_blocking_status, base_url)


async def _await_without_abandoning(task: asyncio.Task):
    """Keep serialized model work owned until its thread finishes."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


async def _watch_reused(base_url: str, fingerprint: str) -> None:
    failures = 0
    while True:
        await asyncio.sleep(5)
        current = await _status(base_url)
        if current is None:
            failures += 1
            if failures < _REUSE_STATUS_FAILURE_LIMIT:
                logger.warning(
                    "reused speaker STT status timed out ({}/{}); retrying",
                    failures,
                    _REUSE_STATUS_FAILURE_LIMIT,
                )
                continue
            raise RuntimeError("reused speaker STT service is no longer available")
        failures = 0
        if current["fingerprint"] != fingerprint:
            raise RuntimeError("reused speaker STT service identity changed")


async def _reuse(cfg: dict, ready_file: Path | None) -> bool:
    base_url = _base_url(cfg)
    status = await _status(base_url)
    if status is None:
        return False
    if status["fingerprint"] != _fingerprint(cfg):
        raise RuntimeError("speaker STT configuration changed; stop the existing speaker service before restarting")
    logger.info("speaker ASR already running at {}; reusing", base_url)
    if ready_file is not None:
        ready_file.touch()
    # Persistent-only stacks permit this readiness proxy to exit. Monitored
    # stacks need it to remain alive until the reused service is lost.
    if os.environ.get(_READY_PROCESS_MAY_EXIT_ENV) != "1":
        try:
            await _watch_reused(base_url, status["fingerprint"])
        finally:
            if ready_file is not None:
                ready_file.unlink(missing_ok=True)
    return True


class _Server:
    def __init__(self, models, *, max_sessions: int = 16, idle_timeout_s: float = 60.0) -> None:
        self.models = models
        self.max_sessions = _validate_capacity(max_sessions)
        self.idle_timeout_s = idle_timeout_s
        self.sessions: dict[str, tuple[object, float]] = {}

    def _expire(self, now: float) -> None:
        for session_id, (_, touched) in list(self.sessions.items()):
            if now - touched > self.idle_timeout_s:
                del self.sessions[session_id]

    def _open(self, body: object) -> str:
        now = time.monotonic()
        self._expire(now)
        if not isinstance(body, dict):
            raise _ProtocolError("opening message must be a JSON object")
        raw = body.get("config")
        if not isinstance(raw, dict):
            raise _ProtocolError("speaker config must be a mapping")
        audio_origin_us = body.get("audio_origin_us")
        if (
            not isinstance(audio_origin_us, int)
            or isinstance(audio_origin_us, bool)
            or audio_origin_us < 0
        ):
            raise _ProtocolError("invalid audio origin timestamp")
        if len(self.sessions) >= self.max_sessions:
            raise _ProtocolError("speaker session capacity reached")
        try:
            cfg = _SpeakerConfig._from_yaml({**raw, "enabled": True})
        except ValueError as exc:
            raise _ProtocolError(str(exc)) from exc
        session_id = uuid4().hex
        self.sessions[session_id] = (self.models._session(cfg, audio_origin_us=audio_origin_us), now)
        return session_id

    def _audio(self, session_id: str, audio: object) -> list[dict]:
        now = time.monotonic()
        self._expire(now)
        entry = self.sessions.get(session_id)
        if entry is None:
            raise _ProtocolError("unknown speaker session; open a new stream")
        session, _ = entry
        if not isinstance(audio, bytes) or len(audio) > 32000:
            raise _ProtocolError("audio must be at most one second of PCM")
        try:
            events = session._feed(audio)
        except Exception:
            self.sessions.pop(session_id, None)
            raise
        self.sessions[session_id] = (session, now)
        return events

    def _close(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


def _build_app(server: _Server, cfg: dict, ready_file: Path | None = None):
    identity = _identity(cfg)
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if ready_file is not None:
            ready_file.touch()
        try:
            yield
        finally:
            if ready_file is not None:
                ready_file.unlink(missing_ok=True)

    app = FastAPI(title="Speaker STT", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return identity

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [
                {"id": cfg["asr_model"], "object": "model", "owned_by": "local"},
                {"id": cfg["diar_model"], "object": "model", "owned_by": "local"},
            ],
        }

    @app.websocket(_STREAM_PATH)
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id = None
        try:
            opening = await websocket.receive_json()
            session_id = server._open(opening)
            await websocket.send_json({"service": _SERVICE, "protocol": _PROTOCOL})
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                audio = message.get("bytes")
                if audio is None:
                    raise _ProtocolError("speaker audio frames must be binary PCM")
                async with inference_lock:
                    work = asyncio.create_task(asyncio.to_thread(server._audio, session_id, audio))
                    events = await _await_without_abandoning(work)
                await websocket.send_json({"events": events})
        except WebSocketDisconnect:
            pass
        except _ProtocolError as exc:
            try:
                await websocket.send_json({"error": str(exc)})
                await websocket.close(code=1008)
            except (RuntimeError, WebSocketDisconnect):
                pass
        except Exception:
            logger.exception("speaker ASR stream failed")
            try:
                await websocket.send_json({"error": "speaker inference failed"})
                await websocket.close(code=1011)
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            if session_id is not None:
                server._close(session_id)

    return app


def _listener(cfg: dict) -> socket.socket:
    host = _validate_host(cfg.get("host", _DEFAULT_HOST))
    port = _validate_port(cfg.get("port", _DEFAULT_PORT))
    family, socktype, proto, _, address = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0]
    listener = socket.socket(family, socktype, proto)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address)
        listener.listen(2048)
        listener.setblocking(False)
    except Exception:
        listener.close()
        raise
    return listener


async def _serve(cfg: dict, ready_file: Path | None) -> None:
    import uvicorn

    max_sessions = _validate_capacity(cfg.get("max_sessions", 16))
    _base_url(cfg)
    if await _reuse(cfg, ready_file):
        return

    # Own the port before loading models so concurrent starts cannot allocate a
    # second GPU copy while the first process is still warming.
    listener = _listener(cfg)
    try:
        from ._inference import _Models

        cache = Path(cfg["model_cache"])
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("NEMO_CACHE_DIR", str((cache / "nemo").resolve()))
        os.environ.setdefault("HF_HOME", str((cache / "huggingface").resolve()))
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        os.environ.setdefault("NEMO_LOGGING_LEVEL", "ERROR")
        models = await asyncio.to_thread(_Models, cfg)
        app = _build_app(_Server(models, max_sessions=max_sessions), cfg, ready_file)
        config = uvicorn.Config(app, log_level="warning")
        uvicorn_server = uvicorn.Server(config)
        logger.info("starting speaker ASR at {}", _base_url(cfg))
        await uvicorn_server.serve(sockets=[listener])
    finally:
        listener.close()
        if ready_file is not None:
            ready_file.unlink(missing_ok=True)


def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "speaker_stt.yaml")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--_serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("speaker STT config must be a mapping")
    cache = Path(cfg.get("model_cache", "../../models"))
    cfg["model_cache"] = str((args.config.parent / cache).resolve())
    if "cuda_visible_devices" in cfg:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])
    setup_logging("speaker-stt")
    if not args._serve:
        port = _validate_port(cfg.get("port", _DEFAULT_PORT))
        cmd = [sys.executable, "-m", "speaker_stt", "--_serve", "--config", str(args.config)]
        if args.ready_file is not None:
            cmd += ["--ready-file", str(args.ready_file)]
        child_env = os.environ | {
            "XR_AI_VLLM_MANAGED": "1",
            "XR_AI_VLLM_PORT": str(port),
        }
        child_env.pop(_READY_PROCESS_MAY_EXIT_ENV, None)
        os.execvpe(sys.executable, cmd, child_env)
    try:
        asyncio.run(_serve(cfg, args.ready_file))
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    _run()
