# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
pocket_tts_server — Pocket TTS HTTP server.

Pocket TTS is a compact, streaming-capable model. This wrapper exposes the
repository's existing OpenAI-compatible API:

    POST /v1/audio/speech
    GET  /v1/models
    GET  /health

Model weights and the precomputed voice embedding come from Kyutai model
repositories on Hugging Face. The selected model and voice are loaded before
the service reports ready.

Accepts --config <path>.yaml (auto-passed by xr-ai-launcher).

Config keys
-----------
    voice:        str   Must be "bill_boerst" (required)
    language:     str   Pocket TTS language/model variant (default: "english")
    device:       str   "cpu", "cuda", or "cuda:N" (default: "cpu")
    port:         int   HTTP port (default: 8105)
    host:         str   Bind address (default: "0.0.0.0")
    startup_timeout_s: float  Seconds allowed for a cold start (default: 600)
    model_cache:  str   Hugging Face cache path, resolved relative to this YAML.
                        Default: ../../models
"""
import argparse
import asyncio
import io
import json
import math
import os
import re
import socket
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import yaml
from loguru import logger
from xr_ai_logging import setup_logging

_DEFAULT_PORT = 8105
_DEFAULT_STARTUP_TIMEOUT_S = 600.0
_PROCESS_GROUP_ENV = "_XR_AI_POCKET_PROCESS_GROUP"
_LAUNCHER_GROUP_OWNER_ENV = "_XR_AI_LAUNCHER_PROCESS_GROUP_OWNER"
_LAUNCHER_GROUP_OWNER = "pocket_tts_server"
_READY_PROCESS_MAY_EXIT_ENV = "_XR_AI_LAUNCHER_READY_PROCESS_MAY_EXIT"
_REUSE_HEALTH_FAILURE_LIMIT = 3
_SUPPORTED_VOICES = frozenset({"bill_boerst"})


class _SynthesisCancelled(RuntimeError):
    """A queued request was abandoned before generation started."""


def _parse_device(value: object) -> str:
    """Validate the supported device choices without importing Torch at bootstrap."""
    if value in ("cpu", "cuda"):
        return value
    if isinstance(value, str) and re.fullmatch(r"cuda:[0-9]+", value):
        return f"cuda:{int(value.split(':')[1])}"
    raise ValueError("'device' must be cpu, cuda, or cuda:N with a non-negative index")


def _parse_startup_timeout(value: object) -> float:
    """Return a positive finite startup timeout from user configuration."""
    try:
        timeout_s = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "'startup_timeout_s' must be a finite number greater than zero"
        ) from exc
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(
            "'startup_timeout_s' must be a finite number greater than zero"
        )
    return timeout_s


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../../models")
    p = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


class _PocketTTSBackend:
    """Thread-safe Pocket TTS voice loader and synthesizer."""

    def __init__(self, voice: str, language: str, device: str = "cpu") -> None:
        self._voice_name = voice
        self._language = language
        self._requested_device = _parse_device(device)
        self.device: str | None = None
        self._model = None
        self._voice_state = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if self._voice_name not in _SUPPORTED_VOICES:
                choices = ", ".join(sorted(_SUPPORTED_VOICES))
                raise ValueError(
                    f"unsupported Pocket TTS voice {self._voice_name!r}; "
                    f"this release supports: {choices}"
                )

            from pocket_tts import TTSModel

            device = self._requested_device
            if device.startswith("cuda"):
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "Pocket TTS requested CUDA, but CUDA is unavailable. Install a CUDA-enabled "
                        "Torch build and check the NVIDIA driver, or select device: cpu."
                    )
                index = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
                if index >= torch.cuda.device_count():
                    raise ValueError(f"Pocket TTS CUDA device index {index} is not visible")
                device = f"cuda:{index}"

            logger.info("Loading Pocket TTS language {!r} on {}…", self._language, device)
            model = TTSModel.load_model(language=self._language).to(device)
            logger.info("Loading voice {!r}…", self._voice_name)
            # Precomputed voice tensors use model.device; load them after moving the model.
            voice_state = model.get_state_for_audio_prompt(self._voice_name)
            self._model = model
            self._voice_state = voice_state
            self.device = device
            weights = (
                "gated voice-cloning"
                if model.has_voice_cloning
                else "ungated no-voice-cloning fallback"
            )
            logger.info(
                "Pocket TTS ready  device={} sample_rate={} weights={}",
                device,
                model.sample_rate,
                weights,
            )

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        self._ensure_loaded()
        return self._model.sample_rate

    def synthesize(
        self,
        text: str,
        response_format: str = "wav",
        cancelled: threading.Event | None = None,
    ) -> bytes:
        """Synthesize text to WAV or signed 16-bit mono PCM bytes."""
        if response_format not in {"wav", "pcm"}:
            raise ValueError(
                f"Pocket TTS supports response_format 'wav' or 'pcm', "
                f"got {response_format!r}"
            )
        self._ensure_loaded()

        pcm = b""
        if text.strip():
            with self._generation_lock:
                if cancelled is not None and cancelled.is_set():
                    raise _SynthesisCancelled
                audio = self._model.generate_audio(self._voice_state, text)
            pcm = (
                (audio.reshape(-1).clamp(-1, 1) * 32767)
                .short()
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            )
        if response_format == "pcm":
            return pcm

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()


def _build_app(cfg: dict, _model_cache: Path):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response
    from pydantic import BaseModel

    voice_name = cfg["voice"]
    language = str(cfg.get("language", "english"))
    backend = _PocketTTSBackend(voice_name, language, cfg.get("device", "cpu"))
    generation_lock = asyncio.Lock()

    app = FastAPI(title="Pocket TTS Server", version="0.1.0")

    class SpeechRequest(BaseModel):
        model:           str   = voice_name
        input:           str
        voice:           str   = "default"
        speed:           float = 1.0
        response_format: str   = "wav"

    @app.get("/health")
    def health():
        if not backend.ready:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ok", "device": backend.device}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": voice_name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/audio/speech")
    async def synthesize(req: SpeechRequest, request: Request):
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()
        try:
            # Waiting for this lock is cancellable, unlike a worker blocked on
            # the model's thread lock. Re-check the connection after acquiring
            # it so abandoned queued sentences never enter Pocket generation.
            async with generation_lock:
                if await request.is_disconnected():
                    raise HTTPException(status_code=499, detail="request disconnected")
                audio_bytes = await loop.run_in_executor(
                    None,
                    backend.synthesize,
                    req.input,
                    req.response_format,
                    cancelled,
                )
        except asyncio.CancelledError:
            cancelled.set()
            raise
        except _SynthesisCancelled as exc:
            raise HTTPException(status_code=499, detail="request cancelled") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        media_type = "audio/pcm" if req.response_format == "pcm" else "audio/wav"
        return Response(content=audio_bytes, media_type=media_type)

    return app, backend


def _health_url_ok(health_url: str) -> bool:
    """Return True if *health_url* answers successfully."""
    try:
        with urllib.request.urlopen(health_url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _check_reused_device(health_url: str, requested: str) -> None:
    """Reject a mismatched or legacy listener without stopping an existing service."""
    try:
        with urllib.request.urlopen(health_url, timeout=2) as response:
            info = json.load(response)
        actual = info.get("device") if isinstance(info, dict) else None
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot verify the existing Pocket TTS device: {exc}") from exc
    matches = actual == requested or (
        requested == "cuda" and isinstance(actual, str) and re.fullmatch(r"cuda:[0-9]+", actual)
    )
    if not matches:
        raise RuntimeError(
            f"existing TTS service reports device={actual!r}, requested {requested!r}. "
            "Stop the owned TTS service before changing device or upgrading a legacy listener; "
            "it has not been stopped automatically."
        )


def _probe_host(bind_host: str) -> str:
    """Return a reachable local address for a configured bind host."""
    return {
        "": "127.0.0.1",
        "0.0.0.0": "127.0.0.1",
        "::": "::1",
    }.get(bind_host, bind_host)


def _health_url(host: str, port: int) -> str:
    """Return the health URL for a concrete probe host and port."""
    url_host = f"[{host}]" if ":" in host else host
    return f"http://{url_host}:{port}/health"


def _port_open(host: str, port: int) -> bool:
    """Return True if a process is already listening on the probe address."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _ensure_owned_process_group() -> int | None:
    """Return the current safely signalable process group, if verified.

    A launcher child remains in the dedicated session created around ``uv`` so
    abort-time SIGKILL can still reach it. A directly started Pocket TTS
    process is accepted only when it already leads its own session; otherwise
    cleanup falls back to the listener PID.
    """
    pid = os.getpid()
    try:
        process_group = os.getpgrp()
        session_id = os.getsid(0)
    except OSError:
        return None

    if process_group != session_id:
        return None
    if process_group == pid:
        return process_group
    if os.environ.get(_LAUNCHER_GROUP_OWNER_ENV) == _LAUNCHER_GROUP_OWNER:
        return process_group
    return None


def _monitor_reused_server(health_url: str, poll_s: float = 5.0) -> None:
    """Remain monitorable until a reused server repeatedly fails health checks."""
    failures = 0
    while failures < _REUSE_HEALTH_FAILURE_LIMIT:
        time.sleep(poll_s)
        if _health_url_ok(health_url):
            failures = 0
            continue
        failures += 1
        if failures < _REUSE_HEALTH_FAILURE_LIMIT:
            print(
                f"[pocket_tts_server] reused server health endpoint unreachable "
                f"({failures}/{_REUSE_HEALTH_FAILURE_LIMIT}); retrying",
                flush=True,
            )
    raise SystemExit(
        f"[pocket_tts_server] reused server failed "
        f"{_REUSE_HEALTH_FAILURE_LIMIT} consecutive health checks"
    )


async def _load_backend(
    backend: _PocketTTSBackend,
) -> None:
    """Load Pocket TTS without letting a stuck native initializer block shutdown."""
    loop = asyncio.get_running_loop()
    loaded = loop.create_future()

    def _publish(error: BaseException | None) -> None:
        if loaded.done():
            return
        if error is None:
            loaded.set_result(None)
        else:
            loaded.set_exception(error)

    def _load() -> None:
        error: BaseException | None = None
        try:
            backend._ensure_loaded()
        except BaseException as exc:
            error = exc
        try:
            loop.call_soon_threadsafe(_publish, error)
        except RuntimeError:
            # A startup timeout closes the loop while the daemon loader is
            # still unwinding. The process is already exiting in that case.
            pass

    threading.Thread(target=_load, name="pocket-model-loader", daemon=True).start()
    await loaded


async def _run(
    cfg: dict,
    yaml_dir: Path,
    ready_file: Path | None = None,
) -> None:
    import uvicorn

    if not cfg.get("voice"):
        logger.error("'voice' is required in config")
        sys.exit(1)

    model_cache = _resolve_model_cache(cfg, yaml_dir)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_XET_CACHE", str(model_cache / "pocket" / "xet"))
    os.environ.setdefault("HF_HOME", str(model_cache / "pocket" / "huggingface"))
    startup_timeout_s = _parse_startup_timeout(
        cfg.get("startup_timeout_s", _DEFAULT_STARTUP_TIMEOUT_S)
    )
    port = int(cfg.get("port", _DEFAULT_PORT))
    host = cfg.get("host", "0.0.0.0")

    app, backend = _build_app(cfg, model_cache)

    serve_task: asyncio.Task | None = None
    try:
        async with asyncio.timeout(startup_timeout_s):
            await _load_backend(backend)

            config = uvicorn.Config(app, host=host, port=port, log_level="warning")
            server = uvicorn.Server(config)

            logger.info("Starting HTTP server on http://localhost:{}/v1", port)
            serve_task = asyncio.create_task(server.serve())
            while not server.started:
                if serve_task.done():
                    await serve_task
                    raise RuntimeError("HTTP server exited before becoming ready")
                await asyncio.sleep(0.05)
    except TimeoutError as exc:
        if serve_task is not None:
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass
        raise TimeoutError(
            f"server did not become ready within {startup_timeout_s:g} seconds"
        ) from exc

    assert serve_task is not None

    logger.info("Ready  →  http://localhost:{}/v1", port)
    if ready_file:
        ready_file.touch()

    await serve_task
    logger.info("Stopped.")


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    setup_logging("tts-pocket")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=Path, default=None)
    p.add_argument("--ready-file", type=Path, default=None)
    p.add_argument("--_serve",     action="store_true",
                   help=argparse.SUPPRESS)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    yaml_dir  = Path.cwd()
    if ns.config and ns.config.exists():
        yaml_dir = ns.config.parent.resolve()
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    if ns._serve:
        try:
            asyncio.run(_run(cfg, yaml_dir, ready_file=ns.ready_file))
        except Exception as exc:
            raise SystemExit(f"[pocket_tts_server] {exc}") from None
        return

    port = int(cfg.get("port", _DEFAULT_PORT))
    probe_host = _probe_host(str(cfg.get("host", "0.0.0.0")))
    try:
        startup_timeout_s = _parse_startup_timeout(
            cfg.get("startup_timeout_s", _DEFAULT_STARTUP_TIMEOUT_S)
        )
        device = _parse_device(cfg.get("device", "cpu"))
    except ValueError as exc:
        raise SystemExit(f"[pocket_tts_server] {exc}") from exc
    health_url = _health_url(probe_host, port)

    if _health_url_ok(health_url):
        if "device" in cfg:
            try:
                _check_reused_device(health_url, device)
            except RuntimeError as exc:
                raise SystemExit(f"[pocket_tts_server] {exc}") from exc
        print(
            f"[pocket_tts_server] already running on port {port} — reusing",
            flush=True,
        )
        if ns.ready_file:
            ns.ready_file.touch()
        if os.environ.get(_READY_PROCESS_MAY_EXIT_ENV) != "1":
            _monitor_reused_server(health_url)
        return

    if _port_open(probe_host, port):
        raise SystemExit(
            f"[pocket_tts_server] port {port} is already in use, but its "
            "/health endpoint is not healthy"
        )

    cmd = [sys.executable, "-m", "pocket_tts_server", "--_serve"]
    if ns.config:
        cmd += ["--config", str(ns.config)]
    if ns.ready_file:
        cmd += ["--ready-file", str(ns.ready_file)]
    print(
        f"[pocket_tts_server] starting managed server on port {port} "
        f"(startup timeout: {startup_timeout_s:g}s)…",
        flush=True,
    )
    child_env = os.environ | {
        "XR_AI_VLLM_MANAGED": "1",
        "XR_AI_VLLM_PORT": str(port),
    }
    child_env.pop(_PROCESS_GROUP_ENV, None)
    child_env.pop(_READY_PROCESS_MAY_EXIT_ENV, None)
    if process_group := _ensure_owned_process_group():
        child_env[_PROCESS_GROUP_ENV] = str(process_group)
    os.execvpe(
        sys.executable,
        cmd,
        child_env,
    )


if __name__ == "__main__":
    run()
