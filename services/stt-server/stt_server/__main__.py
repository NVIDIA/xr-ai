# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
stt_server — OpenAI-compatible Speech-to-Text server.

Loads nvidia/parakeet-tdt-0.6b-v3 (NeMo ASR) in-process and serves an
OpenAI-compatible transcription API:

    POST /v1/audio/transcriptions   (multipart/form-data)
    GET  /v1/models

Accepts --config <path>.yaml (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:             str    NeMo / HuggingFace model name (required)
    device:            str    "cuda" | "cpu" | "auto" (default: "auto")
    port:              int    HTTP port (default: 8103)
    host:              str    Bind address (default: "0.0.0.0")
    startup_timeout_s: float  Seconds allowed for a cold start (default: 900)
    model_cache:       str    NeMo + HF weight cache.  Resolved relative to this YAML.
                              Default: ../../models
"""
import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import warnings
from pathlib import Path

# Silence verbose third-party startup chatter that floods the launcher's
# terminal and the per-run log file. Set before any import that pulls in
# NeMo (which reads NEMO_LOGGING_LEVEL at import time), numexpr (reads
# NUMEXPR_MAX_THREADS), or pydub (whose import-time ffmpeg probe emits a
# RuntimeWarning). Users can override any of these via the env.
os.environ.setdefault("NEMO_LOGGING_LEVEL",  "ERROR")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")
warnings.filterwarnings(
    "ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning,
)

import yaml
from loguru import logger
from xr_ai_logging import setup_logging

_DEFAULT_PORT              = 8103
_DEFAULT_STARTUP_TIMEOUT_S = 900.0
_PROCESS_STOP_TIMEOUT_S    = 10.0


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../../models")
    p   = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


class _AsrBackend:
    """Thread-safe lazy loader for NeMo ASR models."""

    def __init__(self, model_name: str, device: str, model_cache: Path) -> None:
        self._model_name = model_name
        self._device     = device
        self._cache      = model_cache
        self._model      = None
        self._lock       = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            import nemo.collections.asr as nemo_asr

            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading NeMo ASR {!r} on {}…", self._model_name, device)
            # from_pretrained resolves the correct model subclass automatically.
            model = nemo_asr.models.ASRModel.from_pretrained(self._model_name)
            model.eval()
            if device == "cuda":
                model = model.cuda()
            self._model = model
            logger.info("ASR model ready.")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_path: str) -> str:
        """Synchronous. Call from a thread pool."""
        self._ensure_loaded()
        import torch
        # NeMo ASR .transcribe() is not re-entrant/thread-safe (shared model
        # buffers and, on CUDA, shared device state), and the endpoint
        # dispatches each request to a thread pool — serialize inference on the
        # shared model, mirroring the magpie TTS backend. _ensure_loaded() runs
        # BEFORE acquiring the lock because _lock is non-reentrant and
        # _ensure_loaded() takes it itself (the post-load fast path is lock-free).
        with self._lock:
            with torch.inference_mode():
                results = self._model.transcribe([audio_path], verbose=False)
        # NeMo returns a list of strings (or Hypothesis objects).
        if not results:
            return ""
        r = results[0]
        return str(r.text) if hasattr(r, "text") else str(r)


def _build_app(cfg: dict, model_cache: Path):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse

    model_name = cfg["model"]
    device     = cfg.get("device", "auto")

    backend = _AsrBackend(model_name, device, model_cache)

    app = FastAPI(title="STT Server", version="0.1.0")

    @app.get("/health")
    def health():
        if not backend.ready:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file:            UploadFile = File(...),
        response_format: str        = Form("json"),
        # model / language / temperature accepted for API compatibility but not used:
        # parakeet-tdt is English-only and deterministic.
    ):
        from fastapi import HTTPException
        audio_bytes = await file.read()
        suffix      = Path(file.filename or "audio.wav").suffix or ".wav"
        loop        = asyncio.get_running_loop()

        def _run() -> str:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                return backend.transcribe(tmp_path)
            finally:
                if tmp_path:
                    os.unlink(tmp_path)

        try:
            text = await loop.run_in_executor(None, _run)
        except Exception as exc:
            # Full detail goes to the server log only; the wire gets a stable
            # generic message so backend paths and runtime state don't leak.
            logger.exception("transcription failed: {}", exc)
            raise HTTPException(status_code=500, detail="transcription failed") from exc

        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text})

    return app, backend


def _health_ok(port: int) -> bool:
    """Return True if an STT server is already answering /health on *port*."""
    return _health_url_ok(f"http://127.0.0.1:{port}/health")


def _health_url_ok(health_url: str) -> bool:
    """Return True if *health_url* answers successfully."""
    try:
        with urllib.request.urlopen(health_url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


async def _run(cfg: dict, yaml_dir: Path, ready_file: Path | None = None) -> None:
    import uvicorn

    if not cfg.get("model"):
        logger.error("'model' is required in config")
        sys.exit(1)

    # GPU selection — set before any CUDA init.
    cuda_vis = cfg.get("cuda_visible_devices")
    if cuda_vis is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_vis)

    model_cache = _resolve_model_cache(cfg, yaml_dir)

    # Direct NeMo and HuggingFace to the shared model directory.
    os.environ["NEMO_CACHE_DIR"] = str(model_cache / "nemo")
    os.environ["HF_HOME"]        = str(model_cache / "huggingface")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    port = int(cfg.get("port", _DEFAULT_PORT))
    host = cfg.get("host", "0.0.0.0")

    # Reuse a server that survived a previous stack run (weight persistence).
    if _health_ok(port):
        logger.info("STT server already running on :{} — reusing", port)
        if ready_file:
            ready_file.touch()
        await asyncio.get_running_loop().run_in_executor(
            None, _idle_until_stopped, f"http://127.0.0.1:{port}/health"
        )
        return

    app, backend = _build_app(cfg, model_cache)

    # Load weights before serving so a bad model name / OOM crashes the
    # process at startup instead of surfacing as HTTP 500 on the first
    # client request. Doing this before uvicorn.Config() means the port
    # never opens if the model can't be loaded.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, backend._ensure_loaded)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    logger.info("Ready  →  http://localhost:{}/v1", port)
    if ready_file:
        ready_file.touch()
    await server.serve()
    logger.info("Stopped.")


def _wait_until_healthy(
    process: subprocess.Popen,
    health_url: str,
    timeout_s: float,
    poll_s: float = 1.0,
) -> None:
    """Wait for health while reporting a detached server crash immediately."""
    deadline = time.monotonic() + timeout_s
    while True:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"server exited with status {returncode} before becoming healthy"
            )
        if _health_url_ok(health_url):
            return

        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError(
                f"server did not become healthy within {timeout_s:g} seconds"
            )

        # Waiting on the process, rather than sleeping blindly, wakes the
        # wrapper as soon as a failed import, download, or CUDA load exits.
        try:
            returncode = process.wait(timeout=min(poll_s, remaining_s))
        except subprocess.TimeoutExpired:
            continue
        raise RuntimeError(
            f"server exited with status {returncode} before becoming healthy"
        )


def _terminate_process(
    process: subprocess.Popen,
    timeout_s: float = _PROCESS_STOP_TIMEOUT_S,
) -> None:
    """Terminate and reap a detached server, escalating if it does not stop."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _start_persistent_server(
    cmd: list[str],
    health_url: str,
    startup_timeout_s: float,
) -> subprocess.Popen:
    """Start the detached server and return it once its health check passes."""
    process = subprocess.Popen(cmd, start_new_session=True)
    try:
        _wait_until_healthy(process, health_url, startup_timeout_s)
    except TimeoutError:
        _terminate_process(process)
        raise
    return process


def _idle_until_stopped(
    health_url: str,
    process: subprocess.Popen | None = None,
    poll_s: float = 5.0,
) -> None:
    """Keep the wrapper alive while the persisted server is still running."""
    while True:
        if process is None:
            time.sleep(poll_s)
        else:
            try:
                returncode = process.wait(timeout=poll_s)
            except subprocess.TimeoutExpired:
                pass
            else:
                if returncode != 0:
                    raise SystemExit(
                        f"[stt_server] persistent server exited with status {returncode}"
                    )
                return

        if not _health_url_ok(health_url):
            # Any urlopen failure — connection refused, timeout, socket
            # error — means the persisted server is gone; exit so the
            # wrapper subprocess can be reaped.
            return


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    setup_logging("stt")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=Path, default=None)
    p.add_argument("--ready-file", type=Path, default=None)
    p.add_argument("--_serve",     action="store_true",
                   help=argparse.SUPPRESS)  # internal: actual server mode
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    yaml_dir  = Path.cwd()
    if ns.config and ns.config.exists():
        yaml_dir = ns.config.parent.resolve()
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    if ns._serve:
        # Persistent server subprocess — loads model and serves until killed.
        asyncio.run(_run(cfg, yaml_dir, ready_file=None))
        return

    # ── wrapper mode ──────────────────────────────────────────────────────────
    port              = int(cfg.get("port", _DEFAULT_PORT))
    startup_timeout_s = float(cfg.get("startup_timeout_s", _DEFAULT_STARTUP_TIMEOUT_S))
    health_url        = f"http://127.0.0.1:{port}/health"

    if startup_timeout_s <= 0:
        sys.exit("[stt_server] 'startup_timeout_s' must be greater than zero")

    # Reuse an already-running server (survived a previous stack shutdown).
    already_up = _health_url_ok(health_url)

    if already_up:
        print(f"[stt_server] already running on port {port} — reusing", flush=True)
        if ns.ready_file:
            ns.ready_file.touch()
        _idle_until_stopped(health_url)
        return

    # Spawn the actual server in its own process group so it outlives the
    # launcher's killpg when the stack shuts down.
    cmd = [sys.executable, "-m", "stt_server", "--_serve"]
    if ns.config:
        cmd += ["--config", str(ns.config)]
    print(
        f"[stt_server] starting persistent server on port {port} "
        f"(startup timeout: {startup_timeout_s:g}s)…",
        flush=True,
    )
    try:
        process = _start_persistent_server(cmd, health_url, startup_timeout_s)
    except (RuntimeError, TimeoutError) as exc:
        raise SystemExit(f"[stt_server] {exc}") from exc

    print(f"[stt_server] Ready  →  http://localhost:{port}/v1", flush=True)
    if ns.ready_file:
        ns.ready_file.touch()
    _idle_until_stopped(health_url, process)


if __name__ == "__main__":
    run()
