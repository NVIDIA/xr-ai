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
    model:        str   NeMo / HuggingFace model name (required)
    device:       str   "cuda" | "cpu" | "auto" (default: "auto")
    port:         int   HTTP port (default: 8103)
    host:         str   Bind address (default: "0.0.0.0")
    model_cache:  str   NeMo + HF weight cache.  Resolved relative to this YAML.
                        Default: ../models
"""
import argparse
import asyncio
import os
import sys
import tempfile
import threading
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

_DEFAULT_PORT = 8103


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../models")
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
            logger.exception("transcription failed: {}", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text})

    return app, backend


def _health_ok(port: int) -> bool:
    """Return True if an STT server is already answering /health on *port*."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


async def _idle_until_stopped(port: int) -> None:
    """Return when /health stops responding (server shut down externally)."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(5.0)
        alive = await loop.run_in_executor(None, _health_ok, port)
        if not alive:
            break


async def _run(cfg: dict, yaml_dir: Path, ready_file: Path | None = None) -> None:
    import uvicorn

    if not cfg.get("model"):
        logger.error("'model' is required in config")
        sys.exit(1)

    model_cache = _resolve_model_cache(cfg, yaml_dir)

    # GPU selection — set before any CUDA init.
    cuda_vis = cfg.get("cuda_visible_devices")
    if cuda_vis is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_vis)

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
        await _idle_until_stopped(port)
        return

    app, backend = _build_app(cfg, model_cache)

    # Warm up at startup so first request is instant.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, backend._ensure_loaded)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    logger.info("Ready  →  http://localhost:{}/v1", port)
    if ready_file:
        ready_file.touch()
    await server.serve()
    logger.info("Stopped.")


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    setup_logging("stt")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config",     type=Path, default=None)
    p.add_argument("--ready-file", type=Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    yaml_dir  = Path.cwd()
    if ns.config and ns.config.exists():
        yaml_dir = ns.config.parent.resolve()
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(_run(cfg, yaml_dir, ready_file=ns.ready_file))


if __name__ == "__main__":
    run()
