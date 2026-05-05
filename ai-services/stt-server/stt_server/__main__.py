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
from pathlib import Path

import yaml

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

            print(f"[stt_server] Loading NeMo ASR {self._model_name!r} on {device}…")
            # from_pretrained resolves the correct model subclass automatically.
            model = nemo_asr.models.ASRModel.from_pretrained(self._model_name)
            model.eval()
            if device == "cuda":
                model = model.cuda()
            self._model = model
            print("[stt_server] ASR model ready.")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_path: str) -> str:
        """Synchronous. Call from a thread pool."""
        self._ensure_loaded()
        import torch
        with torch.inference_mode():
            results = self._model.transcribe([audio_path])
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

        text = await loop.run_in_executor(None, _run)

        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text})

    return app, backend


async def _run(cfg: dict, yaml_dir: Path, ready_file: Path | None = None) -> None:
    import uvicorn

    if not cfg.get("model"):
        print("[stt_server] 'model' is required in config", file=sys.stderr)
        sys.exit(1)

    model_cache = _resolve_model_cache(cfg, yaml_dir)

    # Direct NeMo and HuggingFace to the shared model directory.
    os.environ["NEMO_CACHE_DIR"] = str(model_cache / "nemo")
    os.environ["HF_HOME"]        = str(model_cache / "huggingface")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    port = int(cfg.get("port", _DEFAULT_PORT))
    host = cfg.get("host", "0.0.0.0")

    app, backend = _build_app(cfg, model_cache)

    # Warm up at startup so first request is instant.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, backend._ensure_loaded)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    print(f"[stt_server] Ready  →  http://localhost:{port}/v1")
    if ready_file:
        ready_file.touch()
    await server.serve()
    print("[stt_server] Stopped.")


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

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
