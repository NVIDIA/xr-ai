# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
magpie_tts_server — OpenAI-compatible Text-to-Speech server.

Loads nvidia/magpie_tts_multilingual_357m (NeMo TTS) in-process and serves an
OpenAI-compatible speech API:

    POST /v1/audio/speech
    GET  /v1/models

Response formats: wav (default), pcm (raw signed-16-bit LE at sample_rate Hz)

Accepts --config <path>.yaml (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:          str   NeMo TTS model name (required)
    device:         str   "cuda" | "cpu" | "auto" (default: "auto")
    port:           int   HTTP port (default: 8104)
    host:           str   Bind address (default: "0.0.0.0")
    sample_rate:    int   Output sample rate Hz (default: 22050)
    model_cache:    str   NeMo + HF weight cache.  Resolved relative to this YAML.
                          Default: ../models
"""
import argparse
import asyncio
import io
import os
import sys
import threading
from pathlib import Path

import yaml
from loguru import logger
from xr_ai_logging import setup_logging

_DEFAULT_PORT        = 8104
_DEFAULT_SAMPLE_RATE = 22050  # NeMo FastPitch/VITS native rate


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../models")
    p   = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


class _TtsBackend:
    """Thread-safe lazy loader for NeMo TTS models."""

    def __init__(self, model_name: str, device: str, sample_rate: int,
                 model_cache: Path) -> None:
        self._model_name = model_name
        self._device     = device
        self._sample_rate = sample_rate
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
            from nemo.collections.tts.models.magpietts import MagpieTTSModel

            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading NeMo TTS {!r} on {}…", self._model_name, device)
            model = MagpieTTSModel.from_pretrained(self._model_name)
            model.eval()
            if device == "cuda":
                model = model.cuda()
            self._model = model
            logger.info("TTS model ready.")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def synthesize(self, text: str) -> bytes:
        """Synthesize text → WAV bytes. Synchronous — call from a thread pool."""
        import io as _io
        import soundfile as sf
        import torch

        self._ensure_loaded()

        # Magpie TTS is not re-entrant — serialize all synthesis calls.
        with self._lock:
            with torch.inference_mode():
                # do_tts returns (audio, audio_len): audio shape (1, T), audio_len shape (1,)
                audio, audio_len = self._model.do_tts(text)

        length   = int(audio_len[0].item())
        audio_np = audio[0, :length].cpu().float().numpy()
        buf = _io.BytesIO()
        sf.write(buf, audio_np, self._sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def _build_app(cfg: dict, model_cache: Path):
    import numpy as np
    from fastapi import FastAPI
    from fastapi.responses import Response
    from pydantic import BaseModel

    model_name  = cfg["model"]
    device      = cfg.get("device", "auto")
    sample_rate = int(cfg.get("sample_rate", _DEFAULT_SAMPLE_RATE))

    backend = _TtsBackend(model_name, device, sample_rate, model_cache)

    app = FastAPI(title="TTS Server", version="0.1.0")

    class SpeechRequest(BaseModel):
        model:           str   = model_name
        input:           str
        voice:           str   = "default"   # magpie is multilingual; voice ignored
        speed:           float = 1.0         # speed not yet supported
        response_format: str   = "wav"

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

    @app.post("/v1/audio/speech")
    async def synthesize(req: SpeechRequest):
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(None, backend.synthesize, req.input)

        if req.response_format == "pcm":
            import soundfile as sf
            audio, _ = sf.read(io.BytesIO(wav_bytes), dtype="int16")
            return Response(content=audio.tobytes(), media_type="audio/pcm")

        return Response(content=wav_bytes, media_type="audio/wav")

    return app, backend


async def _run(cfg: dict, yaml_dir: Path, ready_file: Path | None = None) -> None:
    import uvicorn

    if not cfg.get("model"):
        logger.error("'model' is required in config")
        sys.exit(1)

    model_cache = _resolve_model_cache(cfg, yaml_dir)

    os.environ["NEMO_CACHE_DIR"] = str(model_cache / "nemo")
    os.environ["HF_HOME"]        = str(model_cache / "huggingface")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    port = int(cfg.get("port", _DEFAULT_PORT))
    host = cfg.get("host", "0.0.0.0")

    app, backend = _build_app(cfg, model_cache)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, backend._ensure_loaded)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    logger.info("Ready  →  http://localhost:{}/v1", port)
    # Signal readiness to the launcher AFTER the model is loaded (above) so
    # _wait_ready unblocks. Without this the launcher blocks forever — the
    # process stays alive serving, so it never appears via proc.poll() either.
    if ready_file:
        ready_file.touch()
    await server.serve()
    logger.info("Stopped.")


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    setup_logging("tts-magpie")

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
