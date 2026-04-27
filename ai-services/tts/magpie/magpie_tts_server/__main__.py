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

            print(f"[magpie_tts_server] Loading NeMo TTS {self._model_name!r} on {device}…")
            model = MagpieTTSModel.from_pretrained(self._model_name)
            model.eval()
            if device == "cuda":
                model = model.cuda()
            self._model = model
            print("[magpie_tts_server] TTS model ready.")

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


async def _run(cfg: dict, yaml_dir: Path) -> None:
    import uvicorn

    if not cfg.get("model"):
        print("[magpie_tts_server] 'model' is required in config", file=sys.stderr)
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

    print(f"[magpie_tts_server] Ready  →  http://localhost:{port}/v1")
    await server.serve()
    print("[magpie_tts_server] Stopped.")


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    yaml_dir  = Path.cwd()
    if ns.config and ns.config.exists():
        yaml_dir = ns.config.parent.resolve()
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(_run(cfg, yaml_dir))


if __name__ == "__main__":
    run()
