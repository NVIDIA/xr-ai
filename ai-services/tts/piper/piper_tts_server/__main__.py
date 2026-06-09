# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
piper_tts_server — Piper TTS HTTP server.

Non-autoregressive ONNX-based TTS: ~50-150 ms per sentence on CPU,
vs. 2-5 s for autoregressive models like Magpie.  Drop-in replacement
for tts-server — serves the same OpenAI-compatible API:

    POST /v1/audio/speech
    GET  /v1/models
    GET  /health

Voices are ONNX models from the rhasspy/piper-voices HuggingFace repo.
The server downloads the requested voice on first startup if it is not
already present in model_cache.

Accepts --config <path>.yaml (auto-passed by xr-ai-launcher).

Config keys
-----------
    voice:        str   Piper voice name, e.g. "en_US-lessac-medium" (required)
    port:         int   HTTP port (default: 8105)
    host:         str   Bind address (default: "0.0.0.0")
    use_cuda:     bool  Run ONNX on CUDA (default: false — CPU is fast enough)
    model_cache:  str   Voice model cache path, resolved relative to this YAML.
                        Default: ../models
"""
import argparse
import io
import sys
import threading
import wave
from pathlib import Path

import yaml
from loguru import logger
from xr_ai_logging import setup_logging

_DEFAULT_PORT   = 8105
_HF_REPO        = "rhasspy/piper-voices"

# Exit code for "the voice could not be obtained for environmental reasons"
# (offline with an empty cache, or a transient HuggingFace download failure),
# as distinct from a genuine misconfiguration (unknown voice name → exit 1).
# Callers/tests can treat this as retry-or-skip rather than a hard failure.
_EXIT_VOICE_UNAVAILABLE = 3


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../models")
    p   = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hf_path_for_voice(voice: str) -> tuple[str, str]:
    """Return (onnx_hf_path, json_hf_path) within the rhasspy/piper-voices repo.

    Voice name format: <locale>-<speaker>-<quality>  e.g. en_US-lessac-medium
    HF tree structure: <lang>/<locale>/<speaker>/<quality>/<voice>.onnx[.json]
    """
    parts = voice.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"Voice name {voice!r} must be <locale>-<speaker>-<quality>, "
            "e.g. en_US-lessac-medium"
        )
    locale, speaker, quality = parts[0], parts[1], "-".join(parts[2:])
    lang = locale.split("_")[0]
    base = f"{lang}/{locale}/{speaker}/{quality}/{voice}"
    return f"{base}.onnx", f"{base}.onnx.json"


def _ensure_voice(voice: str, cache_dir: Path) -> tuple[Path, Path]:
    """Return (onnx_path, json_path), downloading from HF if necessary.

    Failures are surfaced as a clear single-line error and SystemExit at
    startup — without this, a typo in the voice name only manifests as a
    cryptic huggingface_hub stack trace on the first /v1/audio/speech call.

    Exit codes:
      1  unknown voice name / repo (a misconfiguration — fix the config).
      3  voice unavailable for environmental reasons: offline with an empty
         cache, or a transient HF download failure (network / rate-limit).
         Retryable, not a code bug — see ``_EXIT_VOICE_UNAVAILABLE``.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    hf_cache = cache_dir / "piper"
    hf_cache.mkdir(parents=True, exist_ok=True)
    onnx_hf, json_hf = _hf_path_for_voice(voice)
    logger.info("Resolving voice {!r} from {}…", voice, _HF_REPO)
    try:
        onnx_path = Path(hf_hub_download(_HF_REPO, onnx_hf, cache_dir=str(hf_cache)))
        json_path = Path(hf_hub_download(_HF_REPO, json_hf, cache_dir=str(hf_cache)))
    # LocalEntryNotFoundError subclasses EntryNotFoundError, so it MUST be caught
    # first — otherwise the (EntryNotFoundError, …) handler below swallows it and
    # exits 1 (hard fail) instead of the retryable _EXIT_VOICE_UNAVAILABLE. A
    # transient HF 429 with no cached copy surfaces as LocalEntryNotFoundError,
    # which is exactly the flake this ordering prevents.
    except LocalEntryNotFoundError:
        logger.error(
            "Voice {!r} is not cached in {} and could not be downloaded "
            "(HF_HUB_OFFLINE / no network / transient HF failure). Pre-fetch it "
            "on a connected host or retry.",
            voice, hf_cache,
        )
        sys.exit(_EXIT_VOICE_UNAVAILABLE)
    except (EntryNotFoundError, RepositoryNotFoundError) as exc:
        logger.error(
            "Voice {!r} not found in {} ({}). "
            "Check the voice name — format is <locale>-<speaker>-<quality>, "
            "e.g. en_US-lessac-medium.",
            voice, _HF_REPO, exc.__class__.__name__,
        )
        sys.exit(1)
    except Exception as exc:
        # Any other huggingface_hub error (HfHubHTTPError, connection reset,
        # read timeout, 429 rate-limit, …) is a transient download problem,
        # not a misconfiguration. Surface a clear single line and exit with the
        # retryable code instead of dumping a raw traceback as exit 1.
        logger.error(
            "Could not download voice {!r} from {} ({}: {}). This is usually a "
            "transient network or HuggingFace availability problem — retry, or "
            "pre-fetch the voice on a connected host.",
            voice, _HF_REPO, exc.__class__.__name__, exc,
        )
        sys.exit(_EXIT_VOICE_UNAVAILABLE)
    logger.info("Voice files ready")
    return onnx_path, json_path


class _PiperBackend:
    """Thread-safe Piper voice loader and synthesizer."""

    def __init__(self, voice: str, cache_dir: Path, use_cuda: bool) -> None:
        self._voice_name = voice
        self._cache_dir  = cache_dir
        self._use_cuda   = use_cuda
        self._voice      = None
        self._lock       = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._voice is not None:
            return
        with self._lock:
            if self._voice is not None:
                return
            from piper import PiperVoice

            onnx_path, json_path = _ensure_voice(self._voice_name, self._cache_dir)
            logger.info("Loading voice {!r}…", self._voice_name)
            self._voice = PiperVoice.load(
                str(onnx_path),
                config_path=str(json_path),
                use_cuda=self._use_cuda,
            )
            logger.info("Voice ready  sample_rate={}", self._voice.config.sample_rate)

    @property
    def ready(self) -> bool:
        return self._voice is not None

    @property
    def sample_rate(self) -> int:
        self._ensure_loaded()
        return self._voice.config.sample_rate

    def synthesize(self, text: str) -> bytes:
        """Synthesize text → WAV bytes. Synchronous — call from a thread pool."""
        self._ensure_loaded()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            if text.strip():
                self._voice.synthesize_wav(text, wf)
            else:
                # Empty/whitespace input yields no synthesized chunks, and
                # Piper's synthesize_wav sets the WAV format params only on the
                # first chunk — so they'd never be set and wave.close() would
                # raise "# channels not specified" (→ unhandled HTTP 500, #194).
                # Emit a valid, empty (silent) WAV instead, matching magpie.
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
        return buf.getvalue()


def _build_app(cfg: dict, model_cache: Path):
    import asyncio

    from fastapi import FastAPI
    from fastapi.responses import Response
    from pydantic import BaseModel

    voice_name = cfg["voice"]
    use_cuda   = bool(cfg.get("use_cuda", False))
    backend    = _PiperBackend(voice_name, model_cache, use_cuda)

    app = FastAPI(title="Piper TTS Server", version="0.1.0")

    class SpeechRequest(BaseModel):
        model:           str   = voice_name
        input:           str
        voice:           str   = "default"
        speed:           float = 1.0          # not yet supported by piper-tts Python API
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
            "data": [{"id": voice_name, "object": "model", "owned_by": "local"}],
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
    import asyncio
    import uvicorn

    if not cfg.get("voice"):
        logger.error("'voice' is required in config")
        sys.exit(1)

    model_cache = _resolve_model_cache(cfg, yaml_dir)
    port = int(cfg.get("port", _DEFAULT_PORT))
    host = cfg.get("host", "0.0.0.0")

    app, backend = _build_app(cfg, model_cache)

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
    import asyncio

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    setup_logging("tts-piper")

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
