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

_DEFAULT_PORT   = 8105
_HF_REPO        = "rhasspy/piper-voices"


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
    """Return (onnx_path, json_path), downloading from HF if necessary."""
    from huggingface_hub import hf_hub_download

    hf_cache = cache_dir / "piper"
    hf_cache.mkdir(parents=True, exist_ok=True)
    onnx_hf, json_hf = _hf_path_for_voice(voice)
    print(f"[piper_tts_server] Resolving voice {voice!r} from {_HF_REPO}…")
    onnx_path = Path(hf_hub_download(_HF_REPO, onnx_hf, cache_dir=str(hf_cache)))
    json_path = Path(hf_hub_download(_HF_REPO, json_hf, cache_dir=str(hf_cache)))
    print(f"[piper_tts_server] Voice files ready")
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
            print(f"[piper_tts_server] Loading voice {self._voice_name!r}…")
            self._voice = PiperVoice.load(
                str(onnx_path),
                config_path=str(json_path),
                use_cuda=self._use_cuda,
            )
            print(f"[piper_tts_server] Voice ready  "
                  f"sample_rate={self._voice.config.sample_rate}")

    @property
    def sample_rate(self) -> int:
        self._ensure_loaded()
        return self._voice.config.sample_rate

    def synthesize(self, text: str) -> bytes:
        """Synthesize text → WAV bytes. Synchronous — call from a thread pool."""
        self._ensure_loaded()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize_wav(text, wf)
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


async def _run(cfg: dict, yaml_dir: Path) -> None:
    import asyncio
    import uvicorn

    if not cfg.get("voice"):
        print("[piper_tts_server] 'voice' is required in config", file=sys.stderr)
        sys.exit(1)

    model_cache = _resolve_model_cache(cfg, yaml_dir)
    port = int(cfg.get("port", _DEFAULT_PORT))
    host = cfg.get("host", "0.0.0.0")

    app, backend = _build_app(cfg, model_cache)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, backend._ensure_loaded)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    print(f"[piper_tts_server] Ready  →  http://localhost:{port}/v1")
    await server.serve()
    print("[piper_tts_server] Stopped.")


def run() -> None:
    import asyncio

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
