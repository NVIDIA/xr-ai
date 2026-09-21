# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose HTTP transcription through the typed Riva STT client."""
from __future__ import annotations

import io
import wave

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from xr_ai_models import AdapterSpec, EndpointSpec, ModelsConfig, STTSpec, make_stt

from .common import create_app


def build_app(config: dict, *, backend=None):
    if backend is None:
        adapter = AdapterSpec(kind="riva_grpc", language=config["language"])
        endpoint = EndpointSpec(base_url=config["base_url"])
        backend = make_stt(ModelsConfig({"stt": STTSpec(adapter=adapter, endpoint=endpoint)}), "stt")
    app = create_app(config, [backend])
    @app.post("/v1/audio/transcriptions")
    async def transcribe(file: UploadFile = File(...), response_format: str = Form("json")):
        if response_format not in ("json", "text", "verbose_json"):
            raise HTTPException(400, "unsupported transcription response_format")
        audio = await file.read()
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getframerate() <= 0:
                    raise ValueError("expected mono 16-bit PCM WAV")
                duration = wav.getnframes() / wav.getframerate()
            text = await backend.transcribe(audio, timeout=120)
        except (ValueError, wave.Error, EOFError) as exc:
            raise HTTPException(400, "expected mono 16-bit PCM WAV") from exc
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "verbose_json":
            return {"text": text, "duration": duration, "language": config["language"]}
        return {"text": text}
    return app
