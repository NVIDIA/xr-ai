# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose Pocket-compatible HTTP speech through the typed Riva client."""
from __future__ import annotations

import asyncio
import io
import wave
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from xr_ai_models import AdapterSpec, EndpointSpec, ModelsConfig, TTSSpec, make_tts

from .common import create_app
from .streaming_speech import SpeechStreamResponse


class SpeechRequest(BaseModel):
    input: str
    response_format: Literal["wav", "pcm"] = "wav"
    stream: bool = False


def build_app(config: dict, *, backend=None):
    rate = int(config.get("sample_rate", 44100))
    if backend is None:
        adapter = AdapterSpec(kind="riva_grpc", language=config["language"],
                              voice=config["voice"], sample_rate=rate)
        endpoint = EndpointSpec(base_url=config["base_url"])
        backend = make_tts(ModelsConfig({"tts": TTSSpec(adapter=adapter, endpoint=endpoint)}), "tts")
    app = create_app(config, [backend])
    generation_lock = asyncio.Lock()

    pause_ms = config.get("post_synthesis_pause_ms", 300)
    if isinstance(pause_ms, bool) or not isinstance(pause_ms, int) or pause_ms < 0:
        raise ValueError("post_synthesis_pause_ms must be a non-negative integer")
    trailing_silence = b"\x00\x00" * round(rate * pause_ms / 1000)

    @app.post("/v1/audio/speech")
    async def synthesize(request: SpeechRequest):
        if request.stream and request.response_format != "pcm":
            raise HTTPException(400, "streaming speech requires response_format 'pcm'")
        if request.stream:
            return SpeechStreamResponse(backend, request.input, rate, generation_lock,
                                        trailing_silence=trailing_silence)
        async with generation_lock:
            audio = await backend.synthesize(request.input, response_format="pcm")
        if audio:
            audio += trailing_silence
        if request.response_format == "wav":
            output = io.BytesIO()
            with wave.open(output, "wb") as wav:
                wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                wav.writeframes(audio)
            audio = output.getvalue()
        media_type = "audio/pcm" if request.response_format == "pcm" else "audio/wav"
        return Response(audio, media_type=media_type)
    return app
