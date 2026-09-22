# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise a running NIM stack through XR AI's typed model clients."""
from __future__ import annotations

import argparse
import asyncio
import audioop
import io
import math
import struct
import wave
import zlib
from contextlib import AsyncExitStack
from pathlib import Path

from xr_ai_models import (
    ChatMessage,
    load_models_config,
    make_embedding,
    make_llm,
    make_stt,
    make_tts,
    make_vlm,
)


def _red_image() -> bytes:
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
    pixels = (b"\x00" + b"\xff\x00\x00" * 64) * 64
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def _wav_16khz(audio: bytes) -> bytes:
    with wave.open(io.BytesIO(audio), "rb") as source:
        width, channels, rate = source.getsampwidth(), source.getnchannels(), source.getframerate()
        pcm = source.readframes(source.getnframes())
    if width != 2 or channels != 1 or not pcm:
        raise ValueError("expected nonempty mono 16-bit speech from Magpie")
    pcm, _ = audioop.ratecv(pcm, width, channels, rate, 16000, None)
    result = io.BytesIO()
    with wave.open(result, "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(pcm)
    return result.getvalue()


async def check(profile: Path) -> None:
    """Check every role, including TTS-to-STT and image inference; fail on errors."""
    config = load_models_config(profile)
    async with AsyncExitStack() as stack:
        clients = {}
        for name, factory in (("stt", make_stt), ("tts", make_tts), ("llm", make_llm),
                              ("agent_llm", make_llm), ("vlm", make_vlm), ("embedding", make_embedding)):
            if name not in config.entries:
                continue
            client = factory(config, name)
            stack.push_async_callback(client.close)
            if not await client.health():
                raise RuntimeError(f"{name} is not ready")
            clients[name] = client

        if "tts" in clients and "stt" in clients:
            speech = await clients["tts"].synthesize("The model server is ready.", timeout=120)
            transcript = await clients["stt"].transcribe(_wav_16khz(speech), timeout=120)
            if not transcript.strip():
                raise RuntimeError("STT returned an empty transcript for Magpie speech")
            print(f"PASS speech round trip: {transcript}", flush=True)
            chunks = [chunk async for chunk in clients["tts"].stream("Streaming speech is ready.", timeout=120)]
            if not chunks or not any(chunk.data for chunk in chunks):
                raise RuntimeError("TTS returned no PCM speech")
            if any(chunk.sample_rate != 44100 or chunk.channels != 1 or len(chunk.data) % 2 for chunk in chunks):
                raise RuntimeError("unexpected Magpie PCM format or metadata")
            print("PASS Pocket client PCM response: 44100 Hz, mono, 16-bit", flush=True)

        for role in ("llm", "agent_llm"):
            if role not in clients:
                continue
            response = await clients[role].chat(
                [ChatMessage("user", "Reply with the word ready.")], max_tokens=64, timeout=120,
            )
            if not response.content.strip():
                raise RuntimeError(f"{role} returned no visible text")
            print(f"PASS {role}: {response.content.strip()}")

        if "vlm" in clients:
            response = await clients["vlm"].ask_image(
                _red_image(), "What color is this image? Answer briefly.", max_tokens=128, timeout=120,
            )
            if not response.content.strip():
                raise RuntimeError("VLM returned no visible text for an image request")
            print(f"PASS image: {response.content.strip()}", flush=True)
            streamed = "".join([text async for text in clients["vlm"].stream(
                _red_image(), "What color is this image? Answer briefly.", max_tokens=128, timeout=120,
            )])
            if not streamed.strip():
                raise RuntimeError("VLM returned no streamed text for an image request")
            print(f"PASS streamed image: {streamed.strip()}", flush=True)

        if "embedding" in clients:
            vectors = await clients["embedding"].embed([
                "query: What is XR AI?", "passage: XR AI connects multimodal agents to XR clients.",
            ])
            if len(vectors) != 2 or any(len(vector) != 2048 for vector in vectors):
                raise RuntimeError("unexpected embedding dimensions")
            if any(not all(math.isfinite(value) for value in vector) or not any(vector) for vector in vectors):
                raise RuntimeError("embedding vector contains invalid values")
            print("PASS embeddings: two finite, nonzero 2048-dimensional vectors", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=Path,
        default=Path(__file__).resolve().parent.parent / "model-servers/yaml/models.default.json",
        help="Existing consumer models JSON; defaults to the original model-servers profile.",
    )
    asyncio.run(check(parser.parse_args().models))
