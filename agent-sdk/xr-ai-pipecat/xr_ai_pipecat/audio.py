# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Audio helpers: PCM ↔ WAV conversion, sentence-batched TTS, VAD bookkeeping.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import time
import wave

import numpy as np

from xr_ai_agent import AudioChunk, ProcessorEndpoint

log = logging.getLogger("xr_ai_pipecat.audio")

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_CHUNK_MS = 20
_CHUNK_US = _CHUNK_MS * 1_000


def now_us() -> int:
    return time.time_ns() // 1_000


def rms_float32(data: bytes) -> float:
    arr = np.frombuffer(data, dtype=np.float32)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0


def chunks_to_wav(chunks: list[AudioChunk]) -> bytes:
    """Concatenate float32 IPC chunks into a single 16-bit PCM WAV blob."""
    raw = b"".join(c.data for c in chunks)
    arr = np.frombuffer(raw, dtype=np.float32)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(chunks[0].channels)
        wf.setsampwidth(2)
        wf.setframerate(chunks[0].sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_to_chunks(wav_bytes: bytes, participant_id: str) -> list[AudioChunk]:
    """Decode a WAV blob into 20 ms float32 AudioChunks at native sample rate."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr  = wf.getframerate()
        ch  = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    chunk_frames = max(1, sr // (1000 // _CHUNK_MS))
    pts = now_us()
    out: list[AudioChunk] = []
    for i in range(0, len(arr), chunk_frames * ch):
        seg = arr[i : i + chunk_frames * ch]
        if not len(seg):
            break
        out.append(AudioChunk(
            pts_us=pts,
            sample_rate=sr,
            channels=ch,
            samples=len(seg) // ch,
            data=seg.tobytes(),
            participant_id=participant_id,
        ))
        pts += _CHUNK_US
    return out


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


async def stream_sentences_to_audio(
    endpoint: ProcessorEndpoint,
    tts_synth,
    text: str,
    participant_id: str,
) -> float:
    """Split *text* into sentences, synthesise in parallel, send in order."""
    sentences = split_sentences(text)
    if not sentences:
        return 0.0

    queue: asyncio.Queue = asyncio.Queue()
    total_samples = 0
    sample_rate   = 0

    async def _sender() -> None:
        nonlocal total_samples, sample_rate
        while True:
            task = await queue.get()
            if task is None:
                return
            try:
                wav = await task
            except Exception:
                log.exception("tts synth failed  pid=%r", participant_id)
                continue
            if not wav:
                continue
            try:
                for chunk in wav_to_chunks(wav, participant_id):
                    await endpoint.send_return_audio(chunk)
                    total_samples += chunk.samples
                    if sample_rate == 0:
                        sample_rate = chunk.sample_rate
            except Exception:
                log.exception("send_return_audio failed  pid=%r", participant_id)

    sender = asyncio.create_task(_sender(), name=f"tts-sender-{participant_id}")
    try:
        for i, sentence in enumerate(sentences):
            await queue.put(asyncio.create_task(
                tts_synth(sentence),
                name=f"tts-synth-{participant_id}-{i}",
            ))
    finally:
        await queue.put(None)
        if not sender.done():
            await asyncio.gather(sender)

    return (total_samples / sample_rate) if sample_rate else 0.0
