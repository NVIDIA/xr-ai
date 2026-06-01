# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio codec helpers shared by the conversation loop.

* ``int16_pcm_to_wav`` — wrap raw int16 PCM in a single-channel WAV for STT.
* ``wav_to_chunks``    — decode a WAV blob into 20 ms float32
  ``AudioChunk``s for the hub's return-audio fan-out.
* ``chunks_to_wav``    — inverse of ``wav_to_chunks`` for tests and
  hub-tap recording paths.
* ``split_sentences``  — break streamed VLM/LLM output on sentence
  boundaries so each sentence can synthesize in parallel.
* ``now_us``           — microsecond wall clock used as the ``pts_us``
  default for canned replies.
"""
from __future__ import annotations

import io
import re
import time
import wave

import numpy as np

from xr_ai_agent import AudioChunk


_CHUNK_MS = 20
_CHUNK_US = _CHUNK_MS * 1_000

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def now_us() -> int:
    return time.time_ns() // 1_000


def int16_pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw int16 PCM bytes in a single-channel WAV container for STT."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def wav_to_chunks(wav_bytes: bytes, participant_id: str) -> list[AudioChunk]:
    """Decode a WAV blob into 20 ms float32 ``AudioChunk``s at native rate."""
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
            pts_us=pts, sample_rate=sr, channels=ch,
            samples=len(seg) // ch, data=seg.tobytes(),
            participant_id=participant_id,
        ))
        pts += _CHUNK_US
    return out


def chunks_to_wav(chunks: list[AudioChunk]) -> bytes:
    """Concatenate float32 ``AudioChunk``s into a single 16-bit PCM WAV blob.

    Inverse of ``wav_to_chunks``; useful for tests that need to round-trip
    audio through the return-audio path and for hub-tap recording. Raises
    ``ValueError`` on an empty list — there is no sane sample rate to pick."""
    if not chunks:
        raise ValueError("chunks_to_wav requires at least one chunk")
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


def split_sentences(text: str) -> list[str]:
    """Split ``text`` on sentence boundaries (``.``, ``!``, ``?`` + whitespace).

    Returns the non-empty stripped sentences in order. The streaming TTS
    path uses the lower-level regex inline to peel sentences as tokens
    arrive; this helper is for callers that already have the full text."""
    parts = SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]
