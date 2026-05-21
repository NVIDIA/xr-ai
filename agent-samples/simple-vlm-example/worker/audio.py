# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio helpers: int16 PCM ⇄ WAV; WAV → float32 AudioChunk."""
from __future__ import annotations

import io
import time
import wave

import numpy as np

from xr_ai_agent import AudioChunk


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
    """Decode a WAV blob into 20 ms float32 AudioChunks for the return path."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr  = wf.getframerate()
        ch  = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    chunk_frames = max(1, sr // 50)  # 20 ms
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
        pts += 20_000
    return out
