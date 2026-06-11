# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Audio helpers: PCM ↔ WAV conversion, float32 ↔ int16, VAD bookkeeping.
"""
from __future__ import annotations

import io
import time
import wave

import numpy as np
from pipecat.frames.frames import OutputAudioRawFrame

from xr_ai_agent import AudioChunk

_CHUNK_MS = 20
_CHUNK_US = _CHUNK_MS * 1_000


def now_us() -> int:
    return time.time_ns() // 1_000


def float32_to_int16(data: bytes) -> bytes:
    f32 = np.frombuffer(data, dtype=np.float32)
    return np.clip(f32 * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def int16_to_float32(data: bytes) -> bytes:
    i16 = np.frombuffer(data, dtype=np.int16)
    return (i16.astype(np.float32) / 32767.0).tobytes()


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


def wav_to_output_frames(wav_bytes: bytes, pid: str) -> list[OutputAudioRawFrame]:
    """Decode a WAV blob into 20 ms int16 ``OutputAudioRawFrame``s for *pid*.

    ``wav_to_chunks`` yields float32 ``AudioChunk``s; pipecat's output path
    expects int16 PCM, so each chunk is converted here and stamped with
    ``transport_destination`` so the output transport knows which participant
    to address. Raises on a malformed WAV — callers log and skip.
    """
    frames: list[OutputAudioRawFrame] = []
    for c in wav_to_chunks(wav_bytes, pid):
        out = OutputAudioRawFrame(
            audio        = float32_to_int16(c.data),
            sample_rate  = c.sample_rate,
            num_channels = c.channels,
        )
        out.transport_destination = pid
        frames.append(out)
    return frames
