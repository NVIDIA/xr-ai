# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Listening-chime synthesis and WAV header parsing for the voice gate."""
from __future__ import annotations

import io
import wave

import numpy as np


def build_chime_wav(sample_rate: int) -> bytes:
    """Synthesize the listening-chime as a self-contained WAV blob.

    Two-tone perfect-fifth ding (880 + 1320 Hz) with exponential decay,
    ~250 ms total, mono int16 PCM. ``sample_rate`` MUST match the rate
    of the return audio track the consumer plays into (e.g. TTS sample
    rate) so the underlying transport doesn't reject frames at a
    different rate.

    Clamped to a sane audio range. A WAV header is uint32 and a hostile
    or corrupted blob can claim a multi-GHz rate, which would drive
    ``np.linspace`` into a multi-GB allocation. Reject outside [8 kHz,
    192 kHz] — the chime caller (``observe_tts_wav``) treats ValueError
    as "disable the chime", so the worst case is no chime.
    """
    if not 8000 <= sample_rate <= 192000:
        raise ValueError(f"unsupported chime sample rate: {sample_rate}")
    dur  = 0.25
    t    = np.linspace(0.0, dur, int(sample_rate * dur), endpoint=False, dtype=np.float32)
    tone = 0.55 * np.sin(2 * np.pi * 880.0 * t) + 0.30 * np.sin(2 * np.pi * 1320.0 * t)
    env  = np.exp(-t * 8.0).astype(np.float32)
    pcm  = (tone * env * 0.5 * 32767.0).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def read_wav_sample_rate(wav_bytes: bytes) -> int:
    """Pull the sample rate from a WAV blob without decoding the audio."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getframerate()
