"""Audio helpers — float32 PCM → WAV and RMS."""
from __future__ import annotations

import io
import time
import wave

import numpy as np

from xr_ai_agent import AudioChunk


def now_us() -> int:
    return time.time_ns() // 1_000


def rms(data: bytes) -> float:
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
