"""Audio helpers: float32 PCM ⇄ WAV, RMS for VAD."""
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
    """Concatenate float32 IPC chunks into a single 16-bit PCM WAV blob for STT."""
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
