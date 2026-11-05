# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speaker-ASR streaming latency: endpointing delay and per-frame RPC cost, measured at real-time pace.

Speech is synthesized by the TTS service, resampled to 16 kHz, padded with silence, and fed at 1x real time
in fixed-size frames over the same private stream the voice pipeline uses. ``end_to_final`` is the wall time from
the instant the last speech sample was fed to the instant the final result arrives (``one_shot`` in ``always_on``
mode, ``transcript`` once a speaker is enrolled in ``conversation`` mode, which is what the samples use).
"""

from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from xr_ai_models import TTSService
from xr_ai_models._speaker_stream import _make_speaker_stream
from xr_ai_voicegate._speaker import _SpeakerConfig

from .._gpu import load_tags
from .._store import RunStore
from .._timing import now_ms

UTTERANCES = {
    "short": "Stop.",
    "medium": "What does the second gauge read right now?",
    "long": (
        "Please walk me through the pressure readings on the east wall, tell me which gauge is drifting, "
        "and remind me what the maintenance log says about the compressor housing."
    ),
}
_RATE = 16_000


SHORT_SET = {
    "yes": "Yes.", "no": "No.", "stop": "Stop.", "next": "Next.", "okay": "Okay.", "go_back": "Go back.",
    "stop_timer": "Stop the timer.", "what_time": "What time is it?",
}
_FINAL = {"one_shot", "transcript"}
_START_PHRASE = "Hey agent, let's start talking."


@dataclass
class SttPlan:
    """Sample counts and frame sizes (ms) to compare; larger frames trade latency for fewer RPCs."""

    samples: int = 5
    always_on_frame_ms: tuple[int, ...] = (10, 40, 160)
    conversation_frame_ms: tuple[int, ...] = (10,)
    lead_silence_s: float = 1.0
    trail_silence_s: float = 4.0
    base_url: str = "http://127.0.0.1:8102"  # overridden by the stack profile's services.speaker_stt
    short_set_only: bool = False


def _pcm16k(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        rate, frames = w.getframerate(), w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    if rate != _RATE:
        positions = np.linspace(0, samples.size - 1, int(samples.size * _RATE / rate))
        samples = np.interp(positions, np.arange(samples.size), samples)
    return samples.astype("<i2")


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * _RATE), "<i2")


@dataclass
class _Feed:
    """What one real-time feed observed."""

    events: dict[str, float]
    texts: dict[str, str]
    rtts: list[float]
    start: float
    speech_end_wall: float | None


async def _feed(stream: Any, pcm: np.ndarray, frame_ms: int, speech_end_sample: int | None, stop_on: set[str],
                ) -> _Feed:
    """Feed ``pcm`` at real-time pace, stopping once any event kind in ``stop_on`` arrives."""
    step = _RATE * frame_ms // 1000
    out = _Feed({}, {}, [], now_ms(), None)
    for offset in range(0, pcm.size - step + 1, step):
        due = out.start + (offset / _RATE) * 1000.0 + frame_ms  # a frame exists once its last sample was captured
        if (wait := due - now_ms()) > 0:
            await asyncio.sleep(wait / 1000.0)
        if speech_end_sample is not None and out.speech_end_wall is None and offset + step >= speech_end_sample:
            out.speech_end_wall = now_ms()
        sent = now_ms()
        result = await stream._feed(pcm[offset:offset + step].tobytes())
        received = now_ms()
        out.rtts.append(received - sent)
        for event in result:
            out.events.setdefault(event["kind"], received)
            if event.get("text"):
                out.texts[event["kind"]] = event["text"]
        if stop_on & out.events.keys():
            break
    return out


async def run(tts: TTSService, store: RunStore, plan: SttPlan) -> None:
    """Feed each utterance at real-time pace in both modes and record event times and per-frame RTT."""
    phrases = SHORT_SET if plan.short_set_only else UTTERANCES
    audio = {name: _pcm16k(await tts.synthesize(text, response_format="wav")) for name, text in phrases.items()}
    start_phrase = _pcm16k(await tts.synthesize(_START_PHRASE, response_format="wav"))
    base = asdict(_SpeakerConfig())
    modes = (("always_on", (10,)),) if plan.short_set_only else (
        ("always_on", plan.always_on_frame_ms), ("conversation", plan.conversation_frame_ms))
    for mode, frames in modes:
        config = {k: v for k, v in base.items() if k not in {"base_url", "backend", "timeout_s"}} | {"mode": mode}
        for name, speech in audio.items():
            for frame_ms in frames:
                for _ in range(plan.samples):
                    await _one(plan, store, config, mode, name, speech, start_phrase, frame_ms)


async def _one(plan: SttPlan, store: RunStore, config: dict, mode: str, name: str, speech: np.ndarray,
               start_phrase: np.ndarray, frame_ms: int) -> None:
    stream = _make_speaker_stream(plan.base_url, 10.0)
    await stream._open(config, 0)
    enroll_ms = None
    try:
        if mode == "conversation":
            enroll = np.concatenate([_silence(0.5), start_phrase, _silence(3.0)])
            begun = now_ms()
            fed = await _feed(stream, enroll, frame_ms, None, {"enrolled"})
            if "enrolled" not in fed.events:
                store.add("stt.speaker", {"utterance": name, "mode": mode, "frame_ms": frame_ms, **load_tags()},
                          enrolled=False)
                return
            enroll_ms = now_ms() - begun
        pcm = np.concatenate([_silence(plan.lead_silence_s), speech, _silence(plan.trail_silence_s)])
        speech_end = plan.lead_silence_s * _RATE + speech.size
        got = await _feed(stream, pcm, frame_ms, int(speech_end), _FINAL)
    finally:
        await stream._close()
    ref = got.speech_end_wall or got.start
    final_kind = next((k for k in _FINAL if k in got.events), None)
    rtts = np.asarray(got.rtts)
    store.add(
        "stt.speaker",
        {"utterance": name, "mode": mode, "frame_ms": frame_ms, "speech_s": round(speech.size / _RATE, 2),
         **load_tags()},
        enrolled=True, enroll_ms=enroll_ms,
        speech_start=(got.events["speech_start"] - got.start) if "speech_start" in got.events else None,
        first_partial=(got.events["partial"] - got.start) if "partial" in got.events else None,
        end_to_speech_stop=(got.events["speech_stop"] - ref) if "speech_stop" in got.events else None,
        end_to_final=(got.events[final_kind] - ref) if final_kind else None,
        final_kind=final_kind, transcript=got.texts.get(final_kind or "", ""),
        rtt_mean=float(rtts.mean()), rtt_p99=float(np.percentile(rtts, 99)), rtt_max=float(rtts.max()),
        late_frames=int((rtts > frame_ms).sum()), frames=int(rtts.size), events=sorted(got.events),
    )
