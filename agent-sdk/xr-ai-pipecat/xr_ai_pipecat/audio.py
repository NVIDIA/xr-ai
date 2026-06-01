# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat audio helpers — codec is re-exported from xr-ai-conversation.

The canonical home for the PCM ↔ WAV codec helpers is
``xr_ai_conversation.audio``; this module re-exports them for
backwards compatibility with existing pipecat consumers. The
pipecat-flavored ``stream_sentences_to_audio`` stays here because it
pushes pipecat-style audio frames and depends on
``xr_ai_agent.ProcessorEndpoint``.
"""
from __future__ import annotations

import asyncio

import numpy as np
from loguru import logger

from xr_ai_agent import ProcessorEndpoint

# Re-exported codec + sentence helpers — canonical location is
# ``xr_ai_conversation.audio``. Importing from here continues to work
# for backwards compatibility with workers that pre-date PR-C.
from xr_ai_conversation.audio import (
    chunks_to_wav,
    int16_pcm_to_wav,
    now_us,
    split_sentences,
    wav_to_chunks,
)


__all__ = [
    # re-exports
    "chunks_to_wav",
    "int16_pcm_to_wav",
    "now_us",
    "split_sentences",
    "wav_to_chunks",
    # pipecat-only
    "rms_float32",
    "stream_sentences_to_audio",
]


def rms_float32(data: bytes) -> float:
    """RMS amplitude of float32 PCM bytes — pipecat-only utility."""
    arr = np.frombuffer(data, dtype=np.float32)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0


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
                logger.exception("tts synth failed  pid={!r}", participant_id)
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
                logger.exception("send_return_audio failed  pid={!r}", participant_id)

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
