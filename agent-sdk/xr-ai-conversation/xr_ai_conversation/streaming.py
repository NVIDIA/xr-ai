# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming token → sentence-batched TTS → audio sink pipeline.

The user's streaming ``on_query`` yields tokens; this helper buffers
them into sentence-sized units, spawns a TTS synth task per sentence so
multiple sentences synth in parallel, and pushes the resulting WAV
bytes to an :class:`AudioSink` in order via a single sender task.

Cancellation is cooperative: the caller is expected to ``await`` this
coroutine inside an ``asyncio.Task`` and cancel that task to interrupt.
On cancel, all pending synth tasks and the sender task are cancelled
and awaited before re-raising.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator, Awaitable, Callable

from xr_ai_models import TTSService
from xr_ai_voicegate import AudioSink


logger = logging.getLogger("xr_ai_conversation")


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


async def stream_text_to_audio(
    *,
    tokens:     AsyncIterator[str],
    pid:        str,
    tts:        TTSService,
    audio_sink: AudioSink,
    on_wav:     Callable[[bytes], None] | None = None,
) -> str:
    """Pipe ``tokens`` through sentence-batched TTS to ``audio_sink``.

    Returns the full accumulated text. ``on_wav`` is invoked synchronously
    for every WAV blob the sender consumes — the conversation loop wires
    it to ``gate.observe_tts_wav`` so the lazy listening-chime build can
    pick up the TTS sample rate from this path too.

    Cancellation contract: the caller owns the outer task. On
    ``asyncio.CancelledError`` here, every spawned synth task and the
    sender task are cancelled and awaited (so no orphaned TTS requests
    keep running) before the cancel re-raises.
    """
    full_text     = ""
    sentence_buf  = ""
    tts_queue: asyncio.Queue[asyncio.Task | None] = asyncio.Queue()
    pending_synth: list[asyncio.Task] = []

    async def _audio_sender() -> None:
        while True:
            task = await tts_queue.get()
            if task is None:
                return
            try:
                wav = await task
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("tts synth error pid=%r", pid)
                continue
            if on_wav is not None:
                try:
                    on_wav(wav)
                except Exception:
                    logger.exception("on_wav callback raised pid=%r", pid)
            try:
                await audio_sink.play_wav(pid, wav)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("audio sink play_wav error pid=%r", pid)

    sender = asyncio.create_task(_audio_sender(), name=f"tts-sender-{pid}")

    try:
        async for token in tokens:
            full_text    += token
            sentence_buf += token
            while True:
                m = _SENTENCE_BOUNDARY_RE.search(sentence_buf)
                if not m:
                    break
                sentence     = sentence_buf[: m.start() + 1].strip()
                sentence_buf = sentence_buf[m.end():]
                if sentence:
                    t = asyncio.create_task(tts.synthesize(sentence))
                    pending_synth.append(t)
                    await tts_queue.put(t)

        tail = sentence_buf.strip()
        if tail:
            t = asyncio.create_task(tts.synthesize(tail))
            pending_synth.append(t)
            await tts_queue.put(t)

        await tts_queue.put(None)
        await sender
        return full_text.strip()

    except asyncio.CancelledError:
        for t in pending_synth:
            t.cancel()
        sender.cancel()
        for t in (*pending_synth, sender):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        raise
