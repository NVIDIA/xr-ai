# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming vs non-streaming: where each wins, and what it does to time-to-first-audio."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from xr_ai_models import ChatMessage, LLMService, ToolDef, TTSService

from .._gpu import load_tags
from .._store import RunStore
from .._timing import now_ms, timed_chat, timed_stream

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SENTENCE_COUNTS = (1, 2, 4, 8)
_PROSE = (
    "Write exactly {n} sentences of about 15 words each describing how a technician inspects an industrial "
    "pressure gauge. Output only the sentences."
)
_TOOL = ToolDef(
    name="read_gauge",
    description="Read the current value of a named gauge.",
    parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
)


@dataclass
class StreamingPlan:
    """Sample counts per experiment."""

    samples: int = 12
    sentences: tuple[int, ...] = SENTENCE_COUNTS
    skip: frozenset[str] = frozenset()


async def run(llm: LLMService, tts: TTSService, store: RunStore, plan: StreamingPlan) -> None:
    """Run the four streaming experiments."""
    if "modes" not in plan.skip:
        await _modes(llm, store, plan)
    if "tools" not in plan.skip:
        await _tool_calls(llm, store, plan)
    if "short" not in plan.skip:
        await _short_outputs(llm, store, plan)
    if "audio" not in plan.skip:
        await _first_audio(llm, tts, store, plan)


def _case(**kw: object) -> dict[str, object]:
    return {**kw, **load_tags()}


async def _plain_stream(llm: LLMService, messages: list[ChatMessage], **kw: object) -> tuple[float, float, int]:
    """``stream()`` (text only): returns first-text ms, total ms, chunk count."""
    start, first, chunks = now_ms(), None, 0
    async for _ in llm.stream(messages, **kw):
        if first is None:
            first = now_ms() - start
        chunks += 1
    return first or 0.0, now_ms() - start, chunks


async def _modes(llm: LLMService, store: RunStore, plan: StreamingPlan) -> None:
    """Same prose request through chat / stream / stream_events: first text, first sentence, total."""
    for n in plan.sentences:
        msg = ChatMessage(role="user", content=_PROSE.format(n=n))
        kw = {"max_tokens": 700, "enable_thinking": False, "temperature": 0.0}
        for _ in range(plan.samples):
            c = await timed_chat(llm, [msg], **kw)
            store.add("stream.modes", _case(sentences=n, mode="chat"), **c.as_metrics())
            first, total, chunks = await _plain_stream(llm, [msg], **kw)
            store.add("stream.modes", _case(sentences=n, mode="stream"), first_text=first, total=total,
                      text_chunks=chunks)
            t = await timed_stream(llm, [msg], **kw)
            store.add("stream.modes", _case(sentences=n, mode="stream_events"), **t.as_metrics())


async def _tool_calls(llm: LLMService, store: RunStore, plan: StreamingPlan) -> None:
    """A turn that must end in a tool call: tool calls are held to stream end, so streaming cannot help."""
    msg = ChatMessage(role="user", content="Read the gauge named boiler_pressure using the tool.")
    kw = {"tools": [_TOOL], "max_tokens": 200, "enable_thinking": False, "temperature": 0.0}
    for _ in range(plan.samples):
        c = await timed_chat(llm, [msg], **kw)
        store.add("stream.tools", _case(mode="chat"), **c.as_metrics())
        t = await timed_stream(llm, [msg], **kw)
        store.add("stream.tools", _case(mode="stream_events"), **t.as_metrics())


async def _short_outputs(llm: LLMService, store: RunStore, plan: StreamingPlan) -> None:
    """Very short outputs (classification-like): fixed streaming overhead vs nothing to overlap."""
    msg = ChatMessage(role="user", content="Answer with exactly one word: is water wet? yes or no.")
    for tokens in (2, 4, 8):
        kw = {"max_tokens": tokens, "enable_thinking": False, "temperature": 0.0}
        for _ in range(plan.samples * 2):
            c = await timed_chat(llm, [msg], **kw)
            store.add("stream.short", _case(max_tokens=tokens, mode="chat"), **c.as_metrics())
            t = await timed_stream(llm, [msg], **kw)
            store.add("stream.short", _case(max_tokens=tokens, mode="stream_events"), **t.as_metrics())


async def _speak_stream(tts: TTSService, text: str) -> tuple[float, float, float]:
    """Stream TTS for ``text``: first-chunk ms since the call, audio seconds, absolute first-chunk clock ms."""
    start, first, first_abs, pcm, rate = now_ms(), None, 0.0, 0, 24_000
    async for chunk in tts.stream(text):  # type: ignore[attr-defined]
        if first is None:
            first_abs = now_ms()
            first = first_abs - start
        pcm += len(chunk.data)
        rate = chunk.sample_rate
    return first or 0.0, pcm / 2 / rate, first_abs


async def _first_audio(llm: LLMService, tts: TTSService, store: RunStore, plan: StreamingPlan) -> None:
    """Time to first audio under four strategies, measured through the real LLM and TTS services.

    S1 chat -> batch TTS of the whole text; S2 chat -> streamed TTS of the whole text;
    S3 stream_events -> sentence chunker -> streamed TTS per sentence (what StreamingTtsProcessor does);
    S4 is S3 plus continuity: the gap between when the previous sentence finishes playing and when the next
    sentence's first audio exists (positive gap = audible stall).
    """
    kw = {"max_tokens": 700, "enable_thinking": False, "temperature": 0.0}
    for n in plan.sentences:
        msg = ChatMessage(role="user", content=_PROSE.format(n=n))
        for _ in range(plan.samples):
            t0 = now_ms()
            text = (await llm.chat([msg], **kw)).content.strip()
            t_llm = now_ms() - t0
            wav = await tts.synthesize(text, response_format="wav")
            s1 = now_ms() - t0
            store.add("stream.audio", _case(sentences=n, strategy="S1_chat_batch_tts"), first_audio=s1, llm_done=t_llm,
                      wav_bytes=len(wav))
            t0 = now_ms()
            text = (await llm.chat([msg], **kw)).content.strip()
            t_llm = now_ms() - t0
            first, secs, _ = await _speak_stream(tts, text)
            store.add("stream.audio", _case(sentences=n, strategy="S2_chat_stream_tts"), first_audio=t_llm + first,
                      llm_done=t_llm, audio_s=secs)
            await _sentence_pipeline(llm, tts, store, msg, kw, n)


async def _sentence_pipeline(llm: LLMService, tts: TTSService, store: RunStore, msg: ChatMessage,
                             kw: dict[str, object], n: int) -> None:
    """S3/S4: synthesize each sentence as soon as the chunker releases it, strictly in order."""
    t0 = now_ms()
    sentences: asyncio.Queue[tuple[float, str] | None] = asyncio.Queue()
    ready_audio: list[tuple[float, float]] = []  # (first-audio time, seconds) per sentence, in order

    async def produce() -> None:
        buffer = ""
        async for piece in llm.stream([msg], **kw):
            buffer += piece
            parts = _SENTENCE_END.split(buffer)
            for done in parts[:-1]:
                await sentences.put((now_ms() - t0, done))
            buffer = parts[-1]
        if buffer.strip():
            await sentences.put((now_ms() - t0, buffer))
        await sentences.put(None)

    async def consume() -> None:
        while (item := await sentences.get()) is not None:
            _, text = item
            _, secs, first_abs = await _speak_stream(tts, text)
            ready_audio.append((first_abs - t0, secs))

    await asyncio.gather(produce(), consume())
    if not ready_audio:
        return
    stalls, play_end = [], ready_audio[0][0] + ready_audio[0][1] * 1000.0
    for first_at, secs in ready_audio[1:]:
        stalls.append(max(first_at - play_end, 0.0))
        play_end = max(play_end, first_at) + secs * 1000.0
    store.add("stream.audio", _case(sentences=n, strategy="S3_stream_sentence_tts"), first_audio=ready_audio[0][0],
              sentences_spoken=len(ready_audio), total_stall=sum(stalls), max_stall=max(stalls, default=0.0))
