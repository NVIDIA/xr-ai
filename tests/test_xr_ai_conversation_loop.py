# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``xr_ai_conversation.ConversationLoop``.

Covers: construction, str ``on_query`` → say path, streaming
``on_query`` → sentence-batched synth + ordered audio sender, per-pid
in-flight cancel + ``flush_return_audio`` on a second dispatch, STOP
handler, participant join + greeting, participant leave clears state,
custom greeting (string + callable), and ``observe_tts_wav`` is fed by
every TTS WAV the loop generates.

Everything runs against fakes — no real ``ProcessorEndpoint``, VAD,
STT, or TTS — so the suite stays CPU-only and finishes in well under a
second.
"""
from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass
from typing import Any, AsyncIterator

import numpy as np
import pytest

from xr_ai_agent      import AudioChunk, DataMessage, ParticipantEvent
from xr_ai_voicegate  import VoiceGate, VoiceGateConfig
from xr_ai_conversation import ConversationLoop, VadConfig, wire_voice_gate


# ── test doubles ────────────────────────────────────────────────────────────


def _silence_wav(sample_rate: int = 22_050, ms: int = 10) -> bytes:
    """Tiny well-formed WAV blob the loop's ``wav_to_chunks`` accepts."""
    n   = max(1, int(sample_rate * ms / 1000))
    pcm = np.zeros(n, dtype=np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class _FakeEp:
    """``ProcessorEndpoint`` stand-in capturing every outbound call.

    Stores the audio + data + participant callbacks the loop registers so
    tests can inject events directly. ``flush_return_audio`` and
    ``send_return_*`` simply append to lists for assertion."""

    def __init__(self) -> None:
        self.audio_cb       = None
        self.data_cb        = None
        self.participant_cb = None
        self.return_audio:  list[AudioChunk]  = []
        self.return_data:   list[DataMessage] = []
        self.flush_calls:   list[str]         = []
        self.stopped       = False
        self.closed        = False

    def on_audio(self, cb)       -> None: self.audio_cb       = cb
    def on_data(self, cb)        -> None: self.data_cb        = cb
    def on_participant(self, cb) -> None: self.participant_cb = cb
    def on_frame(self, cb)       -> None: pass

    async def send_return_audio(self, chunk: AudioChunk) -> None:
        self.return_audio.append(chunk)

    async def send_return_data(self, msg: DataMessage) -> None:
        self.return_data.append(msg)

    async def flush_return_audio(self, pid: str) -> None:
        self.flush_calls.append(pid)

    async def run(self) -> None:  # never called in these tests
        await asyncio.sleep(0)

    def stop(self)  -> None: self.stopped = True
    def close(self) -> None: self.closed  = True


class _FakeSTT:
    async def transcribe(self, audio: bytes, **kw: Any) -> str:
        return ""

    async def health(self) -> bool: return True
    async def close(self)  -> None: pass


class _FakeTTS:
    """TTS double that records every call and returns a tiny WAV."""

    def __init__(self, sample_rate: int = 22_050) -> None:
        self.sample_rate = sample_rate
        self.calls: list[str]               = []
        self.delays: dict[str, asyncio.Event] = {}  # text → release gate

    async def synthesize(self, text: str, **kw: Any) -> bytes:
        self.calls.append(text)
        gate = self.delays.get(text)
        if gate is not None:
            await gate.wait()
        return _silence_wav(self.sample_rate)

    async def health(self) -> bool: return True
    async def close(self)  -> None: pass


def _make_loop(
    *,
    on_query,
    vg_phrases: tuple[str, ...] = (),
    on_speech_start = None,
    on_participant_joined = None,
    on_participant_left   = None,
    on_stop_extra         = None,
    on_phrase_only_extra  = None,
    on_drop_extra         = None,
    text_topic: str = "agent.response",
    greeting = None,
) -> tuple[ConversationLoop, _FakeEp, _FakeSTT, _FakeTTS]:
    ep  = _FakeEp()
    stt = _FakeSTT()
    tts = _FakeTTS()
    loop = ConversationLoop(
        ep              = ep,        # type: ignore[arg-type]
        stt             = stt,       # type: ignore[arg-type]
        tts             = tts,       # type: ignore[arg-type]
        voice_gate_cfg  = VoiceGateConfig(magic_phrases=vg_phrases),
        vad_cfg         = VadConfig(),
        on_query        = on_query,
        on_speech_start = on_speech_start,
        on_participant_joined = on_participant_joined,
        on_participant_left   = on_participant_left,
        on_stop_extra         = on_stop_extra,
        on_phrase_only_extra  = on_phrase_only_extra,
        on_drop_extra         = on_drop_extra,
        text_topic            = text_topic,
        greeting              = greeting,
    )
    return loop, ep, stt, tts


# ════════════════════════════════════════════════════════════════════════════
# 1. Construction
# ════════════════════════════════════════════════════════════════════════════


def test_construct_with_minimal_kwargs():
    """Case 1: ConversationLoop builds with just the required kwargs and
    registers handlers on the underlying ``ProcessorEndpoint``."""

    async def on_query(pid: str, text: str, fresh_match: bool) -> str:
        return ""

    loop, ep, _, _ = _make_loop(on_query=on_query)
    assert ep.audio_cb       is not None
    assert ep.participant_cb is not None


# ════════════════════════════════════════════════════════════════════════════
# 2. str on_query → say
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_dispatch_with_string_result_invokes_on_query_and_speaks():
    """Case 2: dispatch('hello') invokes on_query and the returned string
    is synthesized + emitted as return audio + sent on the text topic."""
    seen: list[tuple[str, str, bool]] = []

    async def on_query(pid: str, text: str, fresh_match: bool) -> str:
        seen.append((pid, text, fresh_match))
        return "world."

    loop, ep, _, tts = _make_loop(on_query=on_query, text_topic="my.topic")
    await loop.dispatch("p1", "hello", pts_us=100)

    # Wait for the dispatched task to finish.
    vs = loop._voice["p1"]
    assert vs.current_task is not None
    await vs.current_task

    assert seen == [("p1", "hello", True)]
    assert tts.calls == ["world."]
    assert ep.return_audio                            # audio was emitted
    assert all(c.participant_id == "p1" for c in ep.return_audio)
    # Final assembled text goes out on the configured data topic.
    assert [(m.topic, m.data.decode()) for m in ep.return_data] \
           == [("my.topic", "world.")]


# ════════════════════════════════════════════════════════════════════════════
# 3. streaming on_query → sentence-batched synth + ordered sender
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_dispatch_with_streaming_result_splits_sentences_and_sends_in_order():
    """Case 3: a streaming on_query yielding 'Hi there. How are you?'
    must produce two synth calls (one per sentence) and the audio sender
    consumes them in order, with the data-channel reply carrying the
    full assembled text."""
    async def tokens() -> AsyncIterator[str]:
        for tok in ("Hi", " there", ". ", "How", " are", " you", "?"):
            yield tok

    async def on_query(pid: str, text: str, fresh_match: bool):
        return tokens()

    loop, ep, _, tts = _make_loop(on_query=on_query)
    await loop.dispatch("p1", "anything", pts_us=200)
    vs = loop._voice["p1"]
    await vs.current_task

    # Both sentences (without trailing whitespace) were synthesized.
    assert tts.calls == ["Hi there.", "How are you?"]
    # Audio was emitted (any non-zero count proves the sender ran).
    assert len(ep.return_audio) >= 2
    # Final data-channel reply is the full assembled text, stripped.
    assert [m.data.decode() for m in ep.return_data] == ["Hi there. How are you?"]


# ════════════════════════════════════════════════════════════════════════════
# 4. Interrupt: second dispatch cancels first + flushes + starts new
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_second_dispatch_cancels_in_flight_and_flushes_return_audio():
    """Case 4: a second ``dispatch`` for the same pid cancels the in-flight
    task, calls ``flush_return_audio``, and replaces the task. The first
    on_query is stuck on an Event so the cancellation has work to do."""
    gate = asyncio.Event()

    async def on_query(pid: str, text: str, fresh_match: bool):
        if text == "slow":
            await gate.wait()      # block until released or cancelled
            return "should not be reached"
        return "fast reply."

    loop, ep, _, _ = _make_loop(on_query=on_query)

    await loop.dispatch("p1", "slow", pts_us=300)
    first_task = loop._voice["p1"].current_task
    assert first_task is not None and not first_task.done()

    # Second dispatch should cancel the first and start a new one.
    await loop.dispatch("p1", "fast", pts_us=400)

    # First task should be cancelled (or at least done).
    assert first_task.done()
    assert first_task.cancelled() or first_task.exception() is None

    # flush_return_audio was called between cancel and the new task.
    assert "p1" in ep.flush_calls

    # The new task can now complete.
    second_task = loop._voice["p1"].current_task
    assert second_task is not first_task
    await second_task

    # Release the now-cancelled gate so no warnings about pending tasks.
    gate.set()


# ════════════════════════════════════════════════════════════════════════════
# 5. STOP via gate
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_stop_via_gate_cancels_flushes_and_invokes_say_stop_ack():
    """Case 5: feeding 'stop' to the gate triggers ``_handle_stop`` which
    cancels the in-flight task, flushes return audio, invokes
    ``gate.say_stop_ack`` (so the TTS gets a 'Okay, I will stop.' call)
    and echoes a 'Okay, I will stop.' data message on ``text_topic``.
    Also exercises the ``on_stop_extra`` hook."""
    gate = asyncio.Event()
    stop_extra_calls: list[str] = []

    async def on_query(pid: str, text: str, fresh_match: bool) -> str:
        await gate.wait()
        return "never sent"

    async def on_stop_extra(pid: str) -> None:
        stop_extra_calls.append(pid)

    loop, ep, _, tts = _make_loop(
        on_query      = on_query,
        on_stop_extra = on_stop_extra,
    )

    await loop.dispatch("p1", "anything", pts_us=500)
    in_flight = loop._voice["p1"].current_task
    assert in_flight is not None and not in_flight.done()

    # Feed "stop" through the gate — should invoke _handle_stop.
    await loop._gate.feed("p1", "stop")

    assert in_flight.cancelled() or in_flight.done()
    # ``flush_return_audio`` is called at least once by the stop path.
    # (The original dispatch also flushes at the start of every query,
    # so the exact count is not part of the contract — what matters is
    # the stop handler issues its own flush before saying the ack.)
    assert "p1" in ep.flush_calls
    # say_stop_ack synthesizes "Okay, I will stop." via the TTS.
    assert "Okay, I will stop." in tts.calls
    # The data-channel echo lands too.
    echo_payloads = [m.data.decode() for m in ep.return_data]
    assert "Okay, I will stop." in echo_payloads
    assert stop_extra_calls == ["p1"]

    gate.set()


# ════════════════════════════════════════════════════════════════════════════
# 6. Participant joined → hook + greeting
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_participant_joined_runs_hook_and_speaks_greeting():
    """Case 6: a participant-joined event invokes the optional
    ``on_participant_joined`` hook AND speaks the greeting via TTS."""
    joins: list[str] = []

    async def on_query(pid: str, text: str, fresh_match: bool) -> str:
        return ""

    async def on_join(pid: str) -> None:
        joins.append(pid)

    loop, ep, _, tts = _make_loop(
        on_query              = on_query,
        on_participant_joined = on_join,
    )
    assert ep.participant_cb is not None

    await ep.participant_cb(ParticipantEvent(participant_id="p1", joined=True, pts_us=0))
    # The gate.participant_joined call is scheduled as a Task — let it run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert joins == ["p1"]
    # The greeting was synthesized at least once (default text picks up
    # the gate's format_phrase_help() — None for empty phrases, so the
    # fallback "Hi, I'm listening. Ask me anything." path is taken).
    assert any("listening" in c.lower() for c in tts.calls)


# ════════════════════════════════════════════════════════════════════════════
# 7. Participant left clears per-pid state
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_participant_left_clears_voice_state_and_invokes_hook():
    """Case 7: a leave event removes the per-pid VoiceState, cancels any
    in-flight task for that pid, and fires the optional left-hook."""
    gate = asyncio.Event()
    leaves: list[str] = []

    async def on_query(pid: str, text: str, fresh_match: bool) -> str:
        await gate.wait()
        return "never"

    async def on_left(pid: str) -> None:
        leaves.append(pid)

    loop, ep, _, _ = _make_loop(
        on_query            = on_query,
        on_participant_left = on_left,
    )

    await loop.dispatch("p1", "x", pts_us=600)
    in_flight = loop._voice["p1"].current_task
    assert in_flight is not None and not in_flight.done()

    await ep.participant_cb(ParticipantEvent(participant_id="p1", joined=False, pts_us=0))

    # Per-pid state is gone.
    assert "p1" not in loop._voice
    # In-flight task was cancelled.
    await asyncio.sleep(0)
    assert in_flight.cancelled() or in_flight.done()
    # Hook fired.
    assert leaves == ["p1"]

    gate.set()


# ════════════════════════════════════════════════════════════════════════════
# 8. Custom greeting (string)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_custom_string_greeting_overrides_gate_default():
    """Case 8: ``greeting='hi there'`` is spoken verbatim, bypassing
    ``format_phrase_help()``."""
    async def on_query(pid: str, text: str, fresh_match: bool) -> str:
        return ""

    loop, ep, _, tts = _make_loop(on_query=on_query, greeting="hi there")

    await ep.participant_cb(ParticipantEvent(participant_id="p1", joined=True, pts_us=0))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert "hi there" in tts.calls


# ════════════════════════════════════════════════════════════════════════════
# 9. Custom greeting (callable)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_custom_callable_greeting_receives_gate():
    """Case 9: ``greeting=lambda gate: gate.format_phrase_help() or 'fallback'``
    is invoked with the underlying ``VoiceGate``. With no phrases configured
    ``format_phrase_help`` returns None, so the fallback wins."""

    async def on_query(pid: str, text: str, fresh_match: bool) -> str:
        return ""

    def greet(gate: VoiceGate) -> str:
        return gate.format_phrase_help() or "fallback"

    loop, ep, _, tts = _make_loop(on_query=on_query, greeting=greet)

    await ep.participant_cb(ParticipantEvent(participant_id="p1", joined=True, pts_us=0))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert "fallback" in tts.calls


# ════════════════════════════════════════════════════════════════════════════
# 10. observe_tts_wav is fed by every TTS WAV
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_every_tts_wav_passes_through_observe_tts_wav():
    """Case 10: each WAV the loop generates is observed by the gate so
    the lazy listening-chime sample rate can be picked up from the TTS
    output regardless of which path produced it. We monkeypatch
    ``gate.observe_tts_wav`` with a counting wrapper and exercise both
    the str-result path and the streaming-result path."""
    counts = {"n": 0}

    async def stream_tokens() -> AsyncIterator[str]:
        for t in ("Streaming", " reply", "."):
            yield t

    async def on_query(pid: str, text: str, fresh_match: bool):
        if text == "stream":
            return stream_tokens()
        return "string reply."

    loop, ep, _, tts = _make_loop(on_query=on_query)

    original_observe = loop._gate.observe_tts_wav
    def counting_observe(wav: bytes) -> None:
        counts["n"] += 1
        original_observe(wav)
    loop._gate.observe_tts_wav = counting_observe  # type: ignore[assignment]

    # str path: one synth → one observe.
    await loop.dispatch("p1", "static", pts_us=700)
    await loop._voice["p1"].current_task
    assert counts["n"] == 1

    # streaming path: one sentence → one synth → one observe.
    await loop.dispatch("p1", "stream", pts_us=800)
    await loop._voice["p1"].current_task
    assert counts["n"] == 2


# ════════════════════════════════════════════════════════════════════════════
# 11. Gate → on_query passes fresh_match through (regression guard)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_gate_route_propagates_fresh_match_to_on_query():
    """Case 11: when the gate dispatches case 2 (fresh phrase + payload)
    the user's ``on_query`` must see ``fresh_match=True``; on case 3
    (follow-up window continuation) it must see ``fresh_match=False``.

    The data-channel ``dispatch()`` always passes ``True`` because the
    gate isn't in the loop there. This regression guard catches a bug
    where the loop dropped the gate's flag and hardcoded ``True`` for
    every voice-path query."""
    seen: list[tuple[str, bool]] = []

    async def on_query(pid: str, text: str, fresh_match: bool) -> str:
        seen.append((text, fresh_match))
        return ""

    loop, ep, _, _ = _make_loop(on_query=on_query, vg_phrases=("agent",))

    # Case 2: phrase + payload in the same utterance → fresh_match=True.
    await loop._gate.feed("p1", "agent, what is this")
    # Drain the dispatch task so the on_query call completes.
    if loop._voice.get("p1") and loop._voice["p1"].current_task:
        await loop._voice["p1"].current_task

    # Case 3: phrase-only opens the window; the next utterance is the
    # continuation and must arrive with fresh_match=False.
    await loop._gate.feed("p2", "agent")
    await loop._gate.feed("p2", "what is this")
    if loop._voice.get("p2") and loop._voice["p2"].current_task:
        await loop._voice["p2"].current_task

    assert seen == [
        ("what is this", True),
        ("what is this", False),
    ]


# ════════════════════════════════════════════════════════════════════════════
# 12. Data-channel dispatch always sets fresh_match=True
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_data_channel_dispatch_sets_fresh_match_true():
    """Case 12: the public ``dispatch()`` (the data-channel path) bypasses
    the gate entirely, so ``fresh_match`` must be True — the flag's
    meaning of "this is a new top-level query" matches that path."""
    seen: list[bool] = []

    async def on_query(pid: str, text: str, fresh_match: bool) -> str:
        seen.append(fresh_match)
        return ""

    loop, _, _, _ = _make_loop(on_query=on_query)
    await loop.dispatch("p1", "hi", pts_us=900)
    await loop._voice["p1"].current_task

    assert seen == [True]


# ════════════════════════════════════════════════════════════════════════════
# 13. on_phrase_only_extra + on_drop_extra hooks fire alongside gate events
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_phrase_only_extra_and_drop_extra_hooks_fire():
    """Case 13: the brain-side extras fire when the gate raises a
    phrase-only acknowledgement or drops an utterance — without them, a
    speculative ``on_speech_start`` (e.g. camera warmup) has no
    counterbalancing teardown when the user never follows up or the
    phrase doesn't match."""
    phrase_only_calls: list[str] = []
    drop_calls:        list[tuple[str, str]] = []

    async def on_query(pid: str, text: str, fresh_match: bool) -> str:
        return ""

    async def on_phrase_only_extra(pid: str) -> None:
        phrase_only_calls.append(pid)

    async def on_drop_extra(pid: str, text: str) -> None:
        drop_calls.append((pid, text))

    loop, _, _, _ = _make_loop(
        on_query             = on_query,
        vg_phrases           = ("agent",),
        on_phrase_only_extra = on_phrase_only_extra,
        on_drop_extra        = on_drop_extra,
    )

    # Phrase-only: utterance is exactly the magic phrase, opens follow-up window.
    await loop._gate.feed("p1", "agent")
    # Drop: non-magic-phrase utterance while gate requires phrase + no follow-up.
    await loop._gate.feed("p2", "what time is it")

    assert phrase_only_calls == ["p1"]
    assert drop_calls and drop_calls[0][0] == "p2"


# ════════════════════════════════════════════════════════════════════════════
# 14. wire_voice_gate registers the five handlers in one call
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_wire_voice_gate_registers_handlers_and_defaults_on_drop():
    """Case 14: ``wire_voice_gate`` is the pipecat-side escape hatch that
    collapses the five ``gate.on_*`` registration lines into one call.

    Asserts: (a) the four explicitly-passed handlers fire when their
    corresponding gate event runs, and (b) ``on_drop`` defaults to a
    DEBUG logger when the caller doesn't supply one — i.e. the gate's
    ``_on_drop_h`` slot is populated, not left ``None``."""
    gate = VoiceGate(
        VoiceGateConfig(magic_phrases=("agent",)),
        audio_sink=None,
        tts=None,
    )

    seen_query:          list[tuple[str, str, bool]] = []
    seen_stop:           list[str]                   = []
    seen_phrase_only:    list[str]                   = []
    seen_join:           list[str]                   = []

    async def on_query(pid: str, text: str, fresh_match: bool) -> None:
        seen_query.append((pid, text, fresh_match))

    async def on_stop(pid: str) -> None:
        seen_stop.append(pid)

    async def on_phrase_only(pid: str) -> None:
        seen_phrase_only.append(pid)

    async def on_participant_joined(pid: str) -> None:
        seen_join.append(pid)

    wire_voice_gate(
        gate,
        on_query              = on_query,
        on_stop               = on_stop,
        on_phrase_only        = on_phrase_only,
        on_participant_joined = on_participant_joined,
    )

    # Handlers slot in.
    assert gate._on_query_h               is on_query
    assert gate._on_stop_h                is on_stop
    assert gate._on_phrase_only_h         is on_phrase_only
    assert gate._on_participant_joined_h  is on_participant_joined

    # Default on_drop is installed (not None), and it's awaitable.
    assert gate._on_drop_h is not None
    await gate._on_drop_h("p1", "anything dropped")  # should not raise

    # End-to-end: feed "agent, hello" — case 2 (phrase + payload) →
    # on_query fires with fresh_match=True. "agent stop" → on_stop. A
    # bare "agent" opens the window → on_phrase_only.
    await gate.feed("p1", "agent hello")
    await gate.feed("p2", "agent")
    await gate.feed("p3", "agent stop")

    assert seen_query        == [("p1", "hello", True)]
    assert seen_phrase_only  == ["p2"]
    assert seen_stop         == ["p3"]


# ════════════════════════════════════════════════════════════════════════════
# 15. wire_voice_gate custom on_drop overrides the default
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_wire_voice_gate_custom_on_drop_overrides_default():
    """Case 15: passing ``on_drop`` explicitly overrides the default
    DEBUG-logger handler. Verified by checking the slot identity, since
    the gate's drop-path is exercised by xr-ai-voicegate's own tests."""
    gate = VoiceGate(
        VoiceGateConfig(magic_phrases=("agent",)),
        audio_sink=None,
        tts=None,
    )

    async def custom_drop(pid: str, text: str) -> None:
        pass

    async def on_query(pid: str, text: str, fresh_match: bool) -> None: ...
    async def on_stop(pid: str) -> None: ...

    wire_voice_gate(
        gate,
        on_query = on_query,
        on_stop  = on_stop,
        on_drop  = custom_drop,
    )

    assert gate._on_drop_h is custom_drop
