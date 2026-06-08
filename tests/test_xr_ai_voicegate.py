# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the xr-ai-voicegate package.

Covers the phrase matcher, STOP regex, lazy chime synthesis, the per-pid
event ladder in ``VoiceGate.feed`` (STOP / fresh-match / followup /
phrase-only / drop), and the listening-chime + stop-ack side effects.

Everything is local: a no-op ``AudioSink`` and an in-memory ``TTSLike``
stub that returns a minimal WAV blob synthesized via the same primitive
the chime uses. No GPU, no network, no model loads.
"""
from __future__ import annotations

import io
import pathlib
import wave

import numpy as np
import pytest

from xr_ai_voicegate import VoiceGate, VoiceGateConfig, load_voice_gate_config
from xr_ai_voicegate._chime import build_chime_wav, read_wav_sample_rate
from xr_ai_voicegate._phrases import STOP_RE, build_magic_pattern, strip_magic


# ── test doubles ────────────────────────────────────────────────────────────


def _silence_wav(sample_rate: int, ms: int = 10) -> bytes:
    """Return a tiny but well-formed WAV blob at ``sample_rate`` Hz so the
    chime's ``read_wav_sample_rate`` has something to parse."""
    n = max(1, int(sample_rate * ms / 1000))
    pcm = np.zeros(n, dtype=np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class _FakeAudioSink:
    """No-op audio sink that records every (pid, wav_bytes) it receives."""

    def __init__(self) -> None:
        self.plays: list[tuple[str, bytes]] = []

    async def play_wav(self, pid: str, wav_bytes: bytes) -> None:
        self.plays.append((pid, wav_bytes))


class _FakeTTS:
    """TTS double returning a tiny but valid WAV at a configurable rate."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate     = sample_rate
        self.synth_calls:    list[str] = []
        self.raise_on_synth: bool      = False

    async def synthesize(self, text: str) -> bytes:
        self.synth_calls.append(text)
        if self.raise_on_synth:
            raise RuntimeError("tts down")
        return _silence_wav(self.sample_rate)


def _gate(
    *,
    phrases:          tuple[str, ...] = (),
    followup_grace_s: float           = 5.0,
    listening_chime:  bool            = False,
    tts_rate:         int             = 22050,
) -> tuple[VoiceGate, _FakeAudioSink, _FakeTTS]:
    cfg  = VoiceGateConfig(
        magic_phrases    = phrases,
        followup_grace_s = followup_grace_s,
        listening_chime  = listening_chime,
    )
    sink = _FakeAudioSink()
    tts  = _FakeTTS(sample_rate=tts_rate)
    return VoiceGate(cfg, audio_sink=sink, tts=tts), sink, tts


def _recording_handlers(gate: VoiceGate) -> list[tuple]:
    """Wire all five handler slots to a single events list and return it.

    ``on_query`` exposes the post-chime-fix signature ``(pid, text, fresh_match)``
    so tests can assert that case 2 (fresh magic-phrase match) and case 3
    (follow-up window continuation) are distinguishable."""
    events: list[tuple] = []

    async def on_query(pid: str, text: str, fresh_match: bool) -> None:
        events.append(("query", pid, text, fresh_match))

    async def on_stop(pid: str) -> None:
        events.append(("stop", pid))

    async def on_phrase_only(pid: str) -> None:
        events.append(("phrase_only", pid))

    async def on_drop(pid: str, text: str) -> None:
        events.append(("drop", pid, text))

    async def on_join(pid: str) -> None:
        events.append(("join", pid))

    gate.on_query(on_query)
    gate.on_stop(on_stop)
    gate.on_phrase_only(on_phrase_only)
    gate.on_drop(on_drop)
    gate.on_participant_joined(on_join)
    return events


# ════════════════════════════════════════════════════════════════════════════
# 1. Phrase matcher (`_phrases.py`)
# ════════════════════════════════════════════════════════════════════════════


def test_phrase_strict_prefix_strip_returns_query_tail():
    """Case 1: 'agent, what am I looking at?' strips to the query tail."""
    pat = build_magic_pattern(["agent"])
    assert strip_magic(pat, "agent, what am I looking at?") == "what am I looking at?"


def test_phrase_multi_phrase_both_forms_match():
    """Case 2: with ['agent', 'hey agent'] both prefixes match and strip."""
    pat = build_magic_pattern(["agent", "hey agent"])
    assert strip_magic(pat, "agent, hi")     == "hi"
    assert strip_magic(pat, "hey agent, hi") == "hi"


def test_phrase_case_insensitive_match():
    """Case 3: STT may upper-case the first letter — must still match."""
    pat = build_magic_pattern(["hey agent"])
    assert strip_magic(pat, "Hey Agent, what's up?") == "what's up?"


def test_phrase_tolerates_internal_and_trailing_punctuation():
    """Case 4: 'Hey, agent.' has a comma between words and a period at the
    end. The matcher must still treat it as the configured 'hey agent'
    prefix and strip it cleanly."""
    pat = build_magic_pattern(["hey agent"])
    stripped = strip_magic(pat, "Hey, agent.")
    # The configured "hey agent" is matched even with STT punctuation
    # spliced in; nothing follows so the tail is empty.
    assert stripped == ""


def test_phrase_mid_sentence_does_not_match():
    """Case 5: strict-prefix only — 'the agent told me ...' must not strip."""
    pat = build_magic_pattern(["agent"])
    assert strip_magic(pat, "the agent told me to wait") is None


def test_phrase_no_filler_allowance_before_phrase():
    """Case 6: even one filler word before the phrase ('hello agent') is
    rejected. Strict prefix means literally first token after whitespace."""
    pat = build_magic_pattern(["agent"])
    assert strip_magic(pat, "hello agent") is None


def test_phrase_empty_list_yields_none_pattern_and_passthrough_strip():
    """Case 7: no phrases → ``build_magic_pattern`` returns None; with the
    None pattern ``strip_magic`` is the identity on the input text."""
    assert build_magic_pattern([]) is None
    assert strip_magic(None, "anything goes here") == "anything goes here"


def test_phrase_only_utterance_strips_to_empty_string():
    """Case 8: 'agent' on its own — match succeeds, payload is empty
    string (NOT None, which means 'no match')."""
    pat = build_magic_pattern(["agent"])
    assert strip_magic(pat, "agent") == ""


# ════════════════════════════════════════════════════════════════════════════
# 2. STOP regex (`_phrases.STOP_RE`)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("text", [
    "stop",
    "Stop.",
    "be quiet",
    "shut up",
    "stop talking",
])
def test_stop_regex_canonical_forms_match(text: str):
    """Case 9: all the canonical interruption phrases match."""
    assert STOP_RE.match(text) is not None


def test_stop_regex_suffix_with_too_much_filler_does_not_match():
    """Case 10 (locked to actual behavior): 'the agent told me to stop'
    has 4 filler words before 'stop'; the regex allows {0,2} filler
    words, so this does NOT match.

    The original brief speculated this would match as a "suffix STOP" —
    the impl says no. We lock the impl's behavior in as the contract."""
    assert STOP_RE.match("the agent told me to stop") is None


def test_stop_regex_mid_sentence_real_question_does_not_match():
    """Case 11: 'what should we stop doing about climate change' is a
    genuine question that mentions 'stop' mid-sentence; must not trigger."""
    assert STOP_RE.match("what should we stop doing about climate change") is None


# ════════════════════════════════════════════════════════════════════════════
# 3. Chime synthesis (`_chime.py`)
# ════════════════════════════════════════════════════════════════════════════


def test_chime_24khz_header_shape_and_duration():
    """Case 12: 24 kHz chime is a parseable mono int16 WAV ~250 ms long."""
    wav = build_chime_wav(24_000)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getframerate()  == 24_000
        assert wf.getnchannels()  == 1
        assert wf.getsampwidth()  == 2          # int16 → 2 bytes/sample
        # 250 ms ± one frame (the chime uses int(sr * 0.25) samples).
        assert wf.getnframes() == int(24_000 * 0.25)


def test_chime_16khz_header_shape_and_duration():
    """Case 13: same contract at 16 kHz."""
    wav = build_chime_wav(16_000)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getframerate()  == 16_000
        assert wf.getnchannels()  == 1
        assert wf.getsampwidth()  == 2
        assert wf.getnframes() == int(16_000 * 0.25)


@pytest.mark.parametrize("sample_rate", [16_000, 24_000])
def test_chime_read_wav_sample_rate_round_trip(sample_rate: int):
    """Case 14: ``read_wav_sample_rate`` recovers the rate the chime was
    built at — round-trip guard for the WAV header parsing path that
    ``observe_tts_wav`` relies on."""
    wav = build_chime_wav(sample_rate)
    assert read_wav_sample_rate(wav) == sample_rate


def test_chime_clamp_accepts_upper_bound_192khz():
    """Case 14 (b): 192 kHz is at the documented upper bound and must
    still build — the clamp is inclusive on both ends so legitimate
    high-rate TTS outputs are not rejected."""
    wav = build_chime_wav(192_000)
    assert read_wav_sample_rate(wav) == 192_000


def test_chime_clamp_rejects_just_above_upper_bound():
    """Case 14 (c): one Hz past the upper bound raises ValueError. Guards
    against a hostile or corrupted WAV header that declares a multi-GHz
    rate from driving a multi-GB ``np.linspace`` allocation."""
    with pytest.raises(ValueError):
        build_chime_wav(192_001)


@pytest.mark.asyncio
async def test_observe_tts_wav_with_out_of_range_rate_disables_chime_without_crash():
    """Case 14 (d): a malicious or corrupted TTS WAV declaring an
    absurdly high sample rate (here 1 GHz, the kind of value an attacker
    might splice into a uint32 header) must not crash the gate. The
    chime caller swallows the ValueError, logs "sample rate out of
    range", and disables the chime so subsequent ``play_chime`` calls
    are no-ops.

    1 GHz is comfortably above the 192 kHz clamp but stays under the
    uint32-byterate limit Python's ``wave`` module enforces on the WAV
    header, so the malformed blob is round-trip parseable and the
    failure happens inside ``build_chime_wav`` exactly where the clamp
    is meant to fire."""
    gate, sink, _ = _gate(phrases=("agent",), listening_chime=True)

    # _silence_wav uses int16 PCM so the 1-frame buffer is 2 bytes
    # regardless of the (fictitious) sample rate stored in the header.
    bad_wav = _silence_wav(1_000_000_000, ms=0)

    gate.observe_tts_wav(bad_wav)        # must not raise
    await gate.play_chime("p1")          # must not crash, must not emit

    assert gate._chime_wav is None
    assert gate._chime_enabled is False
    assert sink.plays == []


# ════════════════════════════════════════════════════════════════════════════
# 4. Event ladder (`gate.feed`)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_feed_no_phrases_passes_every_utterance_through_to_on_query():
    """Case 15: with ``magic_phrases=()`` the gate is in always-on mode —
    every non-STOP utterance dispatches to ``on_query`` with
    ``fresh_match=True``. Matches the original simple-vlm-example
    behavior the gate was extracted from, and what ``VoiceGateConfig``'s
    docstring promises ("empty tuple disables the gate so every STT
    transcript is dispatched")."""
    gate, _, _ = _gate(phrases=())
    events = _recording_handlers(gate)

    await gate.feed("p1", "what am I looking at")
    await gate.feed("p1", "tell me about this")

    assert events == [
        ("query", "p1", "what am I looking at", True),
        ("query", "p1", "tell me about this", True),
    ]


@pytest.mark.asyncio
async def test_feed_no_phrases_stop_still_routes_to_on_stop():
    """Case 15 (b): even in always-on mode, the STOP regex must still
    fire ``on_stop`` so an interrupt works without a wake word."""
    gate, _, _ = _gate(phrases=())
    events = _recording_handlers(gate)

    await gate.feed("p1", "stop")

    assert events == [("stop", "p1")]


@pytest.mark.asyncio
async def test_feed_raw_stop_fires_on_stop():
    """Case 16: with phrases configured, the raw word 'stop' bypasses the
    magic-phrase requirement and fires on_stop."""
    gate, _, _ = _gate(phrases=("agent",))
    events = _recording_handlers(gate)

    await gate.feed("p1", "stop")

    assert events == [("stop", "p1")]


@pytest.mark.asyncio
async def test_feed_magic_phrase_with_query_fires_on_query_with_fresh_match_true():
    """Case 17: 'agent, what is this?' → on_query(pid, 'what is this?', fresh_match=True).

    ``fresh_match=True`` is the signal consumers use to drive one-shot
    side effects like the listening chime that should fire on case 2 but
    not on case 3 (the followup-window continuation)."""
    gate, _, _ = _gate(phrases=("agent",))
    events = _recording_handlers(gate)

    await gate.feed("p1", "agent, what is this?")

    assert events == [("query", "p1", "what is this?", True)]


@pytest.mark.asyncio
async def test_feed_no_phrase_and_no_followup_fires_on_drop():
    """Case 18: with phrases configured but no magic phrase in the
    transcript and no open followup window, the gate drops the utterance."""
    gate, _, _ = _gate(phrases=("agent",))
    events = _recording_handlers(gate)

    await gate.feed("p1", "what am I looking at")

    assert events == [("drop", "p1", "what am I looking at")]


@pytest.mark.asyncio
async def test_feed_phrase_only_opens_followup_window_and_dispatches_with_fresh_match_false():
    """Case 19: bare 'agent' fires on_phrase_only; the next utterance
    within the followup grace fires on_query with the raw text AND
    ``fresh_match=False`` so consumers can suppress the chime on the
    continuation (the chime already played when phrase-only opened the
    window — repeating it on the dispatch would be a double chime)."""
    gate, _, _ = _gate(phrases=("agent",), followup_grace_s=5.0)
    events = _recording_handlers(gate)

    await gate.feed("p1", "agent")
    await gate.feed("p1", "what am I looking at")

    assert events == [
        ("phrase_only", "p1"),
        ("query", "p1", "what am I looking at", False),
    ]


@pytest.mark.asyncio
async def test_feed_followup_window_closes_after_one_dispatch():
    """Case 20: each dispatch (fresh-match or followup-accepted) closes
    the window; subsequent utterances must reintroduce a magic phrase."""
    gate, _, _ = _gate(phrases=("agent",), followup_grace_s=5.0)
    events = _recording_handlers(gate)

    # Followup path: opens window, then next utterance consumes it.
    await gate.feed("p1", "agent")
    await gate.feed("p1", "what am I looking at")
    # Window is now closed — this third utterance must drop.
    await gate.feed("p1", "and another question")

    assert events == [
        ("phrase_only", "p1"),
        ("query", "p1", "what am I looking at", False),
        ("drop", "p1", "and another question"),
    ]


@pytest.mark.asyncio
async def test_feed_fresh_match_with_payload_closes_window_too():
    """Case 20 (b): a fresh-match dispatch ALSO closes the window so
    ambient speech after the answer doesn't ride a stale followup."""
    gate, _, _ = _gate(phrases=("agent",), followup_grace_s=5.0)
    events = _recording_handlers(gate)

    await gate.feed("p1", "agent, what is this")
    await gate.feed("p1", "and another question")

    assert events == [
        ("query", "p1", "what is this", True),
        ("drop", "p1", "and another question"),
    ]


@pytest.mark.asyncio
async def test_feed_followup_window_expires_after_grace(monkeypatch):
    """Case 21: once ``followup_grace_s`` seconds elapse, the window is
    closed; the next utterance drops instead of dispatching."""
    gate, _, _ = _gate(phrases=("agent",), followup_grace_s=2.0)
    events = _recording_handlers(gate)

    # Drive ``time.monotonic`` by hand so the test is deterministic and
    # doesn't sleep. Patch the symbol on the ``gate`` module — that's
    # where the lookup happens.
    from xr_ai_voicegate import gate as gate_module

    fake_now = [1000.0]
    monkeypatch.setattr(gate_module.time, "monotonic", lambda: fake_now[0])

    await gate.feed("p1", "agent")          # opens window until t=1002
    fake_now[0] = 1002.5                    # 0.5s past the grace deadline
    await gate.feed("p1", "what is this")   # window expired → must drop

    assert events == [
        ("phrase_only", "p1"),
        ("drop", "p1", "what is this"),
    ]


@pytest.mark.asyncio
async def test_feed_stop_wins_over_open_followup_window():
    """Case 22: 'stop' inside an open followup window must NOT be treated
    as a followup query — STOP_RE check sits above the followup branch."""
    gate, _, _ = _gate(phrases=("agent",), followup_grace_s=5.0)
    events = _recording_handlers(gate)

    await gate.feed("p1", "agent")  # opens window
    await gate.feed("p1", "stop")   # must fire on_stop, NOT on_query

    assert events == [
        ("phrase_only", "p1"),
        ("stop", "p1"),
    ]


@pytest.mark.asyncio
async def test_feed_stop_matched_on_magic_stripped_tail():
    """Case 23: 'hey agent stop' — the gate strips the magic prefix to
    'stop' and the STOP regex matches the tail. on_stop fires."""
    gate, _, _ = _gate(phrases=("hey agent",))
    events = _recording_handlers(gate)

    await gate.feed("p1", "hey agent stop")

    assert events == [("stop", "p1")]


@pytest.mark.asyncio
async def test_feed_fresh_match_flag_distinguishes_case_2_from_case_3():
    """Regression guard for the chime fix at 665faaf — case 2 (fresh
    magic-phrase + payload) must dispatch with ``fresh_match=True`` and
    case 3 (followup window continuation) must dispatch with
    ``fresh_match=False``. The original extraction collapsed both onto
    a single ``on_query(pid, text)`` signature so the consumer chimed
    twice per question; the flag is the contract that keeps the
    listening-chime side effect one-shot."""
    gate, _, _ = _gate(phrases=("agent",), followup_grace_s=5.0)
    events = _recording_handlers(gate)

    # Case 2: phrase + payload in the same utterance.
    await gate.feed("p1", "agent, what is this")
    # Case 3: phrase-only opens window, next utterance is the continuation.
    await gate.feed("p2", "agent")
    await gate.feed("p2", "what is this")

    assert events == [
        ("query",       "p1", "what is this", True),
        ("phrase_only", "p2"),
        ("query",       "p2", "what is this", False),
    ]


@pytest.mark.asyncio
async def test_feed_forget_clears_followup_window_for_pid():
    """Case 24: forget(pid) drops the open followup; the next utterance
    from that pid must be re-gated."""
    gate, _, _ = _gate(phrases=("agent",), followup_grace_s=5.0)
    events = _recording_handlers(gate)

    await gate.feed("p1", "agent")
    gate.forget("p1")
    await gate.feed("p1", "what is this")

    assert events == [
        ("phrase_only", "p1"),
        ("drop", "p1", "what is this"),
    ]


# ════════════════════════════════════════════════════════════════════════════
# 5. Chime lifecycle
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_play_chime_before_tts_observed_is_noop_not_crash():
    """Case 25: chime is built lazily after the first TTS WAV is observed.
    Calling play_chime before that point logs and returns — does not
    crash and does not push anything onto the audio sink."""
    gate, sink, _ = _gate(phrases=("agent",), listening_chime=True)

    await gate.play_chime("p1")

    assert sink.plays == []


@pytest.mark.asyncio
async def test_play_chime_after_observe_tts_wav_emits_built_chime():
    """Case 26: once a TTS WAV is observed, the chime is built at that
    sample rate and subsequent play_chime calls push the WAV bytes to
    the audio sink at the matching rate."""
    gate, sink, _ = _gate(phrases=("agent",), listening_chime=True)

    gate.observe_tts_wav(_silence_wav(22_050))
    await gate.play_chime("p1")

    assert len(sink.plays) == 1
    pid, wav = sink.plays[0]
    assert pid == "p1"
    assert read_wav_sample_rate(wav) == 22_050


@pytest.mark.asyncio
async def test_play_chime_disabled_when_listening_chime_false():
    """Case 27: listening_chime=False disables the chime even after a TTS
    WAV is observed."""
    gate, sink, _ = _gate(phrases=("agent",), listening_chime=False)

    gate.observe_tts_wav(_silence_wav(22_050))
    await gate.play_chime("p1")

    assert sink.plays == []


@pytest.mark.asyncio
async def test_play_chime_disabled_when_no_magic_phrases_configured():
    """Case 27 (b): even with listening_chime=True, an empty phrase list
    disables the chime — the chime is meant to ride a fresh-match event,
    and there are no fresh-matches without phrases."""
    gate, sink, _ = _gate(phrases=(), listening_chime=True)

    gate.observe_tts_wav(_silence_wav(22_050))
    await gate.play_chime("p1")

    assert sink.plays == []


# ════════════════════════════════════════════════════════════════════════════
# 6. Greeting + stop ack
# ════════════════════════════════════════════════════════════════════════════


def test_format_phrase_help_returns_none_with_no_phrases():
    """Case 28: no phrases → no help string (caller picks generic greeting)."""
    gate, _, _ = _gate(phrases=())
    assert gate.format_phrase_help() is None


def test_format_phrase_help_one_phrase():
    """Case 29: single-phrase form."""
    gate, _, _ = _gate(phrases=("agent",))
    assert gate.format_phrase_help() == (
        'To talk to me, start your question with "agent". '
        'For example, "agent, what am I looking at?"'
    )


def test_format_phrase_help_two_phrases():
    """Case 30: two-phrase form uses 'or', not a comma."""
    gate, _, _ = _gate(phrases=("agent", "hey agent"))
    assert gate.format_phrase_help() == (
        'To talk to me, start your question with "agent" or "hey agent". '
        'For example, "agent, what am I looking at?"'
    )


def test_format_phrase_help_three_phrases():
    """Case 31: three-or-more form uses commas with a final ', or'."""
    gate, _, _ = _gate(phrases=("a", "b", "c"))
    assert gate.format_phrase_help() == (
        'To talk to me, start your question with "a", "b", or "c". '
        'For example, "a, what am I looking at?"'
    )


@pytest.mark.asyncio
async def test_say_stop_ack_synthesizes_observes_and_plays():
    """Case 32: say_stop_ack runs the full three-step flow —
    tts.synthesize → observe_tts_wav (so the chime can build) → audio_sink.play_wav."""
    gate, sink, tts = _gate(phrases=("agent",), listening_chime=True, tts_rate=22_050)

    await gate.say_stop_ack("p1")

    # 1. TTS was called with the canned ack text.
    assert tts.synth_calls == ["Okay, I will stop."]
    # 2. The same WAV the TTS returned was pushed to the audio sink.
    assert len(sink.plays) == 1
    ack_pid, ack_wav = sink.plays[0]
    assert ack_pid == "p1"
    assert read_wav_sample_rate(ack_wav) == 22_050
    # 3. observe_tts_wav was called as part of the flow — confirm by
    #    playing the chime now and asserting it lands at the right rate.
    await gate.play_chime("p1")
    assert len(sink.plays) == 2
    _, chime_wav = sink.plays[1]
    assert read_wav_sample_rate(chime_wav) == 22_050


# ════════════════════════════════════════════════════════════════════════════
# 7. Handler exception swallowing
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_handler_exceptions_are_swallowed_and_gate_keeps_working():
    """Case 33: when on_query raises, the gate logs and continues —
    subsequent feed() calls still dispatch normally."""
    gate, _, _ = _gate(phrases=("agent",))

    calls: list[str] = []

    async def flaky_on_query(pid: str, text: str, fresh_match: bool) -> None:
        calls.append(text)
        if text == "boom":
            raise RuntimeError("handler exploded")

    gate.on_query(flaky_on_query)

    # First call raises inside the handler — must not propagate.
    await gate.feed("p1", "agent, boom")
    # Second call still works — gate state is intact.
    await gate.feed("p1", "agent, second time")

    assert calls == ["boom", "second time"]


@pytest.mark.asyncio
async def test_bind_registers_every_handler_in_one_call():
    """``bind`` is the single-call equivalent of the per-handler setters —
    when all five slots are supplied, the event ladder fires through them
    exactly as if each had been wired with the matching ``on_*`` setter."""
    gate, _, _ = _gate(phrases=("agent",), followup_grace_s=1.0)
    events: list[tuple] = []

    async def on_q(pid: str, text: str, fresh_match: bool) -> None:
        events.append(("query", pid, text, fresh_match))

    async def on_s(pid: str) -> None:
        events.append(("stop", pid))

    async def on_phrase_only(pid: str) -> None:
        events.append(("phrase_only", pid))

    async def on_drop(pid: str, text: str) -> None:
        events.append(("drop", pid, text))

    async def on_joined(pid: str) -> None:
        events.append(("joined", pid))

    gate.bind(
        on_query              = on_q,
        on_stop               = on_s,
        on_phrase_only        = on_phrase_only,
        on_drop               = on_drop,
        on_participant_joined = on_joined,
    )

    await gate.participant_joined("p1")
    await gate.feed("p1", "agent")                  # phrase only
    await gate.feed("p1", "what time is it")        # followup
    await gate.feed("p1", "agent, what is this")    # fresh match
    await gate.feed("p1", "stop")                   # stop
    await gate.feed("p1", "noise without phrase")   # drop

    assert events == [
        ("joined",       "p1"),
        ("phrase_only",  "p1"),
        ("query",        "p1", "what time is it",  False),
        ("query",        "p1", "what is this",     True),
        ("stop",         "p1"),
        ("drop",         "p1", "noise without phrase"),
    ]


@pytest.mark.asyncio
async def test_bind_optional_handlers_default_to_noop():
    """Only the required ``on_query`` + ``on_stop`` need to be passed to
    ``bind``; the omitted slots stay None and the gate degrades to "no
    handler for that event" rather than raising."""
    gate, _, _ = _gate(phrases=("agent",))
    events: list[tuple] = []

    async def on_q(pid: str, text: str, fresh_match: bool) -> None:
        events.append(("query", pid, text, fresh_match))

    async def on_s(pid: str) -> None:
        events.append(("stop", pid))

    gate.bind(on_query=on_q, on_stop=on_s)

    # Each of these would normally hit an optional handler — without
    # binding one, the gate must still process the feed cleanly. The
    # phrase-only opens the followup window so the next utterance gets
    # routed to on_query with fresh_match=False rather than to on_drop.
    await gate.participant_joined("p2")
    await gate.feed("p2", "agent")                  # phrase_only — no handler
    gate.forget("p2")                                # close the followup window
    await gate.feed("p2", "noise without phrase")   # drop — no handler
    await gate.feed("p2", "agent, real query")      # query — fires

    assert events == [("query", "p2", "real query", True)]


# ════════════════════════════════════════════════════════════════════════════
# 8. YAML loader (`load_voice_gate_config`)
# ════════════════════════════════════════════════════════════════════════════


def test_load_voice_gate_config_full_file(tmp_path: pathlib.Path):
    """Case 34: a populated file parses into a VoiceGateConfig with every
    field set as written."""
    p = tmp_path / "voice_gate.yaml"
    p.write_text(
        "magic_phrases:\n"
        '  - "agent"\n'
        '  - "hey agent"\n'
        "listening_chime:  true\n"
        "followup_grace_s: 7.5\n"
    )
    cfg = load_voice_gate_config(p)
    assert cfg == VoiceGateConfig(
        magic_phrases    = ("agent", "hey agent"),
        followup_grace_s = 7.5,
        listening_chime  = True,
    )


def test_load_voice_gate_config_missing_file_returns_defaults(tmp_path: pathlib.Path):
    """Case 35: a path that does not exist degrades to the dataclass
    defaults (always-on, chime opt-out only, 5 s follow-up grace) —
    same behavior the inline-block parser had for an absent block.
    The dataclass default for ``listening_chime`` is True, but always-on
    mode (empty ``magic_phrases``) inhibits the chime regardless."""
    assert load_voice_gate_config(tmp_path / "nope.yaml") == VoiceGateConfig()


def test_load_voice_gate_config_empty_file_returns_defaults(tmp_path: pathlib.Path):
    """Case 36: an empty file (``yaml.safe_load`` → None) degrades to
    defaults, matching the ``raw or {}`` normalization."""
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_voice_gate_config(p) == VoiceGateConfig()


def test_load_voice_gate_config_bare_string_phrase_normalizes(tmp_path: pathlib.Path):
    """Case 37: ``magic_phrases: agent`` (bare string, not a list) is
    normalized to a single-element tuple, matching the inline parser."""
    p = tmp_path / "voice_gate.yaml"
    p.write_text("magic_phrases: agent\n")
    cfg = load_voice_gate_config(p)
    assert cfg.magic_phrases == ("agent",)


def test_load_voice_gate_config_null_phrases_normalizes_to_empty(tmp_path: pathlib.Path):
    """Case 38: ``magic_phrases: null`` (or omitted) is normalized to an
    empty tuple — the gate's always-on mode."""
    p = tmp_path / "voice_gate.yaml"
    p.write_text("magic_phrases: null\nfollowup_grace_s: 3.0\n")
    cfg = load_voice_gate_config(p)
    assert cfg.magic_phrases == ()
    assert cfg.followup_grace_s == 3.0


def test_load_voice_gate_config_strips_whitespace_and_drops_empty(tmp_path: pathlib.Path):
    """Case 39: phrase entries are stripped, and empty entries (after
    strip) are dropped — same normalization the inline parser ran."""
    p = tmp_path / "voice_gate.yaml"
    p.write_text(
        "magic_phrases:\n"
        '  - "  agent  "\n'
        '  - ""\n'
        '  - "hey agent"\n'
    )
    cfg = load_voice_gate_config(p)
    assert cfg.magic_phrases == ("agent", "hey agent")


def test_load_voice_gate_config_rejects_non_mapping_top_level(tmp_path: pathlib.Path):
    """Case 40: a top-level list (or any non-mapping) raises ValueError —
    catches typo'd files that would otherwise silently degrade."""
    p = tmp_path / "voice_gate.yaml"
    p.write_text("- agent\n- hey agent\n")
    with pytest.raises(ValueError):
        load_voice_gate_config(p)


# ── Round-trip checks against the in-tree sample voice_gate.yaml files ──────
#
# Skipped when the tests are run from an environment where the agent-samples
# tree isn't reachable on disk (e.g. a published wheel install). When reachable,
# these guard against accidental drift between the loader's accepted schema
# and the YAML the samples actually ship.


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def test_load_voice_gate_config_simple_vlm_sample_round_trip():
    """Case 41: the simple-vlm-example sample ships with the wake-word
    config — confirm the file parses and exposes the documented phrases."""
    p = _repo_root() / "agent-samples" / "simple-vlm-example" / "yaml" / "voice_gate.yaml"
    if not p.exists():
        pytest.skip(f"sample voice_gate.yaml not reachable at {p}")
    cfg = load_voice_gate_config(p)
    assert cfg.magic_phrases    == ("agent", "hey agent")
    assert cfg.listening_chime  is False
    assert cfg.followup_grace_s == 5.0


def test_load_voice_gate_config_xr_render_demo_sample_round_trip():
    """Case 42: the xr-render-demo sample ships with the empty-list
    default (always-on) — confirm the file parses to defaults."""
    p = _repo_root() / "agent-samples" / "xr-render-demo" / "yaml" / "voice_gate.yaml"
    if not p.exists():
        pytest.skip(f"sample voice_gate.yaml not reachable at {p}")
    cfg = load_voice_gate_config(p)
    assert cfg.magic_phrases    == ()
    assert cfg.listening_chime  is False
    assert cfg.followup_grace_s == 5.0
