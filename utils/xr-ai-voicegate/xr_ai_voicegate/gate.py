# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech-only opt-in gate shared by agent workers.

Owns the magic-phrase + follow-up + STOP ladder, the lazy listening
chime, and the participant-joined greeting hook. Workers feed STT
transcripts via ``feed`` and register handlers for the events the gate
emits (query, stop, phrase-only, drop, participant-joined).

The gate does NOT serialize calls per-pid. Consumers are expected to
gate concurrency themselves (e.g. a per-pid ``transcribing`` flag while
STT is in flight) so two utterances from the same participant don't
race through ``feed`` at the same time.
"""
from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

from ._chime import build_chime_wav, read_wav_sample_rate
from ._phrases import STOP_RE, build_magic_pattern, strip_magic
from .config import AudioSink, TTSLike, VoiceGateConfig


logger = logging.getLogger("xr_ai_voicegate")


QueryHandler             = Callable[[str, str, bool], Awaitable[None]]   # (pid, query, fresh_match)
StopHandler              = Callable[[str], Awaitable[None]]
PhraseOnlyHandler        = Callable[[str], Awaitable[None]]
DropHandler              = Callable[[str, str], Awaitable[None]]
ParticipantJoinedHandler = Callable[[str], Awaitable[None]]


class VoiceGate:
    """Speech-input gating state machine.

    Event ladder in ``feed`` — exactly one event fires per call, in this
    deterministic priority order:

    1. STOP detected on raw text OR on the magic-phrase-stripped tail
       → ``on_stop(pid)``; closes the follow-up window.
    2. Magic phrase matched AND query non-empty
       → ``on_query(pid, query, fresh_match=True)``; closes the follow-up
       window.
    3. Follow-up window still open (and not STOP)
       → ``on_query(pid, raw_text, fresh_match=False)``; closes the
       follow-up window. ``fresh_match`` distinguishes this continuation
       from case 2 so consumers can suppress one-shot side effects
       (e.g. the listening chime) on the follow-up dispatch.
    4. Magic phrase matched AND query empty
       → ``on_phrase_only(pid)``; opens the follow-up window.
    5. Otherwise → ``on_drop(pid, raw_text)``; closes the window
       defensively.

    When ``magic_phrases`` is empty the gate is in always-on mode: STOP
    still wins (interrupts must work without a phrase) and every other
    utterance dispatches straight to ``on_query`` with
    ``fresh_match=True``. The follow-up / phrase-only / drop branches
    are inert in this mode — they only make sense once a phrase exists
    to gate against.

    Handler exceptions are logged and swallowed so one bad handler does
    not kill the gate.
    """

    def __init__(
        self,
        cfg: VoiceGateConfig,
        *,
        audio_sink: AudioSink,
        tts: TTSLike,
    ) -> None:
        self._cfg          = cfg
        self._audio_sink   = audio_sink
        self._tts          = tts
        self._magic_re     = build_magic_pattern(cfg.magic_phrases)
        # Without a phrase the chime would have no fresh-match event to
        # ride; mirror the original simple-vlm-example wiring rather than
        # firing on every utterance.
        self._chime_enabled: bool = bool(cfg.listening_chime) and self._magic_re is not None
        self._chime_wav: bytes | None = None
        self._followup_until: dict[str, float] = {}

        self._on_query_h:               QueryHandler | None             = None
        self._on_stop_h:                StopHandler | None              = None
        self._on_phrase_only_h:         PhraseOnlyHandler | None        = None
        self._on_drop_h:                DropHandler | None              = None
        self._on_participant_joined_h:  ParticipantJoinedHandler | None = None

    # ── handler registration ──────────────────────────────────────────────────

    def on_query(self, h: QueryHandler) -> None:
        self._on_query_h = h

    def on_stop(self, h: StopHandler) -> None:
        self._on_stop_h = h

    def on_phrase_only(self, h: PhraseOnlyHandler) -> None:
        self._on_phrase_only_h = h

    def on_drop(self, h: DropHandler) -> None:
        self._on_drop_h = h

    def on_participant_joined(self, h: ParticipantJoinedHandler) -> None:
        self._on_participant_joined_h = h

    def bind(
        self,
        *,
        on_query: QueryHandler,
        on_stop: StopHandler,
        on_phrase_only: PhraseOnlyHandler | None = None,
        on_drop: DropHandler | None = None,
        on_participant_joined: ParticipantJoinedHandler | None = None,
    ) -> None:
        """Register every handler in one call.

        Equivalent to calling the individual ``on_*`` setters; offered as a
        single bind point for external consumers that have all handlers in
        hand. ``on_query`` and ``on_stop`` are required because the gate is
        useless without them; the others are optional and default to no-op
        when unset.
        """
        self._on_query_h              = on_query
        self._on_stop_h               = on_stop
        self._on_phrase_only_h        = on_phrase_only
        self._on_drop_h               = on_drop
        self._on_participant_joined_h = on_participant_joined

    # ── per-participant lifecycle ─────────────────────────────────────────────

    async def participant_joined(self, pid: str) -> None:
        await self._invoke(self._on_participant_joined_h, "on_participant_joined", pid)

    def forget(self, pid: str) -> None:
        self._followup_until.pop(pid, None)

    # ── event ladder ──────────────────────────────────────────────────────────

    async def feed(self, pid: str, text: str) -> None:
        """Run one transcript through the event ladder; fires exactly one event.

        Not re-entrant per-pid: the follow-up window (``_followup_until``) is
        read-then-mutated without locking, so two concurrent ``feed`` calls
        for the same pid can race the window state. Consumers must serialize
        calls per participant (e.g. a per-pid ``transcribing`` flag).
        """
        # Always-on mode: no phrases configured, so the magic-phrase /
        # follow-up / phrase-only / drop branches don't apply. STOP still
        # wins (interrupts must work even without a phrase), and every
        # other utterance is a fresh query.
        if self._magic_re is None:
            if STOP_RE.match(text):
                logger.info(
                    "gate decision pid=%r kind=STOP fresh_match=False "
                    "followup_window_open=False",
                    pid,
                )
                logger.debug("stop bypass pid=%r %r", pid, text[:80])
                await self._invoke(self._on_stop_h, "on_stop", pid)
                return
            logger.info(
                "gate decision pid=%r kind=DISPATCH fresh_match=True "
                "followup_window_open=False",
                pid,
            )
            logger.debug("audio query pid=%r %r", pid, text[:80])
            await self._invoke(self._on_query_h, "on_query", pid, text, True)
            return

        now_mono       = time.monotonic()
        in_followup    = self._followup_until.get(pid, 0.0) > now_mono
        stripped       = strip_magic(self._magic_re, text)
        matched_magic  = stripped is not None
        stop_candidate = stripped if (matched_magic and stripped) else text

        # 1. STOP — always wins. Matched on both the raw transcript
        #    ("stop") AND on the magic-phrase-stripped tail ("hey agent,
        #    stop") so the fast path triggers either way.
        if STOP_RE.match(stop_candidate):
            logger.info(
                "gate decision pid=%r kind=STOP fresh_match=%s "
                "followup_window_open=%s",
                pid, matched_magic, in_followup,
            )
            logger.debug("stop bypass pid=%r %r", pid, stop_candidate[:80])
            self._followup_until.pop(pid, None)
            await self._invoke(self._on_stop_h, "on_stop", pid)
            return

        if matched_magic:
            query = stripped or ""
            if query:
                # 2. Fresh magic-phrase match with a payload — dispatch
                #    immediately and close the window so ambient speech
                #    after the answer must re-introduce a phrase.
                logger.info(
                    "gate decision pid=%r kind=DISPATCH fresh_match=True "
                    "followup_window_open=%s",
                    pid, in_followup,
                )
                self._followup_until.pop(pid, None)
                logger.debug("audio query pid=%r %r", pid, query[:80])
                await self._invoke(self._on_query_h, "on_query", pid, query, True)
                return
            # 4. Magic phrase with no follow-up payload — open the window
            #    so the user's next utterance counts as the actual query.
            logger.info(
                "gate decision pid=%r kind=PHRASE_ONLY fresh_match=True "
                "followup_window_open=%s",
                pid, in_followup,
            )
            self._followup_until[pid] = now_mono + self._cfg.followup_grace_s
            logger.info(
                "magic phrase only pid=%r — awaiting followup (%.1fs)",
                pid, self._cfg.followup_grace_s,
            )
            await self._invoke(self._on_phrase_only_h, "on_phrase_only", pid)
            return

        if in_followup:
            # 3. Followup continuation — dispatch raw text and close the
            #    window. Refreshing it on accept happens implicitly: the
            #    consumer can decide to re-open by calling back via a
            #    fresh magic-phrase match.
            logger.info(
                "gate decision pid=%r kind=DISPATCH fresh_match=False "
                "followup_window_open=True",
                pid,
            )
            self._followup_until.pop(pid, None)
            logger.debug("followup query pid=%r %r", pid, text[:80])
            await self._invoke(self._on_query_h, "on_query", pid, text, False)
            return

        # 5. Default drop — log and close the window defensively.
        logger.info(
            "gate decision pid=%r kind=DROP fresh_match=False "
            "followup_window_open=%s",
            pid, in_followup,
        )
        logger.debug("drop pid=%r %r", pid, text[:80])
        self._followup_until.pop(pid, None)
        await self._invoke(self._on_drop_h, "on_drop", pid, text)

    # ── worker-callable side effects ──────────────────────────────────────────

    async def play_chime(self, pid: str) -> None:
        """Emit the listening chime on the consumer's audio sink.

        No-op when the chime is disabled or has not been built yet (the
        chime is lazily synthesized to match the TTS sample rate the
        first time ``observe_tts_wav`` is called).
        """
        if not self._chime_enabled:
            return
        if self._chime_wav is None:
            logger.debug("play_chime no-op pid=%r — chime not yet built", pid)
            return
        try:
            await self._audio_sink.play_wav(pid, self._chime_wav)
        except Exception:
            logger.exception("chime send error pid=%r", pid)

    def format_phrase_help(self) -> str | None:
        """Return a sentence fragment telling the user how to address the
        agent given the configured phrases. ``None`` when no phrases are
        configured (the caller picks a generic greeting). The wording
        carries over from the original ``_greet`` implementation."""
        phrases = list(self._cfg.magic_phrases)
        if not phrases:
            return None
        if len(phrases) == 1:
            phrase_list = f'"{phrases[0]}"'
        elif len(phrases) == 2:
            phrase_list = f'"{phrases[0]}" or "{phrases[1]}"'
        else:
            phrase_list = (
                ", ".join(f'"{p}"' for p in phrases[:-1])
                + f', or "{phrases[-1]}"'
            )
        return (
            f'To talk to me, start your question with {phrase_list}. '
            f'For example, "{phrases[0]}, what am I looking at?"'
        )

    async def say_stop_ack(self, pid: str) -> None:
        """Synthesize a short canned stop acknowledgement and play it on
        the consumer's audio sink. Also observes the WAV so the lazy
        chime build can pick up the sample rate from this path."""
        try:
            wav = await self._tts.synthesize("Okay, I will stop.")
        except Exception:
            logger.exception("stop-ack tts error pid=%r", pid)
            return
        self.observe_tts_wav(wav)
        try:
            await self._audio_sink.play_wav(pid, wav)
        except Exception:
            logger.exception("stop-ack audio error pid=%r", pid)

    def observe_tts_wav(self, wav_bytes: bytes) -> None:
        """Build the chime at the TTS sample rate the first time a real
        TTS WAV passes through. No-op once built or when chime is
        disabled. A malformed WAV header disables the chime so the
        consumer doesn't keep paying for failed builds."""
        if self._chime_wav is not None or not self._chime_enabled:
            return
        try:
            sr = read_wav_sample_rate(wav_bytes)
            self._chime_wav = build_chime_wav(sr)
            logger.info("listening chime ready (sr=%d Hz)", sr)
        except ValueError:
            logger.exception("listening chime disabled (sample rate out of range)")
            self._chime_enabled = False
        except Exception:
            logger.exception("listening chime disabled (bad TTS wav header)")
            self._chime_enabled = False

    # ── handler dispatch ──────────────────────────────────────────────────────

    async def _invoke(self, handler, name: str, *args) -> None:
        if handler is None:
            return
        try:
            await handler(*args)
        except Exception:
            logger.exception("voicegate handler %s raised", name)
