# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enrollment and single-speaker utterance selection, independent of inference."""

from __future__ import annotations

from xr_ai_voicegate._conversation import _ControlMatcher
from xr_ai_voicegate._phrases import STOP_RE, build_magic_pattern, strip_magic
from xr_ai_voicegate._speaker import _SpeakerConfig


class _Selection:
    def __init__(self, cfg: _SpeakerConfig) -> None:
        self.cfg = cfg
        self.owner: int | None = None
        self.candidate: int | None = None
        self.duration = 0.0
        self.silence = 0.0
        self.pts_us = 0
        self.text = ""
        self._controls = _ControlMatcher(cfg)
        self._enrollment_controls: dict[int, _ControlMatcher] = {}
        self._wake_pattern = build_magic_pattern(cfg.magic_phrases)
        self._wake_until: dict[int, int] = {}

    def _enroll_completed(
        self, completed: list[tuple[int, str, int, bool]], *, completed_at_us: int | None = None,
    ) -> list[dict]:
        """Resolve attributed controls and turns at one accepted audio boundary."""
        if self.owner is not None:
            return []
        matches, turns, wakes = [], [], []
        for speaker, text, pts_us, truncated in completed:
            # Every completed utterance consumes the prior one-turn permission,
            # including a withheld control. A fresh bare wake can renew it below.
            until = self._wake_until.pop(speaker, None)
            matcher = self._enrollment_controls.setdefault(speaker, _ControlMatcher(self.cfg))
            if truncated:
                matcher._reset()
                self._wake_until.pop(speaker, None)
                continue
            action = matcher._feed(text, key=speaker, at_s=pts_us / 1_000_000)
            if self.cfg.mode in {"conversation", "hybrid"} and action == "start":
                matches.append(speaker)
                continue
            if self.cfg.mode == "conversation":
                continue
            if self.cfg.mode == "hybrid" and action == "stop":
                self._wake_until.pop(speaker, None)
                continue
            if not text.strip():
                self._wake_until.pop(speaker, None)
                continue
            tail = strip_magic(self._wake_pattern, text)
            if self.cfg.mode == "hybrid" and action == "pending" and tail != "":
                continue
            followup = until is not None and pts_us < until
            global_stop = self.cfg.mode == "wake_only" and self.cfg.stop_commands_enabled and STOP_RE.match(text)
            if tail == "" and self.cfg.mode != "always_on":
                wakes.append((speaker, pts_us))
            elif self.cfg.mode == "always_on" or tail is not None or followup or global_stop:
                turns.append({
                    "kind": "one_shot",
                    "text": (text.strip() if self.cfg.mode == "always_on" or tail is None else tail),
                    "pts_us": pts_us, "speaker_id": speaker,
                    "fresh_match": self.cfg.mode == "always_on" or tail is not None,
                })
        if matches:
            self._wake_until.clear()
            if len(matches) != 1:
                self._enrollment_controls.clear()
                return []
            self.owner = matches[0]
            self._enrollment_controls.clear()
            self._clear_utterance()
            return [{"kind": "enrolled"}]
        if self.cfg.mode == "always_on":
            return turns
        if len(turns) + len(wakes) != 1:
            # Never choose a same-boundary speaker by label or decoder order.
            if turns or wakes:
                self._wake_until.clear()
            return []
        if turns:
            return turns
        if wakes:
            if completed_at_us is None:
                raise ValueError("candidate completion requires its accepted audio boundary")
            speaker, pts_us = wakes[0]
            if self.cfg.followup_grace_s > 0:
                self._wake_until[speaker] = completed_at_us + round(self.cfg.followup_grace_s * 1_000_000)
            return [{"kind": "wake_ready", "speaker_id": speaker, "pts_us": pts_us}]
        return []

    def _activity(self, active: set[int], seconds: float, pts_us: int) -> tuple[int | None, list[dict], bool]:
        events = []
        started = False
        if self.owner is None:
            return None, [], False
        if self.candidate is None:
            if self.owner not in active:
                return None, [], False
            self.candidate = self.owner
            self.pts_us = pts_us
            started = True
            if self.owner is not None:
                events.append({"kind": "speech_start"})

        self.duration += seconds
        speaking = self.candidate in active
        self.silence = 0.0 if speaking else self.silence + seconds
        return self.candidate, events, started

    def _finish(self, text: str, *, allow_control: bool = True) -> list[dict]:
        events = []
        if self.owner is not None:
            events.append({"kind": "speech_stop"})
        if self.candidate is not None and text.strip():
            if allow_control:
                action = self._controls._feed(text, key=self.candidate, at_s=self.pts_us / 1_000_000)
            else:
                self._controls._reset()
                action = None
            if action == "stop":
                self.owner = None
                events.append({"kind": "released"})
            elif action == "pending":
                events.append({"kind": "control_pending", "text": text.strip(),
                               "pts_us": self.pts_us, "speaker_id": self.candidate})
            elif action != "start":
                events.append({
                    "kind": "transcript", "text": text.strip(), "released": self._controls._released,
                    "pts_us": self.pts_us, "speaker_id": self.candidate,
                })
        else:
            self._controls._reset()
        self._clear_utterance()
        return events

    def _clear_utterance(self) -> None:
        self.candidate = None
        self.duration = self.silence = 0.0
        self.text = ""

    @property
    def _finished(self) -> bool:
        return self.candidate is not None and (
            self.silence >= self.cfg.silence_duration or self.duration >= self.cfg.max_utterance_s
        )
