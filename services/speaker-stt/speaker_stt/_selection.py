# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enrollment and single-speaker utterance selection, independent of inference."""

from __future__ import annotations

from xr_ai_voicegate._conversation import _ControlMatcher
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

    def _enroll_completed(self, completed: list[tuple[int, str, int, bool]]) -> list[dict]:
        """Resolve all phrase candidates at one accepted audio-step boundary."""
        if self.owner is not None:
            return []
        matches = []
        for speaker, text, pts_us, truncated in completed:
            matcher = self._enrollment_controls.setdefault(speaker, _ControlMatcher(self.cfg))
            if truncated:
                matcher._reset()
                continue
            if matcher._feed(text, key=speaker, at_s=pts_us / 1_000_000) == "start":
                matches.append(speaker)
        if len(matches) != 1:
            if matches:
                self._enrollment_controls.clear()
            return []
        self.owner = matches[0]
        self._enrollment_controls.clear()
        self._clear_utterance()
        return [{"kind": "enrolled"}]

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
                    "kind": "transcript", "text": text.strip(),
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
