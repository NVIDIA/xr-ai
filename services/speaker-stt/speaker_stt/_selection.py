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
        self.ambiguous = False
        self.duration = 0.0
        self.silence = 0.0
        self.pts_us = 0
        self.text = ""
        self._controls = _ControlMatcher(cfg)

    def _activity(self, active: set[int], seconds: float, pts_us: int) -> tuple[int | None, list[dict], bool]:
        events = []
        started = False
        if self.owner is None and self._controls._words and active - {self._controls._key}:
            self._controls._reset()
        if self.candidate is None:
            if self.owner is not None:
                if self.owner not in active:
                    return None, [], False
                self.candidate = self.owner
            elif len(active) == 1:
                self.candidate = next(iter(active))
            else:
                # Ambiguous onset blocks enrollment until an entire silence gap.
                self.ambiguous = self.ambiguous or bool(active)
                if not active:
                    self.silence += seconds
                    if self.silence >= self.cfg.silence_duration:
                        self._clear_utterance()
                return None, [], False
            self.pts_us = pts_us
            started = True
            if self.owner is not None:
                events.append({"kind": "speech_start"})

        self.duration += seconds
        speaking = self.candidate in active
        self.silence = 0.0 if speaking else self.silence + seconds
        if self.owner is None and (active - {self.candidate}):
            self.ambiguous = True
        selected = None if self.ambiguous else self.candidate
        return selected, events, started

    def _finish(self, text: str) -> list[dict]:
        events = []
        if self.owner is not None:
            events.append({"kind": "speech_stop"})
        if self.candidate is not None and not self.ambiguous and text.strip():
            action = self._controls._feed(text, key=self.candidate, at_s=self.pts_us / 1_000_000)
            if self.owner is None:
                if action == "start":
                    self.owner = self.candidate
                    events.append({"kind": "enrolled"})
            elif action == "stop":
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
        self.ambiguous = False
        self.duration = self.silence = 0.0
        self.text = ""

    @property
    def _finished(self) -> bool:
        return self.candidate is not None and (
            self.silence >= self.cfg.silence_duration or self.duration >= self.cfg.max_utterance_s
        )
