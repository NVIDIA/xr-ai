# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Estimate received speech SNR from the voice pipeline's existing VAD windows.

Only accepted utterances may establish a baseline or activate a warning. Keep
power summaries, never recordings. Silence and unavailable measurements cannot
be interpreted as zero SNR.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from xr_ai_voice import VoiceAudioUtterance, VoiceAudioWindow


@dataclass(frozen=True)
class _Estimate:
    start_us: int
    speech_s: float
    ratio: float | None
    received_at: float


class AudioQuality:
    def __init__(self, config, *, clock):
        self.config, self.clock = config, clock
        self.reset()

    def reset(self):
        self.baseline = None
        self.current = None
        self.bad_speech_s = 0.0
        self.noise = deque(maxlen=2048)
        self.estimates = deque(maxlen=64)
        self.last_us = None
        self.discard_utterance()

    def discard_utterance(self):
        self.start_us = None
        self.noise_power = None
        self.energy = self.speech_s = 0.0

    def observe(self, window: VoiceAudioWindow):
        if not math.isfinite(window.power) or window.power < 0 or not 0 < window.duration_s <= 0.1:
            return
        if self.last_us is not None:
            if window.timestamp_us < self.last_us:
                return
            if window.timestamp_us - self.last_us > 1_000_000:
                # A discontinuity invalidates the noise estimate, not the setup baseline.
                self.noise.clear()
                self.discard_utterance()
        self.last_us = window.timestamp_us
        cutoff = window.timestamp_us - int(self.config.audio_noise_window_s * 1e6)
        while self.noise and self.noise[0].timestamp_us < cutoff:
            self.noise.popleft()
        if window.speech:
            if self.start_us is None:
                self.start_us = window.timestamp_us
                # Leave a guard before speech onset to exclude VAD onset uncertainty.
                quiet = [w for w in self.noise if w.timestamp_us < self.start_us - 200_000]
                duration = sum(w.duration_s for w in quiet)
                if duration >= self.config.audio_min_noise_s:
                    self.noise_power = sum(w.power * w.duration_s for w in quiet) / duration
            self.energy += window.power * window.duration_s
            self.speech_s += window.duration_s
        else:
            self.noise.append(window)

    def finish(self, utterance: VoiceAudioUtterance):
        ratio = None
        if (self.speech_s >= self.config.audio_min_speech_s
                and self.noise_power is not None and self.noise_power > 0):
            ratio = self.energy / self.speech_s / self.noise_power - 1
            if ratio <= 0 or not math.isfinite(ratio):
                ratio = None
        self.estimates.append(_Estimate(utterance.timestamp_us, self.speech_s, ratio, self.clock()))
        self.discard_utterance()

    def accept(self, timestamp_us: int, *, setup: bool = False):
        estimate = next((e for e in reversed(self.estimates) if e.start_us == timestamp_us), None)
        if estimate is not None and self.clock() - estimate.received_at > self.config.audio_max_age_s:
            estimate = None
        if setup:
            self.baseline = estimate.ratio if estimate is not None else None
            self.bad_speech_s = 0.0
        elif estimate is not None and estimate is not self.current:
            if self.current is None or self.clock() - self.current.received_at > self.config.audio_max_age_s:
                self.bad_speech_s = 0.0
            if (estimate.ratio is not None and self.baseline is not None
                    and self._below_threshold(estimate.ratio)):
                self.bad_speech_s += estimate.speech_s
            else:
                self.bad_speech_s = 0.0
        self.current = estimate

    def _below_threshold(self, ratio):
        threshold = self.baseline * self.config.audio_snr_drop_ratio
        # Window energy summation must not move an exact threshold across the boundary.
        return ratio <= threshold or math.isclose(ratio, threshold, rel_tol=1e-12)

    def evaluate(self):
        estimate = self.current
        fresh = estimate is not None and self.clock() - estimate.received_at <= self.config.audio_max_age_s
        ratio = estimate.ratio if fresh else None
        if ratio is None:
            self.bad_speech_s = 0.0
        low = (ratio is not None and self.baseline is not None
               and self._below_threshold(ratio)
               and self.bad_speech_s >= self.config.audio_warning_speech_s)
        metrics = {
            "audio_snr_db": 10 * math.log10(ratio) if ratio is not None else None,
            "baseline_audio_snr_db": 10 * math.log10(self.baseline) if self.baseline is not None else None,
            "audio_snr_drop_db": 10 * math.log10(self.baseline / ratio) if low else None,
            "audio_snr_drop_percent": 100 * (1 - ratio / self.baseline) if low else None,
        }
        return low, metrics
