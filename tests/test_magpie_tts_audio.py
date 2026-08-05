# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pitch-preserving speaking-rate checks for Magpie TTS."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa")

_MAGPIE = Path(__file__).parents[1] / "ai-services" / "tts" / "magpie"
sys.path.insert(0, str(_MAGPIE))

from magpie_tts_server.audio import change_speed  # noqa: E402


def test_change_speed_shortens_audio_without_shifting_pitch() -> None:
    sample_rate = 22_050
    samples = np.arange(sample_rate) / sample_rate
    audio = np.sin(2 * np.pi * 220 * samples).astype(np.float32)

    faster = change_speed(audio, 1.1)
    peak_hz = np.fft.rfftfreq(len(faster), 1 / sample_rate)[np.argmax(np.abs(np.fft.rfft(faster)))]

    assert len(faster) == pytest.approx(len(audio) / 1.1, abs=1)
    assert peak_hz == pytest.approx(220, abs=2)
