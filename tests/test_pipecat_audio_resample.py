# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression coverage for #193: hub multi-channel audio must be downmixed to
mono BEFORE resampling. Resampling an interleaved (L R L R …) buffer as a
single stream mixes adjacent channels and yields the wrong sample count.

Tests the pure helper ``_hub_pcm_to_mono_16k`` — no pipeline/transport needed.
"""
from __future__ import annotations

import numpy as np

from xr_ai_pipecat.transport import SAMPLE_RATE, _hub_pcm_to_mono_16k


def test_mono_16k_is_byte_identical_passthrough() -> None:
    pcm = np.array([1, -2, 3, -4, 5], dtype=np.int16).tobytes()
    assert _hub_pcm_to_mono_16k(pcm, 1, SAMPLE_RATE) == pcm


def test_stereo_downmix_averages_channels() -> None:
    # Interleaved L R: (100,200),(100,200) → mono mean 150,150 (no resample).
    pcm = np.array([100, 200, 100, 200], dtype=np.int16).tobytes()
    out = np.frombuffer(_hub_pcm_to_mono_16k(pcm, 2, SAMPLE_RATE), dtype=np.int16)
    assert out.tolist() == [150, 150]


def test_mono_resample_48k_to_16k_length_and_value() -> None:
    # 480 mono samples @48k → ~160 @16k; a constant signal stays ~constant.
    src = np.full(480, 500, dtype=np.int16).tobytes()
    out = np.frombuffer(_hub_pcm_to_mono_16k(src, 1, 48_000), dtype=np.int16)
    assert 150 <= len(out) <= 170
    assert abs(int(out[len(out) // 2]) - 500) <= 5


def test_stereo_48k_downmixes_to_mono_16k_not_interleaved() -> None:
    # 480 interleaved STEREO frames @48k. Correct: downmix → 480 mono @48k →
    # ~160 @16k. The #193 bug resampled the 960-value interleaved buffer as one
    # stream → ~320 samples (and mislabeled them stereo). The sample COUNT is
    # the discriminator: ~160 (fixed) vs ~320 (buggy).
    inter = np.empty(480 * 2, dtype=np.int16)
    inter[0::2] = 700  # L
    inter[1::2] = 700  # R
    out = np.frombuffer(
        _hub_pcm_to_mono_16k(inter.tobytes(), 2, 48_000), dtype=np.int16,
    )
    assert 150 <= len(out) <= 170, f"expected ~160 mono samples, got {len(out)}"
    assert abs(int(out[len(out) // 2]) - 700) <= 5
