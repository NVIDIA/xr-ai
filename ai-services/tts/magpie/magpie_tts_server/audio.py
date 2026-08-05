# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio post-processing for Magpie speech."""

import librosa
import numpy as np


def change_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Change speaking rate while preserving pitch."""
    if speed == 1.0:
        return audio
    return librosa.effects.time_stretch(audio, rate=speed)
