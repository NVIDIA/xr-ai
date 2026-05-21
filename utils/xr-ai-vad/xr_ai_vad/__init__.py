# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Silero-VAD utterance detector for xr-ai agent workers.

Consumes float32 PCM bytes (the format XR Media Hub ``AudioChunk.data`` uses)
and emits int16 PCM utterance bytes via an async callback when speech ends.
Falls back to an adaptive energy gate when ``silero-vad`` is unavailable.
"""
from xr_ai_vad.detector import OnSpeechStartCb, OnUtteranceCb, VadDetector

__all__ = ["VadDetector", "OnUtteranceCb", "OnSpeechStartCb"]
