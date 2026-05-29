# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public surface of the xr-ai-voicegate package."""
from __future__ import annotations

from .config import AudioSink, TTSLike, VoiceGateConfig
from .gate import VoiceGate

__all__ = [
    "AudioSink",
    "TTSLike",
    "VoiceGate",
    "VoiceGateConfig",
]
