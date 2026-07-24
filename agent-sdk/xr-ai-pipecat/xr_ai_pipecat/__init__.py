# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr-ai-pipecat — unified Pipecat voice pipeline for xr-ai agents.

Sample workers compose their :class:`BrainProcessor` with
:func:`make_voice_pipeline`, then use :func:`run_voice_pipeline` to run it.
The package supplies VAD/STT, voice gate, and streaming TTS.

Live-camera VLM Q&A lives in the framework-agnostic ``xr-ai-capabilities``
package (``VisionModule``); a pipecat brain wires it up by passing
``transport.endpoint``.
"""
from __future__ import annotations

from .frames import (
    BrainResponseEndFrame,
    GatedQueryFrame,
    ParticipantJoinedFrame,
    ParticipantLeftFrame,
)
from .pipeline import make_voice_pipeline, run_voice_pipeline
from .processors import (
    BrainProcessor,
    StreamingTtsProcessor,
    VadConfig,
    VadSttProcessor,
    VoiceGateProcessor,
)

__all__ = [
    "BrainProcessor",
    "BrainResponseEndFrame",
    "GatedQueryFrame",
    "ParticipantJoinedFrame",
    "ParticipantLeftFrame",
    "StreamingTtsProcessor",
    "VadConfig",
    "VadSttProcessor",
    "VoiceGateProcessor",
    "make_voice_pipeline",
    "run_voice_pipeline",
]
