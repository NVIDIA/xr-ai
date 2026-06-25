# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable, framework-agnostic agent capabilities for xr-ai samples.

Capabilities are self-contained features an agent brain can compose — they talk
to the hub through a ``ProcessorEndpoint`` and depend only on the core SDK
(``xr-ai-agent`` / ``xr-ai-models``), not on any voice/pipeline framework. The
first capability is :class:`VisionModule` (live-camera VLM question answering);
more (teacher-demo, agent-monitor) will follow here.
"""
from .agent_monitor import (
    FrameRef,
    GuidanceCheckResult,
    StepCheck,
    check_guidance_step_complete,
)
from .pixels import encode_image, frame_to_pil
from .teacher_demo import (
    AnalysisResult,
    RecordingFrame,
    StepKeyInfo,
    VoiceNote,
    analyze_recording,
    derive_step_key_info,
    derive_step_requirements,
)
from .vision import VisionModule, VisionUnavailable

__all__ = [
    # vision
    "VisionModule",
    "VisionUnavailable",
    "encode_image",
    "frame_to_pil",
    # teacher-demo
    "analyze_recording",
    "derive_step_requirements",
    "derive_step_key_info",
    "AnalysisResult",
    "StepKeyInfo",
    "RecordingFrame",
    "VoiceNote",
    # agent-monitor
    "check_guidance_step_complete",
    "GuidanceCheckResult",
    "StepCheck",
    "FrameRef",
]
