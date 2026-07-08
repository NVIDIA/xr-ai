# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable, framework-agnostic agent skills for xr-ai samples.

Skills are self-contained features an agent brain can compose — they talk
to the hub through a ``ProcessorEndpoint`` and depend only on the core SDK
(``xr-ai-agent`` / ``xr-ai-models``), not on any voice/pipeline framework.
``LiveFrameSource`` is the shared live-camera frame-acquisition primitive
(also used by ``video-mcp``, out-of-process, for the same job); the first
skill built on it is :class:`VisionModule` (live-camera VLM question
answering). More (teacher-demo, agent-monitor) will follow here.
"""
from .frame_source import LiveFrameSource, LiveFrameUnavailable
from .pixels import encode_image, frame_to_pil
from .vision import VisionModule, VisionUnavailable

__all__ = [
    "LiveFrameSource",
    "LiveFrameUnavailable",
    "VisionModule",
    "VisionUnavailable",
    "encode_image",
    "frame_to_pil",
]
