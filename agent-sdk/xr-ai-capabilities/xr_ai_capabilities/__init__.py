# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable, framework-agnostic agent capabilities for xr-ai samples.

Capabilities are self-contained features an agent brain can compose — they talk
to the hub through a ``ProcessorEndpoint`` and depend only on the core SDK
(``xr-ai-agent`` / ``xr-ai-models``), not on any voice/pipeline framework. The
first capability is :class:`VisionModule` (live-camera VLM question answering);
more (teacher-demo, agent-monitor) will follow here.
"""
from .pixels import encode_image, frame_to_pil
from .vision import VisionModule, VisionUnavailable

__all__ = [
    "VisionModule",
    "VisionUnavailable",
    "encode_image",
    "frame_to_pil",
]
