# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated forwarding alias for the service protocols.

The protocol and message types moved to the private :mod:`xr_ai_models._protocols`.
Import them from the package root (``from xr_ai_models import VLMService``) instead;
this alias will be removed in a future version.
"""

from __future__ import annotations

import warnings

from ._protocols import (
    Capabilities,
    ChatMessage,
    ChatResponse,
    ContentPart,
    EmbeddingService,
    ImageInput,
    ImagePart,
    LLMService,
    STTService,
    TextPart,
    ToolCall,
    ToolDef,
    TTSService,
    VideoInput,
    VideoPart,
    VLMService,
)

warnings.warn(
    "xr_ai_models.protocols is deprecated; import from xr_ai_models instead. "
    "This alias will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Capabilities",
    "ChatMessage",
    "ChatResponse",
    "ContentPart",
    "EmbeddingService",
    "ImageInput",
    "ImagePart",
    "LLMService",
    "STTService",
    "TextPart",
    "ToolCall",
    "ToolDef",
    "TTSService",
    "VideoInput",
    "VideoPart",
    "VLMService",
]
