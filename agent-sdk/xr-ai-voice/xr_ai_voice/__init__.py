# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public voice-session API for XR agents.

Pipecat, audio framing, and pipeline processors are implementation details.
Applications create a :class:`VoiceSession` and supply an async
:class:`VoiceHandler` callable.
"""

from ._handler import VoiceHandler, VoiceQuery, VoiceResponse, VoiceTurn
from ._processors import VadConfig
from ._session import VoiceSession
from ._text_input import TextMessageInput
from ._transport import HubVoiceTransport

__all__ = [
    "HubVoiceTransport",
    "TextMessageInput",
    "VadConfig",
    "VoiceHandler",
    "VoiceQuery",
    "VoiceResponse",
    "VoiceSession",
    "VoiceTurn",
]
