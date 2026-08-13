# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public voice runtime for XR agents.

Pipecat, audio framing, and pipeline processors are implementation details.
Applications register :class:`VoiceAgent`; :class:`VoiceSession` owns its media
pipeline and service lifecycle.
"""

from ._processors import VadConfig
from ._runtime import (
    VOICE_OUTPUT_TOPIC,
    UserQuery,
    VoiceAgent,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantLeft,
    VoiceStreamClosedError,
)
from ._session import VoiceSession
from ._transport import HubVoiceTransport

__all__ = [
    "HubVoiceTransport",
    "VadConfig",
    "VOICE_OUTPUT_TOPIC",
    "UserQuery",
    "VoiceAgent",
    "VoiceInterrupted",
    "VoiceOutput",
    "VoiceParticipantLeft",
    "VoiceStreamClosedError",
    "VoiceSession",
]
