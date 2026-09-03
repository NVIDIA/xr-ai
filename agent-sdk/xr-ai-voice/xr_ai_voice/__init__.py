# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public voice runtime for XR agents.

Applications configure and register :class:`VoiceAgent` and may route
concurrent producers through :class:`VoiceAggregationAgent`; media sessions,
audio framing, and pipeline processors are implementation details.
"""

from ._aggregation import VOICE_CONTRIBUTION_TOPIC, VoiceAggregationAgent
from ._processors import VadConfig
from ._runtime import (
    VOICE_OUTPUT_TOPIC,
    VOICE_TRANSCRIPT_TOPIC,
    UserQuery,
    VoiceAgent,
    VoiceInterrupted,
    VoiceOutput,
    VoiceParticipantJoined,
    VoiceParticipantLeft,
    VoicePriority,
    VoiceStreamClosedError,
    VoiceTranscript,
)
from ._transport import HubVoiceTransport

__all__ = [
    "HubVoiceTransport",
    "VadConfig",
    "VOICE_CONTRIBUTION_TOPIC",
    "VOICE_OUTPUT_TOPIC",
    "VOICE_TRANSCRIPT_TOPIC",
    "UserQuery",
    "VoiceAggregationAgent",
    "VoiceAgent",
    "VoiceInterrupted",
    "VoiceOutput",
    "VoiceParticipantJoined",
    "VoiceParticipantLeft",
    "VoicePriority",
    "VoiceStreamClosedError",
    "VoiceTranscript",
]
