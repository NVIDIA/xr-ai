# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public voice runtime for XR agents.

Applications configure and register :class:`VoiceAgent`, and may pass it an
update ``combiner`` such as :func:`llm_update_combiner`; media sessions,
audio framing, and pipeline processors are implementation details.
"""

from ._combiner import llm_update_combiner
from ._processors import VadConfig
from ._runtime import VOICE_OUTPUT_TOPIC, VOICE_TRANSCRIPT_TOPIC, VoiceAgent, VoiceStreamClosedError, VoiceTranscript
from ._transport import HubVoiceTransport

__all__ = [
    "HubVoiceTransport",
    "VadConfig",
    "llm_update_combiner",
    "VOICE_OUTPUT_TOPIC",
    "VOICE_TRANSCRIPT_TOPIC",
    "VoiceAgent",
    "VoiceStreamClosedError",
    "VoiceTranscript",
]
