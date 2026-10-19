# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define the runtime topics that connect voice input to the guide agent.

The worker and GuideAgent use these topics for user requests, participant
connections, and speech interruptions. Keeping the topic definitions together
lets both sides share the same event contract, separate from capture narration.
"""

from xr_ai_runtime import Topic
from xr_ai_voice import UserQuery, VoiceInterrupted, VoiceParticipantJoined, VoiceParticipantLeft

USER_QUERY_TOPIC = Topic("sop.guide.query", UserQuery)
PARTICIPANT_JOINED_TOPIC = Topic("sop.guide.joined", VoiceParticipantJoined)
PARTICIPANT_LEFT_TOPIC = Topic("sop.guide.left", VoiceParticipantLeft)
INTERRUPTED_TOPIC = Topic("sop.guide.interrupted", VoiceInterrupted)
