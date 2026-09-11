# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant lifecycle topics used by the recorder."""

from xr_ai_runtime import Topic
from xr_ai_voice import UserQuery, VoiceParticipantJoined, VoiceParticipantLeft

USER_QUERY_TOPIC = Topic("workflow-recorder.user-query", UserQuery)
PARTICIPANT_JOINED_TOPIC = Topic(
    "workflow-recorder.participant-joined",
    VoiceParticipantJoined,
)
PARTICIPANT_LEFT_TOPIC = Topic(
    "workflow-recorder.participant-left",
    VoiceParticipantLeft,
)
