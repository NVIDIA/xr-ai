# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Participant-scoped voice topics for replay, separate from capture narration."""

from xr_ai_runtime import Topic
from xr_ai_voice import UserQuery, VoiceInterrupted, VoiceParticipantJoined, VoiceParticipantLeft

USER_QUERY_TOPIC = Topic("sop.replay.query", UserQuery)
PARTICIPANT_JOINED_TOPIC = Topic("sop.replay.joined", VoiceParticipantJoined)
PARTICIPANT_LEFT_TOPIC = Topic("sop.replay.left", VoiceParticipantLeft)
INTERRUPTED_TOPIC = Topic("sop.replay.interrupted", VoiceInterrupted)
