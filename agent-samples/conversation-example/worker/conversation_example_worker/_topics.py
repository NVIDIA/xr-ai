# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name the voice input and lifecycle topics configured by this sample."""

from xr_ai_interaction import Interrupted, ParticipantJoined, ParticipantLeft, UserQuery
from xr_ai_runtime import Topic

USER_QUERY_TOPIC = Topic("conversation-example.user-query", UserQuery)
PARTICIPANT_LEFT_TOPIC = Topic("conversation-example.participant-left", ParticipantLeft)
INTERRUPTED_TOPIC = Topic("conversation-example.interrupted", Interrupted)
PARTICIPANT_JOINED_TOPIC = Topic("conversation-example.participant-joined", ParticipantJoined)
