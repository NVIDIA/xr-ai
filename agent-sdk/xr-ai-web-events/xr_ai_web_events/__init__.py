# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live, participant-aware browser views over typed agent events."""

from ._agent import WebEventsAgent
from ._models import WEB_EVENT_TOPIC, WebEvent

__all__ = [
    "WEB_EVENT_TOPIC",
    "WebEvent",
    "WebEventsAgent",
]
