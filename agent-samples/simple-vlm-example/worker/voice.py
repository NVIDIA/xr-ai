# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-participant VAD + interruption bookkeeping."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from xr_ai_vad import VadDetector


@dataclass
class VoiceState:
    vad:           VadDetector | None    = None
    transcribing:  bool                  = False  # in-flight STT for this pid
    # In-flight VLM+TTS response.  A new query cancels this; the dispatch
    # lock serialises the cancel-await-flush-restart sequence per pid.
    current_task:  asyncio.Task | None   = None
    dispatch_lock: asyncio.Lock          = field(default_factory=asyncio.Lock)
    # Monotonic timestamp until which the agent is expected to still be
    # speaking (sum of queued TTS audio durations). Incoming VAD
    # utterances captured before this time are dropped — the agent
    # hearing its own voice through the mic was triggering bogus
    # follow-up queries.
    speaking_until: float                = 0.0
