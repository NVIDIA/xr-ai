# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-participant VAD + interruption bookkeeping."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from xr_ai_agent import AudioChunk


@dataclass
class VoiceState:
    chunks:        list[AudioChunk]      = field(default_factory=list)
    speech_s:      float                 = 0.0
    silent_s:      float                 = 0.0
    sample_rate:   int                   = 16000
    channels:      int                   = 1
    transcribing:  bool                  = False  # in-flight STT for this pid
    # In-flight VLM+TTS response.  A new query cancels this; the dispatch
    # lock serialises the cancel-await-flush-restart sequence per pid.
    current_task:  asyncio.Task | None   = None
    dispatch_lock: asyncio.Lock          = field(default_factory=asyncio.Lock)
