# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-participant voice-loop bookkeeping owned by ``ConversationLoop``.

The loop owns the ``VadDetector`` lifecycle for each participant; this
struct only tracks the cancel/flush state for in-flight queries and a
single STT-in-flight flag so two utterances from the same pid don't race
through the gate at the same time.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from xr_ai_vad import VadDetector


@dataclass
class VoiceState:
    vad:           VadDetector | None  = None
    transcribing:  bool                = False  # in-flight STT for this pid
    # In-flight user-supplied response.  A new query cancels this; the
    # dispatch lock serialises the cancel-await-flush-restart sequence
    # per pid so concurrent calls can't both see a stale ``current_task``.
    current_task:  asyncio.Task | None = None
    dispatch_lock: asyncio.Lock        = field(default_factory=asyncio.Lock)
