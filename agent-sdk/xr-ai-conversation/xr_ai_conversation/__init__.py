# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public surface of the xr-ai-conversation package."""
from __future__ import annotations

from .audio import (
    chunks_to_wav,
    int16_pcm_to_wav,
    now_us,
    split_sentences,
    wav_to_chunks,
)
from .gate_wiring import GateBindings, QueryHandler, QueryResult, wire_voice_gate
from .loop import ConversationLoop, VadConfig
from .state import VoiceState
from .streaming import stream_text_to_audio

__all__ = [
    # primary entry point
    "ConversationLoop",
    "VadConfig",
    # helpers exposed for samples that want to reuse pieces
    "GateBindings",
    "QueryHandler",
    "QueryResult",
    "VoiceState",
    "stream_text_to_audio",
    "wire_voice_gate",
    # audio codec helpers
    "chunks_to_wav",
    "int16_pcm_to_wav",
    "now_us",
    "split_sentences",
    "wav_to_chunks",
]
