"""Per-participant VAD bookkeeping for the mcp-agent worker."""
from __future__ import annotations

from dataclasses import dataclass, field

from xr_ai_agent import AudioChunk


@dataclass
class VoiceState:
    chunks:      list[AudioChunk] = field(default_factory=list)
    speech_s:    float            = 0.0
    silent_s:    float            = 0.0
    sample_rate: int              = 16000
    channels:    int              = 1
    processing:  bool             = False
