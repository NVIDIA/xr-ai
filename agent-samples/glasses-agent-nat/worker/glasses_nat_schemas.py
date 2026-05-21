# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for glasses-agent NAT functions."""
from __future__ import annotations

from pydantic import BaseModel, Field


class FrameEntry(BaseModel):
    frame_idx: int = Field(description="Zero-based index within the recording.")
    timestamp_us: int = Field(description="Frame timestamp in microseconds.")
    image_path: str = Field(description="Stable image path for the recorded frame.")
    description: str = Field(description="VLM description captured for this frame.")


class VoiceNoteEntry(BaseModel):
    timestamp_us: int = Field(description="Voice note timestamp in microseconds.")
    text: str = Field(description="User narration captured during the recording.")


class AnalyzeRecordingInput(BaseModel):
    name: str = Field(description="Demonstration name.")
    started_at_us: int = Field(description="Recording start timestamp.")
    frames: list[FrameEntry] = Field(default_factory=list)
    voice_notes: list[VoiceNoteEntry] = Field(default_factory=list)


class AnalyzeRecordingOutput(BaseModel):
    overview: str = ""
    steps: list[str] = Field(default_factory=list)


class ObservationEntry(BaseModel):
    timestamp_us: int
    description: str


class CondenseObservationsInput(BaseModel):
    observations: list[ObservationEntry] = Field(default_factory=list)


class CondenseObservationsOutput(BaseModel):
    overview: str = ""
    events: list[dict] = Field(default_factory=list)
    summary_text: str = ""


class GuidanceStepInput(BaseModel):
    participant_id: str
    instruction: str


class GuidanceStepOutput(BaseModel):
    completed: bool = False
    raw: str = ""
    image_path: str = ""
