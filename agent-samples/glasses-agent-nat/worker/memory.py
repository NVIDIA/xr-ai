# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AgentMemory — in-process timeline of observations and recorded demonstrations.

Also provides TranscriptClient, a thin wrapper around the NAT transcript-mcp
function group for source-scoped persistence and queries.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nat_runtime import NatRuntime

log = logging.getLogger("glasses_agent_nat.memory")


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    timestamp_us: int
    description:  str   # 1-2 sentence VLM description
    image_path:   str   # path to the PNG frame


@dataclass
class DemoStep:
    step_number:  int
    timestamp_us: int
    description:  str   # VLM description of what was observed at this step
    image_path:   str


@dataclass
class RecordedFrame:
    """One VLM observation captured during a demo recording session."""
    frame_idx:    int    # sequential index within the recording
    timestamp_us: int
    image_path:   str    # PNG written by video-mcp (may be gone after restart)
    description:  str    # detailed VLM analysis (2-4 sentences)


@dataclass
class VoiceNote:
    """A timestamped voice narration captured while the user was demonstrating."""
    timestamp_us: int
    text:         str


@dataclass
class Demonstration:
    name:             str
    started_at_us:    int
    ended_at_us:      int
    # Dense frame-by-frame VLM capture written during recording.
    recorded_frames:  list[RecordedFrame] = field(default_factory=list)
    # Voice narrations spoken by the user while demonstrating (timestamped).
    voice_notes:      list[VoiceNote]     = field(default_factory=list)
    # Steps and instructions populated by _analyze_recording AFTER recording.
    steps:            list[DemoStep]      = field(default_factory=list)
    summary:          str                 = ""
    instructions:     list[str]           = field(default_factory=list)


# ── AgentMemory ───────────────────────────────────────────────────────────────

class AgentMemory:
    """Rolling deque of observations + persistent demonstration store.

    Thread safety: all mutating methods are called from a single asyncio
    event loop — no locking required.
    """

    def __init__(self, max_obs: int = 240) -> None:
        self._max_obs       = max_obs
        self._observations: deque[Observation] = deque()
        self._scene_summary = ""
        self._demos: dict[str, Demonstration] = {}
        self._recording: Demonstration | None = None

    # ── observations ──────────────────────────────────────────────────────────

    def add_observation(self, obs: Observation) -> None:
        self._observations.append(obs)
        while len(self._observations) > self._max_obs:
            self._observations.popleft()

    # ── demo recording ────────────────────────────────────────────────────────

    def start_recording(self, name: str) -> None:
        """Begin a new demonstration recording."""
        ts = int(time.time() * 1_000_000)
        self._recording = Demonstration(name=name, started_at_us=ts, ended_at_us=0)
        log.info("demo recording started  name=%r", name)

    def add_voice_note(self, note: VoiceNote) -> None:
        """Store a voice narration captured while the user was demonstrating."""
        if self._recording is None:
            return
        self._recording.voice_notes.append(note)

    def add_recorded_frame(self, frame: RecordedFrame) -> None:
        """Append one VLM frame observation to the active recording.

        No-ops if recording was stopped between the VLM call being dispatched
        and this method being called (race condition guard).
        """
        if self._recording is None:
            return
        self._recording.recorded_frames.append(frame)

    def finish_recording(self) -> Demonstration | None:
        """End the current recording and return it for post-processing.

        The returned Demonstration has recorded_frames populated but steps/
        instructions empty — those are filled by _analyze_recording afterward.
        """
        if self._recording is None:
            log.warning("finish_recording called but no recording in progress")
            return None
        demo = self._recording
        demo.ended_at_us = int(time.time() * 1_000_000)
        self._recording = None
        if demo.recorded_frames:
            # Store immediately so guidance can start once analysis populates steps.
            self._demos[demo.name] = demo
            log.info("demo capture done  name=%r  frames=%d",
                     demo.name, len(demo.recorded_frames))
        else:
            log.warning("demo %r had no frames — discarding", demo.name)
        return demo

    def clear_demonstrations(self) -> int:
        """Delete all stored demonstrations. Returns count removed."""
        n = len(self._demos)
        self._demos.clear()
        return n

    @property
    def recording(self) -> Demonstration | None:
        return self._recording

    # ── scene summary ──────────────────────────────────────────────────────────

    def update_scene_summary(self, summary: str) -> None:
        self._scene_summary = summary.strip()

    # ── context building ──────────────────────────────────────────────────────

    def build_context(self, max_recent: int = 8) -> str:
        """Return a formatted string suitable for injection into the LLM context.

        Sections:
        - Scene summary (condensed from last ~60 s of observations).
        - Last N observation timeline entries.
        - Available demonstrations with step counts.
        """
        parts: list[str] = []

        # Scene summary.
        if self._scene_summary:
            parts.append(f"[Scene summary]\n{self._scene_summary}")
        else:
            parts.append("[Scene summary]\nNo scene summary available yet.")

        # Recent observations timeline.
        recent = list(self._observations)[-max_recent:]
        if recent:
            lines = ["[Recent observations]"]
            for obs in recent:
                # Format timestamp as seconds since epoch (readable).
                ts_s = obs.timestamp_us / 1_000_000
                hms  = time.strftime("%H:%M:%S", time.localtime(ts_s))
                lines.append(f"  {hms}  {obs.description}")
            parts.append("\n".join(lines))
        else:
            parts.append("[Recent observations]\nNone yet.")

        # Available demonstrations.
        if self._demos:
            lines = ["[Available demonstrations]"]
            for name, demo in self._demos.items():
                lines.append(f"  {name!r}: {len(demo.steps)} steps")
                if demo.summary:
                    lines.append(f"    Summary: {demo.summary}")
            parts.append("\n".join(lines))
        else:
            parts.append("[Available demonstrations]\nNone recorded yet.")

        return "\n\n".join(parts)

    # ── demonstration lookup ──────────────────────────────────────────────────

    def get_demonstration(self, name: str) -> Demonstration | None:
        return self._demos.get(name)

    def list_demonstrations(self) -> list[str]:
        return list(self._demos.keys())

    def find_demonstration_fuzzy(self, query: str) -> Demonstration | None:
        """Simple case-insensitive substring match against demo names.

        Returns the first matching demo, or ``None`` if no match.
        """
        q = query.lower()
        for name, demo in self._demos.items():
            if q in name.lower():
                return demo
        return None

    def restore_demonstration(self, demo: Demonstration) -> None:
        """Load a previously persisted demonstration into memory (startup restore)."""
        self._demos[demo.name] = demo

    def restore_observation(self, obs: Observation) -> None:
        """Load a previously persisted observation into the rolling deque (startup restore)."""
        self._observations.append(obs)
        while len(self._observations) > self._max_obs:
            self._observations.popleft()


# ── TranscriptClient ──────────────────────────────────────────────────────────

class TranscriptClient:
    """Client for transcript-mcp via the NAT MCP function group."""

    def __init__(self, runtime: NatRuntime) -> None:
        self._runtime = runtime

    async def _call(self, tool: str, args: dict) -> dict | list | None:
        try:
            result = await self._runtime.call_tool("transcript_mcp", tool, args)
            return result if isinstance(result, dict | list) else None
        except Exception as exc:
            log.warning("transcript-mcp call %s failed: %s", tool, exc)
            return None

    async def add_entry(self, source_id: str, timestamp_us: int, text: str) -> None:
        """Append a transcript entry under *source_id*."""
        await self._call("add_transcript", {
            "source_id":    source_id,
            "timestamp_us": timestamp_us,
            "text":         text,
        })

    async def query_recent(
        self, source_id: str, window_us: int
    ) -> list[dict]:
        """Return entries for *source_id* within the last *window_us* microseconds."""
        now_us  = int(time.time() * 1_000_000)
        start   = now_us - window_us
        result  = await self._call("query_transcripts", {
            "source_id": source_id,
            "start_us":  start,
            "end_us":    now_us,
        })
        if isinstance(result, list):
            return result
        return []

    async def close(self) -> None:
        pass  # no persistent connection to close
