# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AgentMemory — in-process timeline of observations and recorded demonstrations.

Also provides TranscriptClient, a thin HTTP client for the transcript-mcp
server (source: add_entry / query_recent via FastMCP tool calls).
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import json as _json

from fastmcp import Client as McpClient
from langchain_core.messages import SystemMessage

log = logging.getLogger("glasses_agent_langchain.memory")


_DEMO_LOOKUP_STOPWORDS = frozenset({
    "a", "an", "and", "do", "for", "how", "me", "show", "step", "task",
    "teach", "the", "through", "to", "walk",
})


def _normalize_demo_text(text: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    tokens = [
        token for token in clean.split()
        if token not in _DEMO_LOOKUP_STOPWORDS and not token.isdigit()
    ]
    return " ".join(tokens)


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
    task_index:       int                 = 0
    # Dense frame-by-frame VLM capture written during recording.
    recorded_frames:  list[RecordedFrame] = field(default_factory=list)
    # Voice narrations spoken by the user while demonstrating (timestamped).
    voice_notes:      list[VoiceNote]     = field(default_factory=list)
    # Steps and instructions populated by _analyze_recording AFTER recording.
    steps:            list[DemoStep]      = field(default_factory=list)
    summary:          str                 = ""
    instructions:     list[str]           = field(default_factory=list)


@dataclass(frozen=True)
class ObservationSnapshot:
    timestamp_us: int
    time:         str
    description:  str


@dataclass(frozen=True)
class DemoSnapshot:
    task_index:   int
    name:         str
    step_count:   int
    summary:      str
    instructions: tuple[str, ...]


@dataclass(frozen=True)
class MemorySnapshot:
    scene_summary:       str
    recent_observations: tuple[ObservationSnapshot, ...]
    demonstrations:      tuple[DemoSnapshot, ...]

    def format_context(self) -> str:
        """Format XR domain memory for model-visible context."""
        parts: list[str] = []

        if self.scene_summary:
            parts.append(f"[Scene summary]\n{self.scene_summary}")
        else:
            parts.append("[Scene summary]\nNo scene summary available yet.")

        if self.recent_observations:
            lines = ["[Recent observations]"]
            for obs in self.recent_observations:
                lines.append(f"  {obs.time}  {obs.description}")
            parts.append("\n".join(lines))
        else:
            parts.append("[Recent observations]\nNone yet.")

        if self.demonstrations:
            lines = ["[Available demonstrations]"]
            for demo in self.demonstrations:
                label = f"task {demo.task_index} -- {demo.name!r}"
                lines.append(f"  {label}: {demo.step_count} steps")
                if demo.summary:
                    lines.append(f"    Summary: {demo.summary}")
                if demo.instructions:
                    lines.append("    Instructions:")
                    for idx, instruction in enumerate(demo.instructions, start=1):
                        lines.append(f"      {idx}. {instruction}")
            parts.append("\n".join(lines))
        else:
            parts.append("[Available demonstrations]\nNone recorded yet.")

        return "\n\n".join(parts)

    def to_system_message(self) -> SystemMessage:
        """Return the memory snapshot as a LangChain system message fragment."""
        return SystemMessage(content=self.format_context())


def parse_mcp_result(result) -> dict | list | str | None:
    """Extract structured content or JSON text from a FastMCP tool result."""
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    items = getattr(result, "content", None) or []
    if items and hasattr(items[0], "text"):
        try:
            return _json.loads(items[0].text)
        except Exception:
            return items[0].text
    return None


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
        self._next_task_index = 1

    # ── observations ──────────────────────────────────────────────────────────

    def add_observation(self, obs: Observation) -> None:
        self._observations.append(obs)
        while len(self._observations) > self._max_obs:
            self._observations.popleft()

    # ── demo recording ────────────────────────────────────────────────────────

    def start_recording(self, name: str) -> None:
        """Begin a new demonstration recording."""
        ts = int(time.time() * 1_000_000)
        task_index = self._next_task_index
        self._next_task_index += 1
        self._recording = Demonstration(
            name=name,
            started_at_us=ts,
            ended_at_us=0,
            task_index=task_index,
        )
        log.info("demo recording started  task=%d  name=%r", task_index, name)

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
        self._next_task_index = 1
        return n

    @property
    def recording(self) -> Demonstration | None:
        return self._recording

    # ── scene summary ──────────────────────────────────────────────────────────

    def update_scene_summary(self, summary: str) -> None:
        self._scene_summary = summary.strip()

    # ── context building ──────────────────────────────────────────────────────

    def snapshot(self, max_recent: int = 8) -> MemorySnapshot:
        """Return a structured read-only snapshot for agent runtime context."""
        recent = []
        for obs in list(self._observations)[-max_recent:]:
            ts_s = obs.timestamp_us / 1_000_000
            recent.append(ObservationSnapshot(
                timestamp_us = obs.timestamp_us,
                time         = time.strftime("%H:%M:%S", time.localtime(ts_s)),
                description  = obs.description,
            ))

        demos = []
        for _, demo in self.list_demonstrations_with_indices():
            demos.append(DemoSnapshot(
                task_index   = demo.task_index,
                name         = demo.name,
                step_count   = len(demo.steps),
                summary      = demo.summary,
                instructions = tuple(demo.instructions),
            ))

        return MemorySnapshot(
            scene_summary       = self._scene_summary,
            recent_observations = tuple(recent),
            demonstrations      = tuple(demos),
        )

    def build_context(self, max_recent: int = 8) -> str:
        """Return a formatted string suitable for injection into the LLM context.

        Sections:
        - Scene summary (condensed from last ~60 s of observations).
        - Last N observation timeline entries.
        - Available demonstrations with step counts.
        """
        return self.snapshot(max_recent=max_recent).format_context()

    # ── demonstration lookup ──────────────────────────────────────────────────

    def get_demonstration(self, name: str) -> Demonstration | None:
        return self._demos.get(name)

    def list_demonstrations(self) -> list[str]:
        return list(self._demos.keys())

    def list_demonstrations_with_indices(self) -> list[tuple[int, Demonstration]]:
        return sorted(
            ((demo.task_index, demo) for demo in self._demos.values()),
            key=lambda item: item[0],
        )

    def get_demonstration_by_task_index(self, index: int) -> Demonstration | None:
        for demo in self._demos.values():
            if demo.task_index == index:
                return demo
        return None

    def find_demonstration_fuzzy(self, query: str) -> Demonstration | None:
        """Find a demo by normalized name or token overlap.

        Returns the first matching demo, or ``None`` if no match.
        """
        q = _normalize_demo_text(query)
        if not q:
            return None

        # Prefer exact containment in either direction.
        for _, demo in self.list_demonstrations_with_indices():
            name = _normalize_demo_text(demo.name)
            if q in name or name in q:
                return demo

        q_tokens = set(q.split())
        best: tuple[int, Demonstration] | None = None
        for _, demo in self.list_demonstrations_with_indices():
            name_tokens = set(_normalize_demo_text(demo.name).split())
            if not name_tokens:
                continue
            score = len(q_tokens & name_tokens)
            if score and score >= max(1, min(2, len(name_tokens))):
                if best is None or score > best[0]:
                    best = (score, demo)
        if best is not None:
            return best[1]
        return None

    def restore_demonstration(self, demo: Demonstration) -> None:
        """Load a previously persisted demonstration into memory (startup restore)."""
        if demo.task_index <= 0:
            demo.task_index = self._next_task_index
        self._next_task_index = max(self._next_task_index, demo.task_index + 1)
        self._demos[demo.name] = demo

    def restore_observation(self, obs: Observation) -> None:
        """Load a previously persisted observation into the rolling deque (startup restore)."""
        self._observations.append(obs)
        while len(self._observations) > self._max_obs:
            self._observations.popleft()


# ── TranscriptClient ──────────────────────────────────────────────────────────

class TranscriptClient:
    """Client for transcript-mcp via FastMCP StreamableHTTP transport."""

    def __init__(self, base_url: str) -> None:
        self._mcp = base_url.rstrip("/") + "/mcp"

    async def _call(self, tool: str, args: dict) -> dict | list | None:
        try:
            async with McpClient(self._mcp) as client:
                result = await client.call_tool(tool, args)
            return parse_mcp_result(result)
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
