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


# ── demo name normalization (used by find_demonstration_fuzzy) ───────────────

# Baseline set from agent-samples/glasses-agent/worker/memory.py. Do NOT add
# action words like "wear" / "put" / "on": a future demo named "put on
# headset" would normalize to just "headset" and silently collapse with any
# other "headset" demo on a one-token match.
_DEMO_LOOKUP_STOPWORDS = frozenset({
    "a", "an", "and", "do", "for", "how", "me", "show", "step", "task",
    "teach", "the", "through", "to", "walk",
})


def _fold_plural(t: str) -> str:
    """Strip trailing 's'/'es' when the result is still substantive.

    Protects short words like 'bus', 'gas', 'yes'; catches the common STT
    artefact 'headsets' -> 'headset', 'cases' -> 'case' seen in the
    20260526_162310 trace.
    """
    if len(t) >= 6 and t.endswith("es") and not t.endswith("ses"):
        return t[:-2]
    if len(t) >= 5 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _normalize_demo_text(text: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return " ".join(
        _fold_plural(t) for t in clean.split()
        if t not in _DEMO_LOOKUP_STOPWORDS and not t.isdigit()
    )


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    timestamp_us: int
    description:  str   # 1-2 sentence VLM description
    image_path:   str   # path to the PNG frame


@dataclass
class StepKeyInfo:
    """Structured distillation of a step's long text description.

    Turns the free-text instruction + teacher caption into the few facts that
    actually define the step, so the student-monitoring VLM checks ONLY these
    and is not thrown off by irrelevant differences (background, lighting,
    camera angle, clothing, room). ``ignore`` names details that explicitly
    must not affect the verdict.
    """
    objects:      list[str] = field(default_factory=list)  # key objects, e.g. ["VR headset", "strap"]
    action:       str = ""   # the action performed, e.g. "place on head"
    position:     str = ""   # spatial relationship / placement to verify
    target_state: str = ""   # the end state that defines "done", e.g. "headset worn, strap snug"
    ignore:       list[str] = field(default_factory=list)  # irrelevant details to disregard

    def is_empty(self) -> bool:
        return not (self.objects or self.action or self.position or self.target_state)

    def as_prompt_block(self) -> str:
        """Render the key info as a compact block for VLM/LLM prompts."""
        lines: list[str] = []
        if self.objects:
            lines.append(f"  Key objects: {', '.join(self.objects)}")
        if self.action:
            lines.append(f"  Action: {self.action}")
        if self.position:
            lines.append(f"  Position/placement: {self.position}")
        if self.target_state:
            lines.append(f"  Target end-state: {self.target_state}")
        ignore = list(self.ignore) or ["background", "lighting", "camera angle"]
        lines.append(f"  IGNORE (must not affect the verdict): {', '.join(ignore)}")
        return "\n".join(lines)


@dataclass
class DemoStep:
    step_number:     int
    timestamp_us:    int
    description:     str   # Analyzed instruction text for this step (from _analyze_recording)
    image_path:      str
    teacher_caption: str = ""  # VLM description of the after-state frame (reference context only)
    # Atomic visually-checkable requirements derived from the instruction at
    # finalization time. The completion parser uses these as the authoritative
    # checklist; teacher_caption only labels the reference image.
    expected_requirements: list[str] = field(default_factory=list)
    # Structured key info distilled from the description at finalize time.
    # When present, student monitoring checks ONLY these facts (objects /
    # action / position / target_state) and ignores irrelevant differences.
    key_info: StepKeyInfo | None = None
    reference_image_paths: list[str] = field(default_factory=list)
    reference_reliable: bool = True
    text_video_mismatch: bool = False


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
    # True between finish_recording() and the end of the async analysis
    # task — the worker uses this to tell the user "still analyzing"
    # rather than "no steps recorded yet" when guidance is requested.
    is_finalizing:        bool = False
    # Generation token stamped at finish_recording() time; checked by
    # _finalize_demo before mutating the demo or speaking, so a task that
    # races past `clear_demonstrations()` is neutered.
    finalize_generation:  int  = 0


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
        # Wall-clock μs of the last finish_recording() call. Used by the
        # demo→guidance freshness fallback: within ~120 s of a finished
        # demo, an ambiguous follow-up almost certainly means "guide me
        # through what I just recorded".
        self._last_demo_finished_at_us: int = 0
        # Bumped by clear_demonstrations(); _finalize_demo checks its
        # stamped value before mutating or speaking, so a finalize task
        # that races past a clear is silently dropped.
        self._finalize_generation:      int = 0

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
        Also stamps ``_last_demo_finished_at_us`` so ``demo_is_fresh`` can
        decide whether an ambiguous follow-up utterance should default to
        guiding the wearer through what they just demonstrated.
        """
        if self._recording is None:
            log.warning("finish_recording called but no recording in progress")
            return None
        demo = self._recording
        demo.ended_at_us = int(time.time() * 1_000_000)
        self._recording = None
        self._last_demo_finished_at_us = demo.ended_at_us
        if demo.recorded_frames:
            # Store immediately so guidance can start once analysis populates steps.
            demo.is_finalizing       = True
            demo.finalize_generation = self._finalize_generation
            self._demos[demo.name]   = demo
            log.info("demo capture done  name=%r  frames=%d",
                     demo.name, len(demo.recorded_frames))
        else:
            log.warning("demo %r had no frames — discarding", demo.name)
        return demo

    def demo_is_fresh(self, window_s: float) -> bool:
        """Return True if the most recent demo finished within *window_s* seconds.

        Used by the noise gate (skip the LLM intent classifier — mangled
        post-demo utterances are almost always guidance requests) and by
        the worker-side demo→guidance fallback.
        """
        if self._last_demo_finished_at_us == 0:
            return False
        now_us   = int(time.time() * 1_000_000)
        delta_us = now_us - self._last_demo_finished_at_us
        return 0 <= delta_us <= int(window_s * 1_000_000)

    def most_recent_demo(self) -> Demonstration | None:
        """Return the demo with the highest ended_at_us, or None if no demos."""
        if not self._demos:
            return None
        return max(self._demos.values(), key=lambda d: d.ended_at_us)

    def clear_demonstrations(self) -> int:
        """Delete all stored demonstrations. Returns count removed.

        Bumps ``_finalize_generation`` first so any in-flight
        ``_finalize_demo`` task that survives cancellation and reaches
        its identity-check sees a stale generation token and exits
        silently instead of announcing ``"Saved 'X'"`` for a demo that
        no longer exists.
        """
        self._finalize_generation += 1
        n = len(self._demos)
        self._demos.clear()
        return n

    def is_current_finalization(
        self, demo: Demonstration, generation: int,
    ) -> bool:
        """Whether a finalize task for *demo* is still authoritative.

        True only when BOTH (a) the generation token stamped on the task
        still matches the memory's current generation (no
        ``clear_demonstrations()`` happened), AND (b) the demo dict's
        entry for ``demo.name`` is the SAME object as *demo* (so a
        re-record under the same name during finalization invalidates
        the prior task).
        """
        return (
            self._finalize_generation == generation
            and self._demos.get(demo.name) is demo
        )

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

    def find_demonstration_fuzzy(
        self,
        query: str,
        *,
        min_confidence: str = "strict",
    ) -> Demonstration | None:
        """Find a demo by normalized name + token overlap.

        ``min_confidence``:
          ``"strict"``  -- substring match either direction, or token
                           overlap >= 2 with a unique-best demo. Used by
                           the freshness fallback so a stray utterance
                           with one weak token in common doesn't silently
                           override the recent demo.
          ``"lenient"`` -- ``"strict"``, plus a single-token overlap
                           when it is the UNIQUE best candidate. Used by
                           explicit pending-disambiguation where the user
                           is expected to be answering a "which demo?"
                           question, so weak-but-unique evidence is enough.

        Returns ``None`` when no demo passes the threshold.
        """
        q = _normalize_demo_text(query)
        if not q:
            return None

        substr_hits = [
            demo for name, demo in self._demos.items()
            if (n := _normalize_demo_text(name)) and (q in n or n in q)
        ]
        if len(substr_hits) == 1:
            return substr_hits[0]
        if len(substr_hits) > 1:
            return None

        q_tokens = set(q.split())
        scored: list[tuple[int, Demonstration]] = []
        for name, demo in self._demos.items():
            name_tokens = set(_normalize_demo_text(name).split())
            if not name_tokens:
                continue
            score = len(q_tokens & name_tokens)
            if score:
                scored.append((score, demo))
        if not scored:
            return None

        max_score = max(s for s, _ in scored)
        top = [d for s, d in scored if s == max_score]

        threshold = 2 if min_confidence == "strict" else 1
        if max_score < threshold:
            return None
        if len(top) != 1:
            return None
        return top[0]

    def restore_observation(self, obs: Observation) -> None:
        """Load a previously persisted observation into the rolling deque (startup restore).

        Note: only observations are restored across sessions. Demonstrations
        intentionally start clean each run — see ``_restore_memory`` in
        ``glasses_agent_nat_worker.py`` for the rationale.
        """
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
