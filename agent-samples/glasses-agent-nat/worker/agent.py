# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
glasses-agent-nat — always-on smart-glasses assistant on the pipecat pipeline.

Voice I/O is the unified ``xr_ai_pipecat`` pipeline, exactly like the other
agent samples::

    transport.input → VadStt → VoiceGate(always-on) → GlassesBrain
                     → StreamingTts → transport.output

:class:`GlassesBrain` is the pipecat ``BrainProcessor`` for this sample. It is
thin glue around the NAT-backed :class:`processors.QueryProcessor`, which is the
real "brain" (demo detection, guidance, agentic tool loop). The NAT runtime
stays *inside* the brain — ``handle_query`` delegates to ``QueryProcessor.handle``
and the processor drives output through the brain's ``say`` / ``send_text`` /
``flush_audio`` callbacks, which map onto pipeline frames:

  * ``say``         → a ``TextFrame`` (+ ``BrainResponseEndFrame`` flush) so the
                      shared ``StreamingTtsProcessor`` synthesizes + paces it;
  * ``send_text``   → ``transport.send_return_data`` on a custom topic
                      (``agent.response`` / ``agent.progress``);
  * ``flush_audio`` → an ``InterruptionFrame`` so the TTS sender drains and the
                      hub return-audio buffer is flushed (barge-in).

The smart-glasses voice gate is *not* a wake word — it is a two-layer
noise/intent filter (cheap shape gate + an LLM intent classifier) plus a
guidance-mode control filter. That gate runs in ``process_frame`` on each gated
query *before* the base class spawns a turn, so rejected noise never supersedes
an in-flight answer. Typed text on the data channel bypasses the gate (the user
explicitly sent it).

:class:`GlassesPerception` owns the always-on background work that is orthogonal
to voice I/O: the periodic VLM scene-observation loop, dense recording capture,
and the memory condenser. These reach frames/VLM exclusively through NAT MCP
(video-mcp / vlm-mcp), never the audio pipeline, so they live outside the brain
and are started/stopped by the worker.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import re as _re
import shutil as _shutil
import time
from typing import Callable

import httpx
from config import WorkerConfig
from intent import is_real_assistant_request, is_shape_noise
from memory import AgentMemory, Observation, RecordedFrame, TranscriptClient
from nat_runtime import NatRuntime
from pipecat.frames.frames import Frame, InterruptionFrame
from pipecat.processors.frame_processor import FrameDirection
from processors import QueryProcessor
from xr_ai_agent import DataMessage
from xr_ai_pipecat import BrainProcessor, BrainResponseEndFrame, GatedQueryFrame
from xr_ai_pipecat.transport import XRMediaHubTransport

log        = logging.getLogger("glasses_agent_nat.agent")
_trace_log = logging.getLogger("glasses_agent_nat.trace")  # shared with glasses_agent_nat_worker

_DEFAULT_PID = "web-client"


def _now_us() -> int:
    return time.time_ns() // 1_000


class GlassesBrain(BrainProcessor):
    """Pipecat brain for glasses-agent-nat — thin glue over ``QueryProcessor``.

    Parameters
    ----------
    transport:
        Shared ``XRMediaHubTransport``; the brain registers the typed-text data
        side path on its endpoint and sends ``agent.*`` data messages through
        it. Passed to ``BrainProcessor`` so single-participant return routing is
        steered automatically on the first join.
    cfg:
        ``WorkerConfig`` — VAD/gate thresholds, the intent-gate LLM server, etc.
    memory:
        Shared ``AgentMemory`` — read by the gate (recording / fresh-demo state).
    query_processor:
        The NAT-backed brain. ``handle_query`` delegates to its ``handle`` and it
        drives output via this brain's ``say`` / ``send_text`` / ``flush_audio``.
    """

    def __init__(
        self,
        *,
        transport:       XRMediaHubTransport,
        cfg:             WorkerConfig,
        memory:          AgentMemory,
        query_processor: QueryProcessor,
    ) -> None:
        super().__init__(transport=transport)
        self._cfg    = cfg
        self._memory = memory
        self._qproc  = query_processor

        # The worker links the background perception loops here so the
        # observation timeline can be reset when a participant (re)joins.
        self.perception: GlassesPerception | None = None

        # HTTP client for the worker's own LLM intent gate (Layer 2). STT/TTS
        # ride the pipeline; this is the only direct model call the brain makes.
        self._http = httpx.AsyncClient(timeout=120.0)

        # Typed-text side path. The web client's "Send" button publishes text on
        # the data channel; feed it through the same turn machinery as a spoken
        # utterance, bypassing VAD/STT and the noise/intent gate (the user
        # explicitly sent it). Mirrors simple-vlm-example / xr-render-demo.
        transport.endpoint.on_data(self._on_data)

    # ── state exposed to the background perception loops ───────────────────────

    @property
    def active_pid(self) -> str:
        """The participant the background loops should address — the first
        joined pid, or the ``web-client`` default when none is known yet (a
        client may have connected before the worker saw the join)."""
        for pid in self._joined:
            return pid
        return _DEFAULT_PID

    @property
    def query_active(self) -> bool:
        """True while a user-facing turn is in flight — the observation loop
        yields VLM bandwidth to it."""
        return bool(self._inflight)

    # ── BrainProcessor overrides ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Smart-glasses voice gate: run the noise/intent filter on each gated
        # query BEFORE the base class spawns a turn. Dropping here (rather than
        # inside handle_query) means rejected noise never fires the supersede
        # hook / cancels an in-flight answer.
        if isinstance(frame, GatedQueryFrame):
            if not await self._passes_gate(frame.participant_id, frame.text):
                return
        await super().process_frame(frame, direction)

    async def handle_query(self, pid: str, text: str, fresh_match: bool) -> str:
        """Delegate the whole turn to the NAT brain.

        ``QueryProcessor.handle`` produces all output itself via the ``say`` /
        ``send_text`` callbacks below, so there is nothing to return for TTS —
        the empty string yields no extra ``TextFrame``.
        """
        await self._qproc.handle(text, pid, _now_us())
        return ""

    async def on_query_superseded(self, pid: str) -> None:
        # A fresh query replaces the previous one: drain the prior answer's TTS
        # (and flush the hub buffer) so the new turn lands immediately.
        await self.push_frame(InterruptionFrame())

    async def on_participant_joined(self, pid: str) -> None:
        if self.perception is not None:
            self.perception.reset_observation_state()

    async def close(self) -> None:
        await self._http.aclose()

    # ── QueryProcessor output callbacks (pipeline-frame backed) ────────────────

    async def say(self, pid: str, text: str) -> None:
        """Speak *text* to *pid* through the shared streaming-TTS processor.

        Pushes the text downstream as a ``TextFrame`` then a
        ``BrainResponseEndFrame`` so the pending sentence is flushed and
        synthesized immediately — works both inside a query turn and from the
        background guidance/finalize tasks that speak out of turn.
        """
        if not text:
            return
        await self._push_text(text, pid=pid)
        # text="" — the per-turn data echo is disabled (text_topic="") since
        # QueryProcessor sends its own agent.response message; this end frame
        # only flushes the pending sentence into TTS.
        await self.push_frame(BrainResponseEndFrame(pid=pid, text="", pts_us=_now_us()))

    async def send_text(self, pid: str, text: str, topic: str) -> None:
        """Send a text data message to *pid* on *topic* (panel / progress)."""
        try:
            await self._transport.send_return_data(DataMessage(
                participant_id = pid,
                topic          = topic,
                pts_us         = _now_us(),
                data           = text.encode(),
            ))
        except Exception:
            log.exception("send_text failed  pid=%r  topic=%s", pid, topic)

    async def flush_audio(self, pid: str) -> None:
        """Barge-in: drain queued TTS and flush the hub return-audio buffer."""
        await self.push_frame(InterruptionFrame())

    # ── data side path (typed text, gate-bypassing) ────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        try:
            text = msg.data.decode(errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        pid = msg.participant_id or _DEFAULT_PID
        log.info("data query  pid=%r  %r", pid, text[:80])
        await self._spawn_query(GatedQueryFrame(
            participant_id = pid,
            text           = text,
            fresh_match    = False,
            pts_us         = msg.pts_us or _now_us(),
        ))

    # ── voice gate (noise + intent + guidance control) ─────────────────────────

    async def _passes_gate(self, pid: str, text: str) -> bool:
        """Two-layer noise/intent gate + guidance-control filter.

        Returns True if the utterance should reach the brain. Voice path only —
        data-channel text never gets here (it bypasses via ``_on_data``).
        """
        text = text.strip()
        if not text:
            return False

        # ── Layer 1: shape gate — cheap, synchronous. ─────────────────────
        if is_shape_noise(text):
            log.info("noise-gate L1 dropped  pid=%r  %r", pid, text[:80])
            _trace_log.info("NOISE_L1_DROP  pid=%s  %r", pid, text)
            return False

        if self._qproc.is_guiding(pid) and not self._qproc.is_guidance_control_utterance(text):
            log.info("guidance noise dropped  pid=%r  %r", pid, text[:80])
            _trace_log.info("GUIDANCE_NOISE_DROP  pid=%s  %r", pid, text)
            return False

        # ── Layer 2: LLM intent classifier. ───────────────────────────────
        # Skip when recording or guiding (those modes own their narration
        # policy) and when a fresh demo is on the table (post-demo utterances
        # are almost always guidance requests, even if mangled). Layer 2 is
        # fail-open: a classifier error accepts.
        skip_l2 = (
            self._memory.recording is not None
            or self._qproc.is_guiding(pid)
            or self._memory.demo_is_fresh(self._cfg.guidance_freshness_window_s)
        )
        if not skip_l2:
            try:
                accepted = await is_real_assistant_request(
                    self._http, self._cfg.llm_server, text
                )
            except Exception:
                log.exception("intent classifier raised — fail-open")
                accepted = True
            if not accepted:
                log.info("noise-gate L2 dropped  pid=%r  %r", pid, text[:80])
                _trace_log.info("NOISE_L2_DROP  pid=%s  %r", pid, text)
                return False

        return True


class GlassesPerception:
    """Always-on background scene perception, decoupled from voice I/O.

    Runs two loops over NAT MCP (video-mcp / vlm-mcp), independent of the
    pipecat audio pipeline:

      * ``_background_vlm_loop`` — every ``cfg.vlm_interval_s``: dense
        frame capture while recording a demonstration, otherwise
        change-detection scene observations into ``AgentMemory``;
      * ``_memory_condenser_loop`` — every ``cfg.condenser_interval_s``:
        condense recent observations into a scene summary.

    The loops query the live agent state through injected callables so this
    component never imports the brain: ``get_active_pid`` (who to observe),
    ``is_query_active`` (yield VLM bandwidth to user turns), and ``is_guiding``
    (skip observation while guidance owns the VLM checks).
    """

    _UNCHANGED_RESPONSES = frozenset({
        "unchanged", "no change", "nothing new", "same", "no changes",
        "nothing has changed", "nothing changed", "no significant change",
    })

    def __init__(
        self,
        *,
        cfg:            WorkerConfig,
        memory:         AgentMemory,
        transcript:     TranscriptClient,
        nat_runtime:    NatRuntime,
        get_active_pid: Callable[[], str],
        is_query_active: Callable[[], bool],
        is_guiding:     Callable[[str], bool],
    ) -> None:
        self._cfg             = cfg
        self._memory          = memory
        self._transcript      = transcript
        self._nat_runtime     = nat_runtime
        self._get_active_pid  = get_active_pid
        self._is_query_active = is_query_active
        self._is_guiding      = is_guiding

        self._bg_task:        asyncio.Task | None = None
        self._condenser_task: asyncio.Task | None = None

        # Last observed frame timestamp + description (observation loop).
        self._obs_last_ts:   int = 0
        self._obs_last_desc: str = ""
        # Last frame timestamp captured during recording (skip-dedup).
        self._rec_last_ts:    int = 0
        self._rec_warmup_end: int = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._bg_task = asyncio.create_task(
            self._background_vlm_loop(), name="glasses-bg-vlm"
        )
        self._condenser_task = asyncio.create_task(
            self._memory_condenser_loop(), name="glasses-condenser"
        )

    async def stop(self) -> None:
        for task in (self._bg_task, self._condenser_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    def reset_observation_state(self) -> None:
        """Reset the change-detection baseline so a (re)joined stream is
        processed fresh."""
        self._obs_last_ts   = 0
        self._obs_last_desc = ""

    # ── background VLM loop ───────────────────────────────────────────────────

    async def _background_vlm_loop(self) -> None:
        """Observe the scene every vlm_interval_s seconds.

        Recording — dense VLM capture, every frame saved to RecordedFrame buffer
                    and to a per-demo JSONL log on disk. No filtering or dedup —
                    analysis happens after recording stops.

        Normal    — change-detection VLM for the observation/memory timeline.
        """
        log.info("background VLM loop started  gap=%.1fs", self._cfg.vlm_interval_s)
        skipped:          int  = 0
        was_recording:    bool = False
        while True:
            try:
                # Run even with no participants: the client may have connected
                # before the worker saw the join. get_active_pid() falls back to
                # the default; get_latest_frame returns an error dict if no
                # stream is active, which is handled gracefully.
                pid          = self._get_active_pid()
                is_recording = self._memory.recording is not None

                if is_recording and not was_recording:
                    self._rec_last_ts    = 0
                    # Skip the configured warmup window while the wearer is
                    # still positioning before they start the first step.
                    self._rec_warmup_end = _now_us() + int(
                        self._cfg.recording_warmup_s * 1_000_000
                    )
                was_recording = is_recording

                # Recording capture is not gated by query-active — voice notes
                # count as part of the demonstration and must not create gaps in
                # the recorded-frame timeline.
                if is_recording:
                    if _now_us() < self._rec_warmup_end:
                        pass  # skip warmup window — user is still positioning
                    else:
                        frame = await self._capture_recording_frame(
                            pid, skip_ts=self._rec_last_ts
                        )
                        if frame:
                            self._rec_last_ts = frame.timestamp_us
                            self._memory.add_recorded_frame(frame)
                            _trace_log.info("REC_FRAME  %d  %s",
                                            frame.frame_idx + 1, frame.description[:80])
                elif not self._is_query_active() and not self._is_guiding(pid):
                    # Normal change-detection observations are bandwidth-
                    # sensitive — yield while user-facing guidance owns VLM checks.
                    obs = await self._observe_frame(pid, previous=self._obs_last_desc,
                                                    skip_ts=self._obs_last_ts)
                    if obs:
                        self._obs_last_desc = obs.description
                        self._obs_last_ts   = obs.timestamp_us
                        skipped             = 0
                        self._memory.add_observation(obs)
                        _trace_log.info("OBS  %s", obs.description)
                        log.info(
                            "\n  ┌─ observation ──────────────────────────────\n"
                            "  │ %s\n"
                            "  └────────────────────────────────────────────",
                            obs.description,
                        )
                        asyncio.create_task(
                            self._transcript.add_entry(
                                self._cfg.transcript_source + ":observations",
                                obs.timestamp_us,
                                obs.description,
                            )
                        )
                    else:
                        skipped += 1
                        if skipped % 30 == 0:
                            log.info("[vlm] observing — no change in last %ds", skipped)

                await asyncio.sleep(self._cfg.vlm_interval_s)
            except asyncio.CancelledError:
                log.info("background VLM loop cancelled")
                return
            except Exception:
                log.exception("background VLM loop error")

    async def _observe_frame(self, pid: str, previous: str = "",
                             skip_ts: int = 0) -> Observation | None:
        """Get the latest frame and caption what is NEW compared to *previous*."""
        frame_data = await self._call_video("get_latest_frame", {"participant_id": pid})
        if not isinstance(frame_data, dict) or "path" not in frame_data:
            return None
        frame_path = frame_data["path"]
        frame_ts   = int(frame_data.get("timestamp_us", _now_us()))

        if not frame_path or not os.path.isfile(frame_path):
            return None

        # Same frame as last time — hub hasn't delivered a new one yet.
        if skip_ts and frame_ts == skip_ts:
            return None

        if previous:
            prev_clean = previous.split("compared to")[0].strip().rstrip(",")
            question = (
                f"Previous observation: {prev_clean}\n\n"
                "Compare this new frame to the previous observation. "
                "Has something visibly changed — a new action, movement, or object state?\n"
                "If yes: one short phrase describing the change. "
                "Include WHERE in the frame (left/right/center) and position "
                "relative to nearby objects if relevant.\n"
                "If nothing changed: respond with exactly: unchanged"
            )
        else:
            question = (
                "In one short phrase, describe the most notable action or movement "
                "happening in this frame. Include spatial position if relevant "
                "(left/right/center of frame, above/below/beside what object)."
            )

        description = await self._call_vlm(
            "ask_image", {"question": question, "image_path": frame_path}
        )
        if isinstance(description, dict):
            description = (description.get("result") or description.get("text")
                          or next(iter(description.values()), ""))
        if not isinstance(description, str) or not description.strip():
            return None
        desc = description.strip()
        if desc.lower().rstrip(".!") in self._UNCHANGED_RESPONSES:
            return None
        # Dedup: if the model regenerated the same text as before, skip.
        if previous and desc == previous:
            return None

        return Observation(
            timestamp_us = frame_ts,
            description  = description.strip(),
            image_path   = frame_path,
        )

    async def _capture_recording_frame(
        self, pid: str, skip_ts: int = 0
    ) -> RecordedFrame | None:
        """Capture one frame's detailed VLM analysis during recording.

        Returns a RecordedFrame with a rich 2-4 sentence description, or None
        if no new frame is available. Also appends to the per-demo JSONL log.
        No dedup, no filtering — everything is captured; analysis happens later.
        """
        frame_data = await self._call_video("get_latest_frame", {"participant_id": pid})
        if not isinstance(frame_data, dict) or "path" not in frame_data:
            _trace_log.info("REC_NO_FRAME  %s", str(frame_data)[:120] if frame_data else "None")
            return None
        frame_path = frame_data["path"]
        frame_ts   = int(frame_data.get("timestamp_us", _now_us()))
        if not frame_path or not os.path.isfile(frame_path):
            _trace_log.info("REC_NO_FILE  path=%s", frame_path)
            return None
        if skip_ts and frame_ts == skip_ts:
            return None  # same frame as last cycle — normal, no log needed

        # Get recording context before the VLM call (race-safe: check again after).
        recording = self._memory.recording
        if recording is None:
            return None

        # Copy frame to a stable path so it persists after get_latest_frame
        # overwrites the temp file on the next cycle.
        frame_idx_now = len(recording.recorded_frames)
        run_dir_now   = os.environ.get("XR_RUN_DIR", "/tmp")
        safe_now      = _re.sub(r"[^a-zA-Z0-9_-]", "_", recording.name)[:40]
        frames_dir    = os.path.join(run_dir_now, "recordings",
                                     f"{safe_now}_{recording.started_at_us}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        stable_path = os.path.join(frames_dir, f"frame_{frame_idx_now:04d}.png")
        try:
            _shutil.copy2(frame_path, stable_path)
            frame_path = stable_path
        except Exception as exc:
            log.warning("frame copy failed: %s", exc)

        # Build context from demo name + most recent voice note.
        ctx = f"The user is recording a demonstration of: {recording.name!r}."
        last_note = recording.voice_notes[-1].text if recording.voice_notes else ""
        if last_note:
            ctx += f'\n They just said: "{last_note}".'

        question = (
            f"{ctx}\n\n"
            "Describe ONLY what you observe in this image right now. "
            "Do not predict, suggest, or describe what should happen next.\n"
            "Include:\n"
            "- Which hand(s) are active and the exact action being performed\n"
            "- What object is being held or touched (describe by color, shape, size "
            "if its name is uncertain)\n"
            "- Precise spatial position: left/right/center, above/below/beside which "
            "other visible object\n"
            "2-3 factual sentences. Describe the current moment only."
        )
        description = await self._call_vlm(
            "ask_image", {"question": question, "image_path": frame_path}
        )

        # ask_image returns a str, but the NAT runtime may JSON-decode it into
        # {"result": "..."} if the content happens to be valid JSON.
        if isinstance(description, dict):
            description = (description.get("result") or description.get("text")
                          or next(iter(description.values()), ""))
        if not isinstance(description, str) or not description.strip():
            _trace_log.info("REC_NO_DESC  vlm returned empty  raw=%s", str(description)[:80])
            return None

        # Re-check in case recording stopped while the VLM call was in flight.
        if self._memory.recording is None:
            _trace_log.info("REC_RACE  recording stopped mid-capture")
            return None

        frame_idx = len(recording.recorded_frames)
        desc      = description.strip()

        # Write to per-demo JSONL log (one JSON per line, crash-safe).
        run_dir  = os.environ.get("XR_RUN_DIR", "/tmp")
        safe     = _re.sub(r"[^a-zA-Z0-9_-]", "_", recording.name)[:40]
        log_dir  = os.path.join(run_dir, "recordings")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{safe}_{recording.started_at_us}.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(_json.dumps({
                    "frame_idx":  frame_idx,
                    "ts_us":      frame_ts,
                    "image_path": frame_path,
                    "description": desc,
                }) + "\n")
        except Exception as exc:
            log.warning("recording log write failed: %s", exc)

        return RecordedFrame(
            frame_idx    = frame_idx,
            timestamp_us = frame_ts,
            image_path   = frame_path,
            description  = desc,
        )

    async def _call_vlm(self, tool: str, args: dict) -> dict | str | None:
        """Call a vlm-mcp tool through NAT."""
        try:
            return await self._nat_runtime.call_tool("vlm_mcp", tool, args)
        except Exception as exc:
            log.error("vlm-mcp %s failed: %s", tool, exc)
            return None

    async def _call_video(self, tool: str, args: dict) -> dict | str | None:
        """Call a video-mcp tool through NAT."""
        try:
            return await self._nat_runtime.call_tool("video_mcp", tool, args)
        except Exception as exc:
            log.error("video-mcp %s failed: %s", tool, exc)
            return None

    # ── memory condenser loop ─────────────────────────────────────────────────

    async def _memory_condenser_loop(self) -> None:
        """Every cfg.condenser_interval_s seconds, condense observations into a summary."""
        log.info("memory condenser started  interval=%.1fs", self._cfg.condenser_interval_s)
        while True:
            try:
                await asyncio.sleep(self._cfg.condenser_interval_s)
                await self._condense_observations()
            except asyncio.CancelledError:
                log.info("memory condenser cancelled")
                return
            except Exception:
                log.exception("memory condenser error")

    async def _condense_observations(self) -> None:
        """Condense the last 20 observations through the NAT worker task group."""
        recent = list(self._memory._observations)[-20:]
        if not recent:
            return

        try:
            structured = await asyncio.wait_for(
                self._nat_runtime.call_tool(
                    "glasses_worker_tasks",
                    "condense_observations",
                    {
                        "observations": [
                            {
                                "timestamp_us": o.timestamp_us,
                                "description": o.description,
                            }
                            for o in recent
                        ]
                    },
                ),
                timeout=20.0,
            )
            if not isinstance(structured, dict):
                return

            overview = structured.get("overview", "").strip()
            events   = structured.get("events", [])
            summary_text = structured.get("summary_text", "").strip()

            if not overview and not summary_text:
                return

            if not summary_text:
                lines = [overview]
                for ev in events:
                    ts_us = ev.get("timestamp_us", 0)
                    hms   = ev.get("time", "")
                    desc  = ev.get("description", "")
                    lines.append(f"  [{hms} | {ts_us} us] {desc}")
                summary_text = "\n".join(lines)

            self._memory.update_scene_summary(summary_text)
            _trace_log.info("CONDENSER  %s", summary_text.replace("\n", " | "))

            # Persist the full structured JSON so video-mcp can be queried by timestamp.
            ts = _now_us()
            asyncio.create_task(
                self._transcript.add_entry(
                    self._cfg.transcript_source + ":scene_summary",
                    ts,
                    _json.dumps(structured),
                )
            )

            log.info("[scene summary]\n%s", summary_text)
        except Exception:
            log.exception("condenser LLM call failed")
