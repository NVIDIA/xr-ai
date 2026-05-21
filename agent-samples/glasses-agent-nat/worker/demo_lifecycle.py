# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo recording lifecycle: start, end, post-recording analysis.

The controller owns the speak/send-text turn for each lifecycle event and
delegates analysis to the NAT worker task group.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from memory import AgentMemory, Demonstration, DemoStep
from nat_runtime import NatRuntime

log = logging.getLogger("glasses_agent_nat.demo_lifecycle")
_trace_log = logging.getLogger("glasses_agent_nat.trace")

SendTextCb = Callable[[str, str, str], Awaitable[None]]
SayCb = Callable[[str, str], Awaitable[None]]

_AGENT_RESPONSE_TOPIC = "agent.response"
_AGENT_PROGRESS_TOPIC = "agent.progress"


class DemoController:
    """Owns demo start / end / analyze flows."""

    def __init__(
        self,
        memory: AgentMemory,
        nat_runtime: NatRuntime,
        *,
        send_text: SendTextCb,
        say: SayCb,
    ) -> None:
        self._memory = memory
        self._nat_runtime = nat_runtime
        self._send_text = send_text
        self._say = say

    async def handle_start(self, name: str, pid: str) -> None:
        self._memory.start_recording(name)
        response = (
            f"Recording '{name}'. Go ahead and demonstrate — "
            "I'm watching your hands and will capture each step."
        )
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        _trace_log.info("DEMO_START  name=%s", name)

    async def handle_end(self, pid: str, ref_us: int) -> None:
        demo = self._memory.finish_recording()
        if demo is None:
            response = "No demonstration was being recorded."
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return
        if not demo.recorded_frames:
            response = (
                "Recording stopped but no camera frames were captured — "
                "is the camera on? Start the camera then try recording again."
            )
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return

        n_frames = len(demo.recorded_frames)
        _trace_log.info("DEMO_END  name=%s  frames=%d", demo.name, n_frames)

        await self._send_text(
            pid, f"Got it — {n_frames} frames captured. Analyzing now…",
            _AGENT_PROGRESS_TOPIC,
        )
        await self._say(pid, "Got it. Analyzing the recording now.")

        # Follow-up ping after 10 s so the user knows it's still working.
        async def _ping():
            await asyncio.sleep(10)
            await self._send_text(
                pid, "Still analyzing — reviewing key frames…",
                _AGENT_PROGRESS_TOPIC,
            )
        ping = asyncio.create_task(_ping())

        try:
            overview, instructions = await self._analyze_recording(demo)
        finally:
            ping.cancel()

        if not instructions:
            import os as _os, re as _re
            run_dir  = _os.environ.get("XR_RUN_DIR", "/tmp")
            safe     = _re.sub(r"[^a-zA-Z0-9_-]", "_", demo.name)[:40]
            log_path = f"{run_dir}/recordings/{safe}_{demo.started_at_us}.jsonl"
            _trace_log.info("DEMO_ANALYSIS_FAILED  log=%s", log_path)
            response = (
                f"Recorded {n_frames} frames but analysis failed. "
                f"Raw observations saved to: {log_path}"
            )
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, "Recording saved but analysis failed.")
            return

        demo.summary      = overview
        demo.instructions = instructions

        # Each step's reference frame is the COMPLETED state — the frame just
        # before the next voice note. Using the before-state confuses the
        # vlm-check (it sees pre-step state and says NO even when done).
        n       = len(instructions)
        notes   = demo.voice_notes
        frames  = demo.recorded_frames

        def _after_frame(step_idx: int) -> str:
            if not frames:
                return ""
            next_note_ts = None
            if step_idx + 1 < len(notes):
                next_note_ts = notes[step_idx + 1].timestamp_us
            if next_note_ts is not None:
                before = [f for f in frames if f.timestamp_us < next_note_ts]
                return before[-1].image_path if before else frames[-1].image_path
            return frames[-1].image_path

        span = max(demo.ended_at_us - demo.started_at_us, 1)
        for i, instr in enumerate(instructions):
            ts = demo.started_at_us + int(i / max(n - 1, 1) * span)
            demo.steps.append(DemoStep(
                step_number  = i + 1,
                timestamp_us = ts,
                description  = instr,
                image_path   = _after_frame(i),
            ))

        if demo.recorded_frames:
            _trace_log.info("STEP_FRAMES  %s",
                            " | ".join(
                                f"{s.step_number}:{s.image_path[-20:]}"
                                for s in demo.steps if s.image_path
                            ))

        _trace_log.info("DEMO_INSTRUCTIONS  %s",
                        " | ".join(f"{i+1}:{s[:40]}" for i, s in enumerate(instructions)))

        response = f"Saved '{demo.name}' with {n} steps."
        if overview:
            response += f" {overview}"
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, f"Saved demonstration '{demo.name}' with {n} steps.")

    async def _analyze_recording(
        self, demo: Demonstration
    ) -> tuple[str, list[str]]:
        """Run NAT analyze_recording task and unpack (overview, steps)."""
        frames = demo.recorded_frames
        if not frames:
            return "", []

        _trace_log.info("ANALYSIS_START  demo=%r  frames=%d  voice_notes=%d",
                        demo.name, len(frames), len(demo.voice_notes))
        try:
            result = await self._nat_runtime.call_tool("glasses_worker_tasks", "analyze_recording", {
                "name": demo.name,
                "started_at_us": demo.started_at_us,
                "frames": [
                    {
                        "frame_idx": f.frame_idx,
                        "timestamp_us": f.timestamp_us,
                        "image_path": f.image_path,
                        "description": f.description,
                    }
                    for f in frames
                ],
                "voice_notes": [
                    {"timestamp_us": v.timestamp_us, "text": v.text}
                    for v in demo.voice_notes
                ],
            })
        except Exception:
            log.exception("analysis NAT task failed")
            return "", []
        if not isinstance(result, dict):
            return "", []
        overview = str(result.get("overview", "")).strip()
        steps = [str(s).strip() for s in result.get("steps", []) if str(s).strip()]
        _trace_log.info("ANALYSIS_RESULT  %s", str(result)[:300])
        return overview, steps
