# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guidance mode controller: request, advance, finish, question handling,
plus the background monitor that auto-advances when a step looks done.

Owns all guidance-related state so it can't drift between the routing layer
and the monitor task.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from xr_ai_models import ChatMessage, LLMService

from config import WorkerConfig
from memory import AgentMemory, Demonstration
from nat_runtime import NatRuntime
from mcp_shims import call_vlm, get_latest_frame_path

log = logging.getLogger("glasses_agent_nat.guidance")
_trace_log = logging.getLogger("glasses_agent_nat.trace")

SendTextCb = Callable[[str, str, str], Awaitable[None]]
SayCb = Callable[[str, str], Awaitable[None]]

_AGENT_RESPONSE_TOPIC = "agent.response"


class GuidanceController:
    """Owns guidance session state + the auto-advance monitor task."""

    def __init__(
        self,
        cfg: WorkerConfig,
        memory: AgentMemory,
        nat_runtime: NatRuntime,
        *,
        worker_llm: LLMService,
        send_text: SendTextCb,
        say: SayCb,
    ) -> None:
        self._cfg = cfg
        self._memory = memory
        self._nat_runtime = nat_runtime
        self._send_text = send_text
        self._say = say
        # Owned by QueryProcessor; we just call it.
        self._worker_llm = worker_llm

        self._demo: Demonstration | None = None
        self._step: int = 0
        self._monitor_task: asyncio.Task | None = None
        self._advancing: bool = False
        self._step_obs_baseline: int = 0
        self._monitor_idle_cycles: int = 0
        self._consecutive_yes: int = 0  # consecutive 2-frame YES responses

    # ── state read by processors._agentic_loop ────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._demo is not None

    def context_block(self) -> str | None:
        """Return the guidance context block for the agentic prompt, or None."""
        if self._demo is None:
            return None
        total       = len(self._demo.steps)
        step_num    = self._step + 1
        instruction = self._instruction_for_step(self._demo)
        return (
            f"[Guidance mode — step {step_num} of {total}]\n"
            f"Current instruction: {instruction}\n"
            f"Help the user complete THIS step. Be concise and direct."
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def handle_request(self, query: str, pid: str) -> None:
        """Find a matching demo and enter guidance mode."""
        demo = self._memory.find_demonstration_fuzzy(query)
        if demo is None:
            demos = self._memory.list_demonstrations()
            if demos:
                demo = self._memory.get_demonstration(demos[0])
        if demo is None:
            response = "I don't have any recorded demonstrations yet. Show me first!"
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return

        if not demo.steps:
            response = f"The demonstration '{demo.name}' has no steps recorded."
            await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, response)
            return

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            self._monitor_task = None
        self._advancing = False

        self._demo = demo
        self._step = 0
        _trace_log.info("GUIDANCE_START  demo=%s  steps=%d", demo.name, len(demo.steps))
        await self._speak_current_step(pid)
        self._start_monitor(pid)

    async def advance(self, pid: str) -> None:
        if self._demo is None or self._advancing:
            return
        self._advancing = True
        try:
            self._step += 1
            if self._step >= len(self._demo.steps):
                await self.finish(pid)
            else:
                await self._speak_current_step(pid)
        finally:
            self._advancing = False

    async def finish(self, pid: str) -> None:
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            self._monitor_task = None
        demo = self._demo
        self._demo = None
        self._step = 0
        if demo:
            response = f"You've completed all steps in '{demo.name}'. Well done!"
        else:
            response = "Guidance complete."
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        _trace_log.info("GUIDANCE_DONE")

    async def handle_question(self, transcript: str, pid: str) -> None:
        """Answer a question during guidance. Pre-fetches the camera frame so
        questions like 'where is the knife?' get a real VLM-based answer."""
        if self._demo is None:
            return
        instruction = self._instruction_for_step(self._demo)
        step_num    = self._step + 1
        total       = len(self._demo.steps)

        lower = transcript.lower()
        if any(p in lower for p in ("doing it right", "doing step", "correct", "done", "finished")):
            result = await self._completion_result(pid)
            raw = str(result.get("raw", "")) if isinstance(result, dict) else ""
            completed = bool(result.get("completed")) if isinstance(result, dict) else False
            if completed:
                reply = f"Yes, step {step_num} looks complete. {instruction}"
            else:
                detail = f" I see: {raw}" if raw and raw.upper() != "NO" else ""
                reply = f"Not yet. Step {step_num} is: {instruction}.{detail}"
            await self._send_text(pid, reply, _AGENT_RESPONSE_TOPIC)
            await self._say(pid, reply)
            return

        # Reference-frame comparison conflated visual intent (user near target)
        # with completion. Ask the VLM what it currently sees and let the LLM judge.
        frame_path = await get_latest_frame_path(self._nat_runtime, pid, ref_us=0)

        system = (
            f"You are guiding someone through a procedure. "
            f"They are on step {step_num} of {total}: \"{instruction}\". "
            "Answer their question honestly in 1-2 short sentences based on "
            "what the camera currently shows. Be direct — if it is not done "
            "correctly, say so clearly."
        )
        user_text = transcript
        if frame_path:
            try:
                result = await asyncio.wait_for(
                    call_vlm(
                        self._nat_runtime,
                        "ask_image",
                        {"question": (
                            f"Step to complete: \"{instruction}\"\n"
                            f"User asks: {transcript}\n"
                            "Describe exactly what you see in the current frame "
                            "relevant to this step."
                        ),
                         "image_path": frame_path},
                        silent=True,
                    ),
                    timeout=8.0,
                )
                if isinstance(result, dict):
                    result = (result.get("result") or result.get("text")
                              or next(iter(result.values()), ""))
                if isinstance(result, str) and result.strip():
                    user_text = (
                        f"What the camera sees: {result.strip()}\n\n"
                        f"User question: {transcript}"
                    )
            except (asyncio.TimeoutError, Exception):
                pass

        try:
            resp = await asyncio.wait_for(
                self._worker_llm.chat(
                    [
                        ChatMessage(role="system", content=system),
                        ChatMessage(role="user",   content=user_text),
                    ],
                    max_tokens=100,
                    temperature=0.1,
                ),
                timeout=10.0,
            )
            reply = (resp.content or "").strip()
            if reply:
                await self._send_text(pid, reply, _AGENT_RESPONSE_TOPIC)
                await self._say(pid, reply)
        except Exception:
            log.exception("guidance question failed")

    # ── per-step helpers ──────────────────────────────────────────────────────

    def _instruction_for_step(self, demo: Demonstration) -> str:
        """Clean instruction for the current step, falling back to raw description."""
        idx = self._step
        if demo.instructions and idx < len(demo.instructions):
            return demo.instructions[idx]
        return demo.steps[idx].description

    async def _speak_current_step(self, pid: str) -> None:
        demo = self._demo
        if demo is None:
            return
        self._step_obs_baseline   = len(self._memory._observations)
        self._monitor_idle_cycles = 0
        self._consecutive_yes     = 0
        total       = len(demo.steps)
        step_num    = self._step + 1
        instruction = self._instruction_for_step(demo)
        response = f"Step {step_num} of {total}: {instruction}"
        await self._send_text(pid, response, _AGENT_RESPONSE_TOPIC)
        await self._say(pid, response)
        _trace_log.info("GUIDANCE_STEP  %d/%d  %s", step_num, total, instruction[:60])

    # ── auto-advance monitor ──────────────────────────────────────────────────

    def _start_monitor(self, pid: str) -> None:
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(pid), name="guidance-monitor"
        )

    async def _monitor_loop(self, pid: str) -> None:
        """Advance guidance automatically when the user completes a step.

        Primary signal: the background VLM observation loop adds a new entry
        when something visibly changes — obs_delta > 0 means the user did something.

        Fallback (fine-motor actions): after enough idle cycles with no
        observation change, do a direct VLM check asking whether the step
        looks complete. This catches button presses / switch flips that don't
        produce large enough scene changes for the observation loop.
        """
        _trace_log.info("GUIDANCE_MONITOR  start  step=%d", self._step)
        try:
            while self._demo is not None:
                await asyncio.sleep(self._cfg.guidance_check_interval_s)
                if self._demo is None:
                    break

                current_obs = len(self._memory._observations)
                delta = current_obs - self._step_obs_baseline
                _trace_log.info(
                    "GUIDANCE_MONITOR  step=%d  obs_delta=%d  idle=%d",
                    self._step, delta, self._monitor_idle_cycles,
                )

                should_check = delta > 0
                if not should_check:
                    self._monitor_idle_cycles += 1
                    should_check = self._monitor_idle_cycles >= 1
                if not should_check:
                    continue

                self._monitor_idle_cycles = 0
                if await self._vlm_step_complete(pid):
                    self._consecutive_yes += 1
                    _trace_log.info(
                        "GUIDANCE_MONITOR  yes-count=%d  step=%d",
                        self._consecutive_yes, self._step,
                    )
                    if self._consecutive_yes >= 2:
                        self._consecutive_yes = 0
                        _trace_log.info("GUIDANCE_MONITOR  vlm-advance  step=%d",
                                        self._step)
                        await self.advance(pid)
                else:
                    self._consecutive_yes = 0
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("guidance monitor error")
        _trace_log.info("GUIDANCE_MONITOR  exit")

    async def _completion_result(self, pid: str) -> dict:
        if self._demo is None:
            return {}
        instruction = self._instruction_for_step(self._demo)
        try:
            result = await asyncio.wait_for(
                self._nat_runtime.call_tool(
                    "glasses_worker_tasks",
                    "check_guidance_step_complete",
                    {"participant_id": pid, "instruction": instruction},
                    participant_id=pid,
                ),
                timeout=10.0,
            )
        except (asyncio.TimeoutError, Exception):
            result = {}
        return result if isinstance(result, dict) else {}

    async def _vlm_step_complete(self, pid: str) -> bool:
        """Ask the NAT guidance task whether the current step looks done."""
        result = await self._completion_result(pid)
        raw = str(result.get("raw", "")) if isinstance(result, dict) else ""
        completed = bool(result.get("completed")) if isinstance(result, dict) else False
        instruction = self._instruction_for_step(self._demo) if self._demo else ""
        _trace_log.info("GUIDANCE_MONITOR  vlm-check  %r → %s  raw=%r",
                        instruction[:40], "YES" if completed else "NO", raw[:30])
        return completed

    # ── shutdown ──────────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        # ``_worker_llm`` is owned by QueryProcessor — do not close it here.
