# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SimpleVlmBrain — vision Q&A on the unified pipecat pipeline.

Behaviour is identical to the original sample; the difference is that the
live-camera machinery (frame tracking, camera-on-demand, the streaming VLM
call) is no longer re-implemented here — it lives in the shared, reusable
:class:`xr_ai_capabilities.VisionModule`. This brain is thin glue: it routes a query
to the module, owns the data-channel side path, and interrupts on supersede.
"""
from __future__ import annotations

from typing import AsyncIterator

from loguru import logger
from pipecat.frames.frames import InterruptionFrame
from xr_ai_agent import DataMessage
from xr_ai_models import VLMService
from xr_ai_capabilities import VisionModule
from xr_ai_pipecat import BrainProcessor, GatedQueryFrame
from xr_ai_pipecat.transport import XRMediaHubTransport


DEFAULT_SYSTEM_PROMPT = (
    "You are an XR assistant. You can see the user's live camera feed, "
    "but you are not required to use it. Decide per question:\n"
    "- If the question is about what is visible (e.g. 'what am I looking "
    "at', 'what does this say', 'is the door open', 'describe this', "
    "'what color is the X'), answer from the image.\n"
    "- If the question is general knowledge, a definition, a calculation, "
    "a chat, or anything not tied to the scene (e.g. 'what's the capital "
    "of France', 'tell me a joke', 'explain entropy', 'how do I boil "
    "pasta'), answer like a normal assistant and ignore the image.\n"
    "- When it is ambiguous, prefer the visual answer if the camera shows "
    "something obviously relevant; otherwise answer generally.\n"
    "\n"
    "Style:\n"
    "- Speak directly to me in second person where natural: 'You are looking "
    "at…', 'I can see…'. Never refer to 'the user' in the third person.\n"
    "- Reply in plain conversational English — never JSON, code, or markdown.\n"
    "- Keep replies to 10-15 words by default. Only go longer when I "
    "explicitly ask for detail (e.g. 'describe in detail', 'tell me more', "
    "'elaborate', 'explain').\n"
    "- If I say 'stop', ask you to be quiet, or ask you to stop "
    "talking, just acknowledge briefly with something like 'Okay, I will stop.' "
    "and say nothing else."
)


class SimpleVlmBrain(BrainProcessor):
    """Camera + VLM brain, built on the shared :class:`VisionModule`."""

    def __init__(
        self,
        *,
        transport: XRMediaHubTransport,
        vlm: VLMService,
        default_prompt:      str   = "Describe what you see.",
        system_prompt:       str   = DEFAULT_SYSTEM_PROMPT,
        frame_max_age_s:     float = 2.0,
        camera_on_timeout_s: float = 15.0,
        camera_grace_s:      float = 5.0,
    ) -> None:
        super().__init__()
        self._transport = transport
        self._default_prompt = default_prompt

        # All the live-camera machinery lives in the shared module.
        self._vision = VisionModule(
            transport.endpoint, vlm,
            system_prompt       = system_prompt,
            frame_max_age_s     = frame_max_age_s,
            camera_on_timeout_s = camera_on_timeout_s,
            camera_grace_s      = camera_grace_s,
        )
        self._vision.register()

        # Data-channel side path (typed queries). Participant-leave teardown
        # rides the base BrainProcessor frame path → on_participant_left.
        transport.endpoint.on_data(self._on_data)

    # ── BrainProcessor overrides ──────────────────────────────────────────────

    async def handle_query(
        self, pid: str, text: str, fresh_match: bool,
    ) -> AsyncIterator[str]:
        # Return (not yield) the async iterator — the base awaits then iterates.
        return self._vision.ask(pid, text)

    async def on_user_started_speaking(self, pid: str) -> None:
        # Speculative camera warmup at the leading edge of speech.
        await self._vision.warmup(pid)

    async def on_query_superseded(self, pid: str) -> None:
        # Vision Q&A turns are short; cut the previous answer's audio so the new
        # one lands immediately (library default is queue-behind).
        await self.push_frame(InterruptionFrame())

    async def on_participant_left(self, pid: str) -> None:
        self._vision.release(pid)

    # ── data-channel side path ────────────────────────────────────────────────

    async def _on_data(self, msg: DataMessage) -> None:
        try:
            text = msg.data.decode(errors="replace").strip()
        except Exception:
            return
        if not text:
            return
        logger.info("data query  pid={!r}  {!r}", msg.participant_id, text[:80])
        query = self._default_prompt if text.lower() == "ping" else text
        await self._spawn_query(GatedQueryFrame(
            participant_id = msg.participant_id,
            text           = query,
            fresh_match    = True,
            pts_us         = msg.pts_us,
        ))
