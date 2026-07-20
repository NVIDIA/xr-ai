# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SimpleVlmBrain — vision Q&A on the unified pipecat pipeline.

The brain is only the Pipecat boundary: live-camera acquisition and VLM
streaming are owned by the injected native NAT function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import AsyncIterator

from loguru import logger
from pipecat.frames.frames import InterruptionFrame
from xr_ai_agent import DataMessage
from nat.builder.function import Function
from xr_ai_pipecat import BrainProcessor, GatedQueryFrame
from xr_ai_pipecat.transport import XRMediaHubTransport
from xr_ai_nat.functions.vision import LiveVisionRequest


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
    """Adapt one native streaming vision function to the voice pipeline."""

    def __init__(
        self,
        *,
        transport: XRMediaHubTransport,
        vision: Function,
        release_vision: Callable[[str], None],
        default_prompt: str = "Describe what you see.",
    ) -> None:
        super().__init__()
        self._transport = transport
        self._default_prompt = default_prompt
        self._vision = vision
        self._release_vision = release_vision

        # Data-channel side path (typed queries). Participant-leave teardown
        # rides the base BrainProcessor frame path → on_participant_left.
        transport.endpoint.on_data(self._on_data)

    # ── BrainProcessor overrides ──────────────────────────────────────────────

    async def handle_query(
        self,
        pid: str,
        text: str,
        fresh_match: bool,
    ) -> AsyncIterator[str]:
        del fresh_match

        async def tokens() -> AsyncIterator[str]:
            request = LiveVisionRequest(participant_id=pid, question=text)
            async for chunk in self._vision.astream(request):
                yield chunk.text

        return tokens()

    async def on_user_started_speaking(self, pid: str) -> None:
        pass

    async def on_query_superseded(self, pid: str) -> None:
        # Vision Q&A turns are short; cut the previous answer's audio so the new
        # one lands immediately (library default is queue-behind).
        await self.push_frame(InterruptionFrame())

    async def on_participant_left(self, pid: str) -> None:
        self._release_vision(pid)

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
        await self._spawn_query(
            GatedQueryFrame(
                participant_id=msg.participant_id,
                text=query,
                fresh_match=True,
                pts_us=msg.pts_us,
            )
        )
