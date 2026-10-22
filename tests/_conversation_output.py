# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture destination-owned output while exercising sample routing policy."""

from contextvars import ContextVar
from typing import Any

from xr_ai_routing import AgentRouter, AgentTurn
from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
from xr_ai_voice import VOICE_OUTPUT_TOPIC, VoiceOutput

_output = ContextVar("conversation_test_output", default=None)


async def capture_emit(source, text):
    emit = _output.get()
    if emit is not None:
        await emit(source, text)


async def capture_router_output(_turn: AgentTurn, source: str, text: str) -> None:
    await capture_emit(source, text)


class CapturingAgentRouter(AgentRouter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("output", capture_router_output)
        super().__init__(*args, **kwargs)


async def standalone_started() -> bool:
    return True


async def answer(destination, turn, *, emit):
    token = _output.set(emit)
    try:
        return await destination.answer(turn, started=standalone_started)
    finally:
        _output.reset(token)


async def handle(router, turn, *, emit):
    token = _output.set(emit)
    try:
        return await router.handle(turn)
    finally:
        _output.reset(token)


class _OutputCapture(Agent):
    @subscribe(VOICE_OUTPUT_TOPIC)
    async def receive(self, message: VoiceOutput, ctx: RuntimeContext) -> None:
        if message.text:
            await capture_emit(ctx.metadata.source, message.text)


class CaptureRuntime(AgentRuntime):
    """Use typed output delivery while routing unit-test text into a local callback."""

    def __init__(self) -> None:
        super().__init__()
        self.register("test-output-capture", _OutputCapture())

    async def publish(self, topic, message, *, participant_id=None, source="application"):
        if not self.running:
            await self.start()
        await super().publish(topic, message, participant_id=participant_id, source=source)
