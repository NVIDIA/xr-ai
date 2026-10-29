"""A tool-free live voice sample composed exclusively from public SDK APIs."""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import replace

from loguru import logger
from xr_ai_interaction import Interrupted, ParticipantLeft, ResponseOutput, UserQuery
from xr_ai_models import (
    ChatMessage,
    ModelsConfig,
    load_models_config,
    make_decision,
    make_llm,
    make_tts,
)
from xr_ai_runtime import Agent, AgentRuntime, Topic, subscribe
from xr_ai_voice import VOICE_OUTPUT_TOPIC, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

from ._turns import _Policy
from ._voice_policy import _ClefVoicePolicy

_QUERY = Topic("semantic-prototype.query", UserQuery)
_INTERRUPTED = Topic("semantic-prototype.interrupted", Interrupted)
_LEFT = Topic("semantic-prototype.left", ParticipantLeft)


class _Assistant(Agent):
    def __init__(self, llm):
        super().__init__()
        self._llm = llm
        self._tasks = {}
        self._history = {}
        self._lock = asyncio.Lock()

    async def _cancel(self, pid):
        task = self._tasks.pop(pid, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @subscribe(_QUERY)
    async def query(self, query: UserQuery, ctx):
        pid = ctx.metadata.participant_id
        async with self._lock:
            await self._cancel(pid)
            self._tasks[pid] = asyncio.create_task(self._respond(pid, query, ctx))

    @subscribe(_INTERRUPTED)
    async def interrupted(self, event, ctx):
        async with self._lock:
            for pid in (
                [ctx.metadata.participant_id] if ctx.metadata.participant_id else tuple(self._tasks)
            ):
                await self._cancel(pid)

    @subscribe(_LEFT)
    async def left(self, event, ctx):
        async with self._lock:
            await self._cancel(ctx.metadata.participant_id)
            self._history.pop(ctx.metadata.participant_id, None)

    async def _respond(self, pid, query, ctx):
        request_id = query.request_id or ctx.metadata.message_id
        history = self._history.setdefault(pid, [])
        history.append(ChatMessage("user", query.text))
        del history[:-8]
        messages = [
            ChatMessage("system", "Answer briefly in natural spoken language. You have no tools."),
            *history,
        ]
        stream = self._llm.stream(messages, enable_thinking=False, max_tokens=256)
        text = []
        try:
            async for chunk in stream:
                text.append(chunk)
                await ctx.publish(
                    VOICE_OUTPUT_TOPIC,
                    ResponseOutput(
                        kind="reply",
                        request_id=request_id,
                        response_id=request_id,
                        text=chunk,
                        final=False,
                    ),
                )
            if not text:
                await ctx.publish(
                    VOICE_OUTPUT_TOPIC,
                    ResponseOutput(
                        kind="reply",
                        request_id=request_id,
                        response_id=request_id,
                        text="I did not get a usable answer. Please try again.",
                        final=False,
                    ),
                )
            await ctx.publish(
                VOICE_OUTPUT_TOPIC,
                ResponseOutput(
                    kind="reply",
                    request_id=request_id,
                    response_id=request_id,
                    final=True,
                ),
            )
            if text:
                history.append(ChatMessage("assistant", "".join(text)))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - owned policy/task supervision
            # isolation boundary: report the owned generation task failure.
            logger.exception("prototype response failed pid={!r}", pid)
            await ctx.publish(
                VOICE_OUTPUT_TOPIC,
                ResponseOutput(
                    kind="reply",
                    request_id=request_id,
                    response_id=request_id,
                    text="The response failed. Please try again.",
                    final=True,
                ),
            )
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
            if self._tasks.get(pid) is asyncio.current_task():
                self._tasks.pop(pid, None)

    async def close(self):
        async with self._lock:
            for pid in tuple(self._tasks):
                await self._cancel(pid)


async def run_live(args):
    """Use existing hub/model services; no downloads or model server startup."""
    if args.models is None or args.voice_gate is None:
        raise ValueError("--live requires --models and --voice-gate")
    models = load_models_config(args.models)
    gate = load_voice_gate_config(args.voice_gate)
    policy_cfg = _Policy(**json.loads(args.policy.read_text())) if args.policy else _Policy()
    async with AsyncExitStack() as stack:
        spec = models.llm("llm")
        extras = dict(spec.adapter.default_extras)
        # Shared sample profiles can pin a large token limit in request extras.
        # Let this voice worker's short per-call limit take effect.
        extras.pop("max_tokens", None)
        voice_models = ModelsConfig(
            {
                **models.entries,
                "llm": replace(spec, adapter=replace(spec.adapter, default_extras=extras)),
            }
        )
        llm = make_llm(voice_models, "llm")
        stack.push_async_callback(llm.close)
        decisions = make_decision(models, "router")
        stack.push_async_callback(decisions.close)
        tts = make_tts(models, "tts")
        stack.push_async_callback(tts.close)
        # This demonstration cannot execute tools or external effects, so
        # unspoken generation can be safely replaced. Tool samples default off.
        policy = _ClefVoicePolicy(decisions, policy_cfg, allow_unspoken_merge=True)
        voice = VoiceAgent.from_models(
            models=models,
            query_topic=_QUERY,
            tts=tts,
            vad=VadConfig(),
            voice_gate=gate,
            turn_policy=policy,
            interrupted_topic=_INTERRUPTED,
            participant_left_topic=_LEFT,
            probes={"llm": llm.health, "clef": decisions.health, "tts": tts.health},
            text_input=False,
            interrupt_on_supersede=True,
        )
        runtime = AgentRuntime()
        assistant = _Assistant(llm)
        runtime.register("voice", voice)
        runtime.register("assistant", assistant)
        await stack.enter_async_context(runtime)
        stack.push_async_callback(assistant.close)
        await voice.run(runtime)
