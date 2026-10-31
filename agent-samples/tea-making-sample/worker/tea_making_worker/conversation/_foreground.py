# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic foreground routing with one participant-scoped tool loop."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field

import nemo_relay
from loguru import logger
from pydantic import ValidationError
from xr_ai_conversation import ContextManager
from xr_ai_hub import FrameUnavailable
from xr_ai_interaction import (
    Interrupted,
    ParticipantLeft,
    ResponseOutputs,
    UserQuery,
)
from xr_ai_models import ChatMessage, DecisionService, LLMService, VLMService
from xr_ai_routing import AgentTurn
from xr_ai_runtime import Agent, RuntimeClosedError, RuntimeContext, subscribe
from xr_ai_tools import Tool, ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.tool_calling import run_tool_loop
from xr_ai_tools.vision import (
    ImageQueryRequest,
    ImageQueryResult,
    ImageQueryTool,
    StreamingImageQueryTool,
)
from xr_ai_voice import VOICE_OUTPUT_TOPIC

from .._capture import _attempt, _event, _frame
from ..events import (
    FOREGROUND_RECORD_TOPIC,
    INTERRUPTED_TOPIC,
    PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    ForegroundRecord,
    ParticipantCleanupComplete,
)
from ..flows import VisualValidationStep
from ..images import ParticipantImageAgent
from ..sop._engine import GuidanceAgent
from ..sop._tools import CurrentViewRequest, participant_current_view_tool, rag_lookup_tool
from ._capture import _CompactionLanguage, _context_owner
from ._policy import _Action
from ._routing import _make_router


@dataclass(slots=True)
class _RoutedOutput:
    ctx: RuntimeContext | None
    timestamp_us: int | None
    text: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    spoken: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedTurn:
    system_prompt: str
    user_message: str
    route: str


class ForegroundAgent(Agent):
    """Select the idle root or active tea tool set before invoking the model."""

    def __init__(
        self,
        *,
        llm: LLMService,
        decisions: DecisionService,
        images: ParticipantImageAgent,
        vlm: VLMService,
        rag: RAGTools,
        guidance: GuidanceAgent,
        application_tools: Callable[[str], ToolSet],
        prompt: str,
        current_view_prompt: str,
        vlm_timeout_s: float = 15.0,
        context: ContextManager | None = None,
    ) -> None:
        if vlm_timeout_s <= 0:
            raise ValueError("vlm_timeout_s must be positive")
        if not current_view_prompt.strip():
            raise ValueError("current_view_prompt must not be empty")
        self._view_prompt = current_view_prompt.strip()
        self._vision = StreamingImageQueryTool(
            images=images.images,
            vlm=vlm,
            system_prompt=self._view_prompt,
        )
        super().__init__((self._vision,))
        self._vlm = vlm
        self._llm = llm
        self._context = context or ContextManager(_CompactionLanguage(llm))
        self._owns_context = context is None
        self._decisions = decisions
        self._images = images
        self._rag = rag
        self._guidance = guidance
        self._background_tools = application_tools
        self._prompt = prompt.strip()
        if guidance.workflow.foreground is None:
            raise ValueError("foreground routing requires task.foreground in workflow YAML")
        self._policy = guidance.workflow.foreground
        self._vlm_timeout_s = vlm_timeout_s
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._outputs: dict[str, _RoutedOutput] = {}
        self._router = _make_router(
            guidance,
            decisions,
            self._policy,
            self._workflow_tools,
            self._application_tools,
            self._execute_routed,
            self._stale_routed,
            self._router_output,
        )

    @subscribe(USER_QUERY_TOPIC)
    async def answer(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)
        await self._cancel(participant_id)
        task = asyncio.create_task(
            self._run_turn(query, ctx),
            name=f"tea-foreground:{participant_id}",
            context=nemo_relay.fork_asyncio_context(),
        )
        self._tasks[participant_id] = task
        task.add_done_callback(lambda completed, pid=participant_id: self._discard(pid, completed))

    @subscribe(PARTICIPANT_LEFT_TOPIC)
    async def participant_left(
        self,
        _event: ParticipantLeft,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = self._participant(ctx)
        await self._cancel(participant_id)
        await self._router.participant_left(participant_id)
        if self._owns_context:
            await self._context.clear(participant_id)
        await ctx.publish(
            PARTICIPANT_CLEANUP_COMPLETE_TOPIC,
            ParticipantCleanupComplete(
                generation=ctx.metadata.message_id,
                producer="foreground",
            ),
        )

    @subscribe(INTERRUPTED_TOPIC)
    async def interrupted(
        self,
        _event: Interrupted,
        ctx: RuntimeContext,
    ) -> None:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            for participant in tuple(self._tasks):
                await self._cancel(participant)
            await self._router.interrupt()
        else:
            await self._cancel(participant_id)

    async def stop(self) -> None:
        """Cancel every participant turn owned by this agent."""

        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._router.stop()
        if self._owns_context:
            await self._context.close()

    async def _run_turn(self, query: UserQuery, ctx: RuntimeContext) -> None:
        participant_id = self._participant(ctx)

        def snapshot():
            session = self._guidance.store.find(participant_id)
            return (
                {
                    "state": dict(session.state),
                    "step_id": session.step_id,
                    "revision": session.revision,
                    "run_id": session.run_id,
                    "evidence_hits": session.evidence_hits,
                    "active": session.active,
                }
                if session
                else {}
            )

        session = self._guidance.store.find(participant_id)
        async with _attempt(
            participant_id,
            "foreground",
            step_id=session.step_id if session else None,
            snapshot=snapshot,
            request_id=ctx.metadata.message_id,
        ):
            await self._record_turn(query, ctx, participant_id)

    async def _record_turn(self, query, ctx, participant_id):
        await _event("query.received", text=query.text, timestamp_us=query.timestamp_us)
        with nemo_relay.use_scope_stack(nemo_relay.create_scope_stack()):
            try:
                response, tools, spoken = await self._answer(
                    query.text,
                    participant_id,
                    ctx,
                    timestamp_us=query.timestamp_us,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.opt(exception=True).error("tea foreground query failed pid={!r}", participant_id)
                response = "I couldn't complete that request. Please try again."
                tools = []
                spoken = False
            try:
                if not spoken:
                    outputs = ResponseOutputs(ctx, VOICE_OUTPUT_TOPIC, closed_errors=(RuntimeClosedError,))
                    async with outputs.reply(
                        participant_id,
                        request_id=ctx.metadata.message_id,
                        timestamp_us=query.timestamp_us,
                    ) as reply:
                        await reply.send(response)
            except RuntimeClosedError:
                return
            self._remember_exchange(participant_id, query.text, response, ctx.metadata.message_id)
            await _event("answer.produced", response=response, tools=tools)
            try:
                await ctx.publish(
                    FOREGROUND_RECORD_TOPIC,
                    ForegroundRecord(
                        timestamp_us=query.timestamp_us,
                        query=query.text,
                        response=response,
                        tools=tools,
                    ),
                )
            except RuntimeClosedError:
                return
            except Exception:
                logger.opt(exception=True).error("tea foreground record failed pid={!r}", participant_id)

    def _workflow_tools(self, participant_id: str) -> ToolSet:
        active = self._guidance.active_tools(participant_id)
        return active if active is not None else self._guidance.root_tools(participant_id)

    def _application_tools(self, participant_id: str) -> ToolSet:
        output = self._outputs.get(participant_id)
        return self._root_tools(
            participant_id,
            ctx=output.ctx if output else None,
            timestamp_us=output.timestamp_us if output else None,
        )

    async def _router_output(self, turn: AgentTurn, _route: str, text: str) -> None:
        self._outputs[turn.participant_id].text.append(text)

    async def _stale_routed(self, turn, started) -> None:
        await started()
        self._outputs[turn.participant_id].text.append(self._policy.stale_message)

    async def _action_request(self, tool, arguments):
        try:
            return tool.request_model.model_validate(arguments)
        except ValidationError as exc:
            errors = exc.errors()
            if len(errors) != 1 or errors[0]["type"] != "string_too_long" or len(errors[0]["loc"]) != 1:
                raise
            error = errors[0]
            field = error["loc"][0]
            response = await self._llm.chat(
                (
                    ChatMessage(role="system", content=self._policy.argument_prompt),
                    ChatMessage(
                        role="user", content=_json(text=arguments[field], max_characters=error["ctx"]["max_length"])
                    ),
                ),
                max_tokens=512,
                temperature=0.0,
                enable_thinking=False,
            )
            if response.tool_calls:
                raise ValueError("argument generation cannot choose a tool") from exc
            return tool.request_model.model_validate({**arguments, field: response.content.strip()})

    async def _execute_routed(self, turn, action, started, *, workflow=False) -> str:
        await self._execute_action(turn, action, started, workflow=workflow)
        return " ".join(self._outputs[turn.participant_id].text).strip()

    async def _execute_action(self, turn, action: _Action, started, *, workflow: bool = False) -> None:
        output = self._outputs[turn.participant_id]
        observation = None
        if action.tool is not None:
            tools = (
                self._workflow_tools(turn.participant_id) if workflow else self._application_tools(turn.participant_id)
            )
            tool = tools.get(action.tool)
            if tool is None:
                await self._stale_routed(turn, started)
                return
            session = self._guidance.store.find(turn.participant_id)
            state = dict(session.state) if session is not None else {}
            arguments = _bind_arguments(action.arguments, turn.query, state)
            request = await self._action_request(tool, arguments)
            if action.observation is not None:
                tool = participant_current_view_tool(
                    turn.participant_id,
                    self._images.get_current_frame,
                    ImageQueryTool(
                        images=self._images.images,
                        vlm=self._vlm,
                        system_prompt=action.observation.extraction_question(self._view_prompt),
                    ),
                    timeout_s=self._vlm_timeout_s,
                )
            await started()
            await _event("tool.started", name=action.tool, arguments=request.model_dump(mode="json"))
            result = await tool.execute(request)
            await _event("tool.finished", name=action.tool, result=result)
            output.calls.append(action.tool)
            if action.observation is not None:
                if not result.available:
                    output.text.append(result.text)
                    return
                try:
                    validator = VisualValidationStep(
                        question=request.question, policy=action.observation,
                        observer=self._guidance._image_query, decisions=self._decisions, llm=self._llm,
                        tools=tools, timeout_s=self._vlm_timeout_s, on_event=_event,
                    )
                    fact = await validator.evaluate(result.text, expected=state)
                except ValueError:
                    logger.opt(exception=True).warning("structured current-view fact unavailable")
                    output.text.append(self._policy.unavailable_message)
                    return
                await _event(
                    "observation.judged", evidence=fact.evidence, values=fact.values, native_result=fact.result
                )
                if fact.evidence == "unknown":
                    output.text.append(self._policy.unavailable_message)
                    return
                observation = fact.result if fact.result is not None else fact.values
                if action.observation.evaluation is not None:
                    output.calls.append(action.observation.evaluation.function)
            elif action.generate:
                observation = result.model_dump(mode="json")
            else:
                text = getattr(result, "message", None) or getattr(result, "text", None)
                if text is None:
                    raise ValueError("direct action requires a human-readable tool result")
                output.text.append(text)
                output.spoken = bool(action.tool == "current_view" and tool.return_direct and output.ctx is not None)
                return
        text, calls, spoken = await self._generate_answer(
            turn.query,
            turn.participant_id,
            output.ctx,
            timestamp_us=turn.timestamp_us,
            observation=observation,
            workflow_context=workflow,
        )
        await started()
        output.text.append(text)
        output.calls.extend(calls)
        output.spoken = output.spoken or spoken

    async def _answer(self, query, participant_id, ctx=None, *, timestamp_us=None):
        output = _RoutedOutput(ctx, timestamp_us)
        self._outputs[participant_id] = output
        try:
            await self._router.handle(
                AgentTurn(
                    participant_id,
                    ctx.metadata.message_id if ctx else f"local:{id(output)}",
                    query,
                    timestamp_us=timestamp_us,
                )
            )
            text = " ".join(output.text).strip()
            if ctx is None:
                self._remember_exchange(participant_id, query, text, None)
            return text, output.calls, output.spoken
        finally:
            if self._outputs.get(participant_id) is output:
                self._outputs.pop(participant_id, None)

    def _remember_exchange(self, participant_id, query, response, exchange_id) -> None:
        with _context_owner(participant_id):
            self._context.record_exchange(
                participant_id, query, response,
                timestamp_us=time.time_ns() // 1_000, exchange_id=exchange_id,
            )

    async def _generate_answer(
        self,
        query: str,
        participant_id: str,
        ctx: RuntimeContext | None = None,
        *,
        timestamp_us: int | None = None,
        observation: object = None,
        workflow_context: bool = False,
    ) -> tuple[str, list[str], bool]:
        await self._refresh_conversation_context(participant_id)
        if workflow_context:
            turn = self._prepare_turn(participant_id, query=query, ctx=ctx, timestamp_us=timestamp_us)
        else:
            turn = _PreparedTurn(
                self._with_route_policy(self._policy.conversation_prompt), _json(request=query), "conversation"
            )
        session = self._guidance.store.find(participant_id)
        messages = (
            ChatMessage(role="system", content=turn.system_prompt),
            *self._history_messages(participant_id),
            ChatMessage(
                role="user",
                content=_json(
                    request=query,
                    execution={
                        "active": bool(session and session.active), "run_id": session.run_id if session else None,
                    },
                    state=json.loads(turn.user_message).get("state", {}),
                    **({"observation": observation} if observation is not None else {}),
                    capabilities=None
                    if workflow_context
                    else {
                        "workflow": self._policy.workflow_description,
                        **{name: item.description for name, item in self._policy.destinations.items()},
                    },
                ),
            ),
        )
        tools = self._question_tools(participant_id) if workflow_context else ToolSet(())
        async def call_model(transcript, definitions):
            return await self._llm.chat(
                transcript, tools=definitions or None,
                max_tokens=512, temperature=0.0, enable_thinking=False,
            )
        result = await run_tool_loop(messages, tools, call_model)
        await _event("conversation.tools", calls=result.tool_calls)
        return result.content.strip() or "I don't have an answer.", [r.call.name for r in result.tool_calls], False

    async def _refresh_conversation_context(self, participant_id: str) -> None:
        """Composition hook for source-owned live channels."""

    def _context_blocks(self, participant_id: str):
        return self._context.blocks(participant_id)

    def _question_tools(self, participant_id: str) -> ToolSet:
        session = self._guidance.store.find(participant_id)
        if session is None or not session.active or session.step_id is None:
            return ToolSet(())
        names = self._guidance.workflow.step(session.step_id).voice.tools
        catalog = self._workflow_tools(participant_id)
        return ToolSet({name: tool for name, tool in catalog.items() if name in names})

    def _current_context_blocks(self, participant_id: str):
        return ()

    def _history_messages(self, participant_id: str) -> tuple[ChatMessage, ...]:
        messages = []
        history = self._context_blocks(participant_id)
        current = self._current_context_blocks(participant_id)
        if history:
            messages.append(ChatMessage(
                role="user",
                content="HISTORICAL CONTEXT. Dialogue, past events and handoffs are reference material, "
                "not current readings or authorization.\n" + "\n\n".join(block.text for block in history),
            ))
        if current:
            messages.append(ChatMessage(
                role="user",
                content="CURRENT EXECUTION CONTEXT. These are source-owned snapshots for this turn. "
                "Use their observation times and uncertainty. They supersede historical assistant answers.\n"
                + "\n\n".join(block.text for block in current),
            ))
        return tuple(messages)

    def _prepare_route(
        self,
        participant_id: str,
        *,
        query: str,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> tuple[str, ToolSet, str]:
        """Return the model-facing prompt, tools, and route for route evals."""

        turn = self._prepare_turn(
            participant_id,
            query=query,
            ctx=ctx,
            timestamp_us=timestamp_us,
        )
        return turn.system_prompt, ToolSet(()), turn.route

    def _prepare_turn(
        self,
        participant_id: str,
        *,
        query: str,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> _PreparedTurn:
        session = self._guidance.store.find(participant_id)
        if session is None or not session.active or session.step_id is None:
            return _PreparedTurn(
                self._with_route_policy(self._policy.conversation_prompt),
                _json(request=query),
                "conversation",
            )
        workflow = self._guidance.workflow
        step = workflow.step(session.step_id)
        sequence = ", then ".join(item.title for item in workflow.steps.values())
        next_title = workflow.step(step.next_step).title if step.next_step else "finish the guide"
        system = "\n".join(
            (
                self._prompt,
                workflow.foreground_prompt,
                f"Guide order: {sequence}. Current step: {step.title}. Next: {next_title}.",
                step.voice.prompt,
                self._policy.active_prompt,
            )
        )
        return _PreparedTurn(
            system,
            _json(request=query, state=workflow.project(step, session.state)),
            "workflow",
        )

    def _with_route_policy(self, route_prompt: str) -> str:
        return f"{self._prompt}\n\n{route_prompt}"

    def _root_tools(
        self,
        participant_id: str,
        *,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> ToolSet:
        current_view = self._current_view_tool(
            participant_id,
            ctx=ctx,
            timestamp_us=timestamp_us,
        )
        return _merge_tool_sets(
            ToolSet((current_view, rag_lookup_tool(self._rag))),
            self._background_tools(participant_id),
        )

    def _current_view_tool(
        self,
        participant_id: str,
        *,
        ctx: RuntimeContext | None,
        timestamp_us: int | None,
    ) -> Tool[CurrentViewRequest, ImageQueryResult]:
        async def inspect(request: CurrentViewRequest) -> ImageQueryResult:
            if ctx is None:
                raise RuntimeError("current-view streaming requires a runtime context")
            return await self._stream_current_view(
                request.question,
                participant_id,
                ctx,
                timestamp_us=timestamp_us,
            )

        return Tool(
            "current_view",
            "Inspect the participant's camera only when answering requires evidence from the visible present scene.",
            CurrentViewRequest,
            ImageQueryResult,
            inspect,
            return_direct=True,
            render_result=lambda result: result.text,
        )

    async def _stream_current_view(
        self,
        query: str,
        participant_id: str,
        ctx: RuntimeContext,
        *,
        timestamp_us: int | None,
    ) -> ImageQueryResult:
        chunks: list[str] = []
        outputs = ResponseOutputs(ctx, VOICE_OUTPUT_TOPIC, closed_errors=(RuntimeClosedError,))
        async with outputs.reply(
            participant_id,
            request_id=ctx.metadata.message_id,
            timestamp_us=timestamp_us,
        ) as reply:
            try:
                async with asyncio.timeout(self._vlm_timeout_s):
                    try:
                        frame = await self._images.get_current_frame.execute(
                            CurrentFrameRequest(participant_id=participant_id)
                        )
                    except FrameUnavailable as exc:
                        unavailable = str(exc)
                        await reply.send(unavailable)
                        return ImageQueryResult(text=unavailable, available=False)
                    await _frame(frame)
                    stream = self._vision.stream(ImageQueryRequest(image=frame.image, query=query))
                    try:
                        async for chunk in stream:
                            if not chunks and not chunk.text.strip():
                                continue
                            chunks.append(chunk.text)
                            await reply.send(chunk.text)
                    finally:
                        close = getattr(stream, "aclose", None)
                        if close is not None:
                            await close()
                    if not chunks:
                        unavailable = (
                            "Unable to inspect the current frame because the vision model returned no description."
                        )
                        await reply.send(unavailable)
                        return ImageQueryResult(text=unavailable, available=False)
                return ImageQueryResult(text="".join(chunks))
            except TimeoutError:
                unavailable = "Unable to inspect the current frame before the vision timeout."
                await reply.send(unavailable)
                return ImageQueryResult(text=unavailable, available=False)

    async def _cancel(self, participant_id: str) -> None:
        task = self._tasks.pop(participant_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _discard(self, participant_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(participant_id) is task:
            self._tasks.pop(participant_id, None)
        if task.cancelled():
            return
        with suppress(asyncio.CancelledError):
            if error := task.exception():
                logger.error("tea foreground stopped pid={!r}: {!r}", participant_id, error)

    @staticmethod
    def _participant(ctx: RuntimeContext) -> str:
        participant_id = ctx.metadata.participant_id
        if participant_id is None:
            raise ValueError("foreground queries require a participant")
        return participant_id


def _merge_tool_sets(*catalogs: ToolSet) -> ToolSet:
    tools: dict[str, Tool] = {}
    for catalog in catalogs:
        for name, tool in catalog.items():
            if name in tools:
                raise ValueError(f"duplicate participant tool: {name}")
            tools[name] = tool
    return ToolSet(tools)


def _bind_arguments(value: object, query: str, state: dict[str, object]) -> object:
    if isinstance(value, dict):
        return {key: _bind_arguments(item, query, state) for key, item in value.items()}
    if isinstance(value, list):
        return [_bind_arguments(item, query, state) for item in value]
    if value == "$request":
        return query
    if isinstance(value, str) and value.startswith("$state."):
        return state[value.removeprefix("$state.")]
    return value


def _json(**value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


__all__ = ["ForegroundAgent"]
