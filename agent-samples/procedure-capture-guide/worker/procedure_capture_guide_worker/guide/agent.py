# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose guide controls, SDK conversation, and the YAML execution backend.

The SDK router owns foreground scheduling and participant lifecycle. The guide
backend owns progress and background verification; conversation sees detached
state and read-only tools. Joining loads the selected guide without starting it.
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from xr_ai_conversation import ContextManager, Conversation, ConversationContextUpdate
from xr_ai_interaction import ResponseOutputs
from xr_ai_models import ChatMessage
from xr_ai_routing import Admission, AgentRouter, AgentTurn, DestinationResult, RoutingAid
from xr_ai_runtime import Agent, RuntimeClosedError
from xr_ai_tools import ToolSet
from xr_ai_tools.tool_calling import run_tool_loop
from xr_ai_voice import VOICE_OUTPUT_TOPIC

from .controls import classify_control
from .conversation import GuideConversation, ReadOnlyLanguage, _AnswerScope
from .events import INTERRUPTED_TOPIC, PARTICIPANT_JOINED_TOPIC, PARTICIPANT_LEFT_TOPIC, USER_QUERY_TOPIC
from .guiding import _Guide
from .policy import load_policy

_CONTEXT = "procedure-guide.execution"


class GuideAgent(Agent):
    """Compose a selected guide with SDK-owned controls and conversation routing."""

    def __init__(
        self,
        *,
        guide,
        llm,
        decisions,
        current_frame,
        image_query,
        vision_timeout_s,
        decision_timeout_s=15.0,
        measurement_query=None,
        traces_dir=None,
        policy=None,
    ):
        super().__init__()
        self._policy = policy or load_policy()
        self._llm, self._decisions = llm, decisions
        self._decision_timeout_s = decision_timeout_s
        self._runtime = self._router = self._context = self._conversation = None
        self._connected = set()
        self._pending = {}
        self._backend = _Guide(
            guide=guide,
            decisions=decisions,
            current_frame=current_frame,
            image_query=image_query,
            vision_timeout_s=vision_timeout_s,
            run_model=self._tool_loop,
            decision_timeout_s=decision_timeout_s,
            measurement_query=measurement_query,
            traces_dir=traces_dir,
            announce=self._announce,
            invalidate=self._withdraw,
            policy=self._policy,
            connected=lambda pid: pid in self._connected,
        )

    def bind_runtime(self, runtime):
        if self._runtime is not None:
            if self._runtime is not runtime:
                raise RuntimeError("Guide already belongs to another runtime")
            return
        self._runtime = runtime
        self._context = ContextManager(
            self._llm,
            summary_context_topics=(_CONTEXT,),
            dialogue_budget_bytes=16000,
            topic_budget_bytes=64000,
            compaction_input_budget_bytes=64000,
        )
        conversation = Conversation(
            ReadOnlyLanguage(self._llm, self._tool_loop),
            context=self._context,
            system_prompt=self._policy.conversation_prompt,
            instructions="Use the selected guide and evidence as read-only context. Never execute guide controls.",
            routing=RoutingAid("conversation", self._policy.conversation_description),
            runtime=runtime,
            output_topic=VOICE_OUTPUT_TOPIC,
        )
        self._conversation = GuideConversation(conversation, self._prepare_answer, self._record_answer)
        self._router = AgentRouter(
            {"guide": self, "conversation": self._conversation},
            decisions=self._decisions,
            query_topic=USER_QUERY_TOPIC,
            output_topic=VOICE_OUTPUT_TOPIC,
            output=self._routing_output,
            participant_joined_topic=PARTICIPANT_JOINED_TOPIC,
            participant_left_topic=PARTICIPANT_LEFT_TOPIC,
            interrupted_topic=INTERRUPTED_TOPIC,
            initialize_participant=self._join,
            cleanup_participant=self._leave,
        )
        runtime.register("guide-routing", self._router.runtime_subscriber())

    async def _routing_output(self, turn, _destination, text):
        async with self._outputs().reply(
            turn.participant_id, request_id=turn.turn_id, timestamp_us=turn.timestamp_us
        ) as reply:
            await reply.send(text)

    async def _join(self, pid):
        if pid in self._connected:
            return
        self._connected.add(pid)
        await self._say(pid, self._backend.join(pid))

    async def _leave(self, pid):
        self._connected.discard(pid)
        self._pending.pop(pid, None)
        await self._backend.leave(pid)
        await self._context.clear(pid)
        await self._withdraw(pid)

    def routing_aid(self, pid):
        return RoutingAid("guide", self._policy.guide_description, eligible=pid in self._connected)

    async def admit(self, turn, *, focused):
        guidance = await self._backend.guidance(turn.participant_id)
        if guidance is None:
            return Admission("decline")
        binding, decision = await classify_control(
            self._decisions,
            turn.query,
            guidance.context,
            timeout_s=self._decision_timeout_s,
            policy=self._policy,
        )
        self._backend.record_event(turn.participant_id, "control_decision", request_id=turn.turn_id, **decision)
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        if binding is None or turn.participant_id not in self._connected:
            return Admission("decline")
        self._pending[turn.participant_id] = (turn.turn_id, guidance, binding)
        return Admission("accept")

    async def answer(self, turn, *, started):
        proposal = self._pending.pop(turn.participant_id, None)
        if proposal is None or proposal[0] != turn.turn_id:
            return DestinationResult()
        await started()
        text = await self._execute_control(turn.participant_id, turn.query, proposal[1], proposal[2])
        async with self._outputs().reply(
            turn.participant_id, request_id=turn.turn_id, timestamp_us=turn.timestamp_us
        ) as reply:
            await reply.send(text)
        return DestinationResult("answer", text)

    async def release(self, pid):
        self._pending.pop(pid, None)

    async def _execute_control(self, pid, text, guidance, binding):
        run_id = self._backend.record_event(pid, "user_request", text=text)
        try:
            name, arguments = binding
            tool = guidance.tools.get(name)
            if tool is None:
                result = "That control is unavailable in the current guide state."
            else:
                result = (await tool.execute(tool.request_model.model_validate(arguments))).message
            result = await guidance.finish(result, refresh=False)
            self._backend.record_event(
                pid, "user_control_result", run_id=run_id, text=text, tool=name, arguments=arguments, response=result
            )
            return result
        except asyncio.CancelledError:
            await guidance.cancel()
            raise

    async def _prepare_answer(self, turn):
        guidance = await self._backend.guidance(turn.participant_id)
        if guidance is None:
            raise RuntimeError("Participant no longer owns a guide")
        await self._context.update(
            turn.participant_id,
            _CONTEXT,
            ConversationContextUpdate(
                text=json.dumps({"instructions": guidance.instructions, **guidance.context}, ensure_ascii=False)
            ),
            mode="summary",
        )
        tools = ToolSet(
            {
                name: tool
                for name, tool in guidance.tools.items()
                if name == "workflow__status" or not name.startswith("workflow__")
            }
        )
        self._backend.record_event(
            turn.participant_id, "conversation_request", run_id=guidance.context["run_id"],
            request_id=turn.turn_id, request=turn.query, guidance=guidance.context,
        )
        return _AnswerScope(guidance, tools)

    def _record_answer(self, turn, scope, result):
        if scope.guidance.context["run_id"] is None:
            return
        self._backend.record_event(
            turn.participant_id, "conversation_reply", run_id=scope.guidance.context["run_id"],
            request_id=turn.turn_id, request=turn.query, response=result.text,
            step_id=scope.guidance.context["step_id"], revision=scope.guidance.context["step_revision"],
        )

    async def _route(self, text, participant_id):
        """Execute one direct turn using the same admission and answer boundaries."""
        turn = AgentTurn(participant_id, uuid4().hex, text)
        if (await self.admit(turn, focused=False)).disposition == "accept":
            return (await self.answer(turn, started=_started)).text
        return (await self._conversation.answer(turn, started=_started)).text

    async def _tool_loop(self, system, user, tools, *, foreground=False, messages=None):
        options = self._policy.foreground_options if foreground else self._policy.observation_options

        async def call_model(transcript, definitions):
            return await self._llm.chat(transcript, tools=definitions, temperature=0.0, **options)

        result = await run_tool_loop(
            messages or (ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)),
            tools,
            call_model,
            max_iterations=3,
            max_tool_calls=6,
        )
        return result.content.strip() or "Done."

    def _outputs(self):
        return ResponseOutputs(
            self._runtime, VOICE_OUTPUT_TOPIC, source="procedure-guide", closed_errors=(RuntimeClosedError,)
        )

    async def _say(self, pid, text):
        if self._runtime is not None and self._runtime.running:
            await self._outputs().update(pid, text, urgent=True)

    async def _announce(self, pid, text):
        if self._runtime is not None and self._runtime.running:
            await self._outputs().update(pid, text)

    async def _withdraw(self, pid):
        if self._runtime is not None and self._runtime.running:
            await self._outputs().withdraw(pid)

    async def stop(self):
        if self._router is not None:
            await self._router.stop()
        await self._backend.stop()
        self._connected.clear()
        self._pending.clear()
        if self._conversation is not None:
            await self._conversation.close()
        if self._context is not None:
            await self._context.close()


async def _started():
    pass
