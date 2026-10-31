# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt optional sample capabilities to the SDK destination contract.

Background destinations classify and authorize their existing typed controls;
question destinations use the shared answer path or stream the current camera
view. AgentRouter owns admission lifecycle, cancellation and focus. These adapters
neither supervise user-turn tasks nor interpret a generic action language.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable

from pydantic import ValidationError
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage, DecisionService, LLMService
from xr_ai_routing import Admission, DestinationResult, RoutableAgent, RoutingAid
from xr_ai_tools import ToolSet
from xr_ai_tools.current_frame import CurrentFrameRequest, CurrentFrameTool
from xr_ai_tools.rag import RAGTools
from xr_ai_tools.vision import ImageQueryRequest, StreamingImageQueryTool

from .._capture import _event, _frame
from ..conversation._policy import ConversationPolicy, _DestinationPolicy
from ..conversation._routing import _authorized, _choose
from .sop._tools import rag_lookup_tool


class _BackgroundControlAgent(RoutableAgent):
    """Expose one optional background capability through bounded typed controls."""

    def __init__(
        self,
        *,
        name: str,
        prefix: str,
        policy: _DestinationPolicy,
        control_policy: ConversationPolicy,
        decisions: DecisionService,
        llm: LLMService,
        participant_tools: Callable[[str], ToolSet],
        respond: Callable[..., Awaitable[str]],
    ) -> None:
        """Bind a capability's existing tools/policy and shared response callback.

        This is a top-level routed agent, not another runtime subscriber. The
        capability's runtime agent owns background work; AgentRouter owns turns.
        The injected callback records delivered replies in the shared dialogue.
        """
        self.name = name
        self.prefix = prefix
        self.policy = policy
        self._control_policy = control_policy
        self._decisions = decisions
        self._llm = llm
        self._participant_tools = participant_tools
        self._respond = respond
        self.pending = {}

    def routing_aid(self, _participant):
        """Describe the optional capability using its external destination policy."""
        return RoutingAid(self.name, self.policy.description)

    async def admit(self, turn, *, focused):
        """Classify only available tools and separately authorize state-changing controls."""
        self.pending.pop(turn.participant_id, None)
        tools = self._participant_tools(turn.participant_id)
        criteria = {name: text for name, text in self.policy.question.criteria.items()
                    if name == "other" or tools.get(self.prefix + "__" + name) is not None}
        if set(criteria) == {"other"}:
            return Admission("decline")
        question = self.policy.question.model_copy(update={"criteria": criteria})
        action = await _choose(self._decisions, question, turn, {})
        if action == "other":
            return Admission("decline")
        arguments = {"instruction": turn.query} if self.name == "watch" and action == "start" else {}
        operation = self.prefix + "__" + action
        if action != "status" and not await _authorized(
            self._decisions, self._control_policy, turn, operation=operation,
            description=self.policy.question.criteria[action], arguments=arguments, target=self.policy.description,
        ):
            return Admission("decline")
        self.pending[turn.participant_id] = (turn.turn_id, action, arguments)
        return Admission("accept")

    async def answer(self, turn, *, started):
        """Execute the matching admitted control and publish through SDK reply outputs.

        Only the existing watcher instruction-length validation gets a model-assisted
        shortening retry; other validation errors preserve their normal failure path.
        """
        pending = self.pending.pop(turn.participant_id, None)
        if pending is None or pending[0] != turn.turn_id:
            return DestinationResult()

        async def produce(reply):
            name = self.prefix + "__" + pending[1]
            tool = self._participant_tools(turn.participant_id).get(name)
            arguments = pending[2]
            try:
                request = tool.request_model.model_validate(arguments)
            except ValidationError as error:
                # The watcher owns a bounded instruction string; generation may
                # shorten it but cannot select a different operation or target.
                failures = error.errors()
                if (self.name != "watch" or pending[1] != "start" or len(failures) != 1
                        or failures[0]["type"] != "string_too_long"):
                    raise
                response = await self._llm.chat((
                    ChatMessage(role="system", content=self._control_policy.argument_prompt),
                    ChatMessage(role="user", content=json.dumps({
                        "text": arguments["instruction"], "max_characters": failures[0]["ctx"]["max_length"],
                    })),
                ), max_tokens=512, temperature=0.0, enable_thinking=False)
                if response.tool_calls:
                    raise ValueError("watch instruction shortening cannot choose a tool") from error
                request = tool.request_model.model_validate({"instruction": response.content.strip()})
            await started()
            await _event("tool.started", name=name, arguments=request.model_dump(mode="json"))
            result = await tool.execute(request)
            await _event("tool.finished", name=name, result=result)
            await reply.send(result.message)
            return result.message, [name]

        return DestinationResult(text=await self._respond(turn, started, produce))

    async def release(self, participant_id):
        """Discard the capability's unconsumed admission for this participant."""
        self.pending.pop(participant_id, None)


class _QuestionAgent(RoutableAgent):
    """Answer informational requests without owning a workflow or turn supervisor."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        respond: Callable[..., Awaitable[str]],
        generate_answer: Callable[..., Awaitable[tuple[str, list[str], bool]]],
        participant_tools: Callable[[str], ToolSet],
        current_frame: CurrentFrameTool,
        vision: StreamingImageQueryTool,
        rag: RAGTools,
        vlm_timeout_s: float,
    ) -> None:
        """Bind one informational route to explicit source and answer dependencies.

        Shared answer/output callbacks keep the existing context and capture path.
        This agent does not own or inspect SOPAgent or its workflow internals.
        """
        self.name = name
        self._description = description
        self._respond = respond
        self._generate_answer = generate_answer
        self._participant_tools = participant_tools
        self._current_frame = current_frame
        self._vision = vision
        self._rag = rag
        self._vlm_timeout_s = vlm_timeout_s

    def routing_aid(self, _participant):
        """Describe general conversation or a configured informational destination."""
        return RoutingAid(self.name, self._description)

    async def admit(self, turn, *, focused):
        """Accept questions only when their required optional history tool exists."""
        if self.name == "history" and self._participant_tools(turn.participant_id).get(
            "application_context__query"
        ) is None:
            return Admission("decline")
        return Admission("accept")

    async def answer(self, turn, *, started):
        """Read any requested source, then use the common answer or camera stream path."""
        async def produce(reply):
            if self.name == "observation":
                return await self._camera(turn, reply)
            observation, calls = None, []
            if self.name in {"reference", "history"}:
                if self.name == "reference":
                    tool = rag_lookup_tool(self._rag)
                    arguments = {"query": turn.query, "top_k": 2}
                else:
                    tool = self._participant_tools(turn.participant_id).get("application_context__query")
                    arguments = {}
                await started()
                await _event("tool.started", name=tool.name, arguments=arguments)
                result = await tool.execute(tool.request_model.model_validate(arguments))
                await _event("tool.finished", name=tool.name, result=result)
                observation, calls = result.model_dump(mode="json"), [tool.name]
            text, extra, _ = await self._generate_answer(
                turn.query, turn.participant_id, observation=observation,
            )
            await reply.send(text)
            return text, calls + extra

        return DestinationResult(text=await self._respond(turn, started, produce))

    async def _camera(self, turn, reply):
        """Stream the current frame directly through the SDK reply context.

        Whitespace-only prefixes are skipped, but subsequent chunks retain their exact
        text. Always close the model stream, including interruption and inference failure.
        """
        pieces = []
        try:
            async with asyncio.timeout(self._vlm_timeout_s):
                frame = await self._current_frame.execute(
                    CurrentFrameRequest(participant_id=turn.participant_id)
                )
                await _frame(frame)
                stream = self._vision.stream(ImageQueryRequest(image=frame.image, query=turn.query))
                try:
                    async for part in stream:
                        if part.text.strip() or pieces:
                            pieces.append(part.text)
                            await reply.send(part.text)
                finally:
                    await stream.aclose()
        except FrameUnavailable as error:
            pieces = [str(error)]
            await reply.send(pieces[0])
        if not pieces:
            pieces = ["Unable to inspect the current frame because the vision model returned no description."]
            await reply.send(pieces[0])
        return "".join(pieces).strip(), ["current_view"]

    async def release(self, _participant):
        """Release router focus; informational questions retain no admission state."""
        pass


def _destinations(
    sop: RoutableAgent,
    *,
    policy: ConversationPolicy,
    decisions: DecisionService,
    llm: LLMService,
    participant_tools: Callable[[str], ToolSet],
    respond: Callable[..., Awaitable[str]],
    generate_answer: Callable[..., Awaitable[tuple[str, list[str], bool]]],
    current_frame: CurrentFrameTool,
    vision: StreamingImageQueryTool,
    rag: RAGTools,
    vlm_timeout_s: float,
) -> dict[str, RoutableAgent]:
    """Compose the same top-level routes from explicit application dependencies.

    The SOP is one destination, not an implicit dependency container for its peers.
    The caller supplies existing callbacks for shared answer/output/context behavior;
    this factory introduces no agent hierarchy or additional turn lifecycle.
    """
    destinations = {"workflow": sop}

    def question_agent(name, description):
        return _QuestionAgent(
            name=name,
            description=description,
            respond=respond,
            generate_answer=generate_answer,
            participant_tools=participant_tools,
            current_frame=current_frame,
            vision=vision,
            rag=rag,
            vlm_timeout_s=vlm_timeout_s,
        )

    destinations["conversation"] = question_agent("conversation", policy.conversation_description)
    for name, prefix in {"watch": "change_watch", "transcript": "transcript", "journal": "video_log"}.items():
        if name in policy.destinations:
            destinations[name] = _BackgroundControlAgent(
                name=name,
                prefix=prefix,
                policy=policy.destinations[name],
                control_policy=policy,
                decisions=decisions,
                llm=llm,
                participant_tools=participant_tools,
                respond=respond,
            )
    for name in ("observation", "reference", "history"):
        if name in policy.destinations:
            destinations[name] = question_agent(name, policy.destinations[name].description)
    return destinations
