# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small adapters for the sample's existing optional application capabilities."""

import asyncio
import json

from pydantic import ValidationError
from xr_ai_hub import FrameUnavailable
from xr_ai_models import ChatMessage
from xr_ai_routing import Admission, DestinationResult, RoutingAid
from xr_ai_tools.current_frame import CurrentFrameRequest
from xr_ai_tools.vision import ImageQueryRequest

from .._capture import _event, _frame
from ..sop._tools import rag_lookup_tool
from ._routing import _authorized, _choose


class _BackgroundDestination:
    def __init__(self, sop, name, prefix):
        self.sop, self.name, self.prefix = sop, name, prefix
        self.policy = sop._policy.destinations[name]
        self.pending = {}

    def routing_aid(self, _participant):
        return RoutingAid(self.name, self.policy.description)

    async def admit(self, turn, *, focused):
        self.pending.pop(turn.participant_id, None)
        tools = self.sop._background_tools(turn.participant_id)
        criteria = {name: text for name, text in self.policy.question.criteria.items()
                    if name == "other" or tools.get(self.prefix + "__" + name) is not None}
        if set(criteria) == {"other"}:
            return Admission("decline")
        question = self.policy.question.model_copy(update={"criteria": criteria})
        action = await _choose(self.sop._decisions, question, turn, {})
        if action == "other":
            return Admission("decline")
        arguments = {"instruction": turn.query} if self.name == "watch" and action == "start" else {}
        operation = self.prefix + "__" + action
        if action != "status" and not await _authorized(
            self.sop._decisions, self.sop._policy, turn, operation=operation,
            description=self.policy.question.criteria[action], arguments=arguments, target=self.policy.description,
        ):
            return Admission("decline")
        self.pending[turn.participant_id] = (turn.turn_id, action, arguments)
        return Admission("accept")

    async def answer(self, turn, *, started):
        pending = self.pending.pop(turn.participant_id, None)
        if pending is None or pending[0] != turn.turn_id:
            return DestinationResult()

        async def produce(reply):
            name = self.prefix + "__" + pending[1]
            tool = self.sop._background_tools(turn.participant_id).get(name)
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
                response = await self.sop._llm.chat((
                    ChatMessage(role="system", content=self.sop._policy.argument_prompt),
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

        return DestinationResult(text=await self.sop._respond(turn, started, produce))

    async def release(self, participant_id):
        self.pending.pop(participant_id, None)


class _QuestionDestination:
    def __init__(self, sop, name):
        self.sop, self.name = sop, name

    def routing_aid(self, _participant):
        description = (self.sop._policy.conversation_description if self.name == "conversation"
                       else self.sop._policy.destinations[self.name].description)
        return RoutingAid(self.name, description)

    async def admit(self, turn, *, focused):
        if self.name == "history" and self.sop._background_tools(turn.participant_id).get(
            "application_context__query"
        ) is None:
            return Admission("decline")
        return Admission("accept")

    async def answer(self, turn, *, started):
        async def produce(reply):
            if self.name == "observation":
                return await self._camera(turn, reply)
            observation, calls = None, []
            if self.name in {"reference", "history"}:
                if self.name == "reference":
                    tool = rag_lookup_tool(self.sop._rag)
                    arguments = {"query": turn.query, "top_k": 2}
                else:
                    tool = self.sop._background_tools(turn.participant_id).get("application_context__query")
                    arguments = {}
                await started()
                await _event("tool.started", name=tool.name, arguments=arguments)
                result = await tool.execute(tool.request_model.model_validate(arguments))
                await _event("tool.finished", name=tool.name, result=result)
                observation, calls = result.model_dump(mode="json"), [tool.name]
            text, extra, _ = await self.sop._generate_answer(
                turn.query, turn.participant_id, observation=observation,
            )
            await reply.send(text)
            return text, calls + extra

        return DestinationResult(text=await self.sop._respond(turn, started, produce))

    async def _camera(self, turn, reply):
        pieces = []
        try:
            async with asyncio.timeout(self.sop._vlm_timeout_s):
                frame = await self.sop._images.get_current_frame.execute(
                    CurrentFrameRequest(participant_id=turn.participant_id)
                )
                await _frame(frame)
                stream = self.sop._vision.stream(ImageQueryRequest(image=frame.image, query=turn.query))
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
        pass


def _destinations(sop):
    destinations = {"workflow": sop, "conversation": _QuestionDestination(sop, "conversation")}
    for name, prefix in {"watch": "change_watch", "transcript": "transcript", "journal": "video_log"}.items():
        if name in sop._policy.destinations:
            destinations[name] = _BackgroundDestination(sop, name, prefix)
    for name in ("observation", "reference", "history"):
        if name in sop._policy.destinations:
            destinations[name] = _QuestionDestination(sop, name)
    return destinations
