<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-conversation

`xr-ai-conversation` provides a streamed general conversation destination with
current-view visual question answering and a typed front end for routing to
application destinations. The package depends on `xr_ai_models` and native
`xr_ai_tools`; it adds no model client or transport.

## Streamed conversation and visual questions

`Conversation` uses one `LLMService.stream_events()` request to answer directly
or select the `current_view` tool. A direct answer streams as text events arrive.
A current-view call provides a required standalone visual question, including
resolved follow-up references; the conversation then executes
`CurrentFrameTool` and streams `StreamingImageQueryTool` output. Frame selection
and visual inference remain separate tools. Tool-call preamble guidance is a
prompt contract: text already emitted before a later tool call cannot be
retracted. The core prompt is deliberately small; the application supplies its
own instructions, while the tool definition describes how to formulate a
standalone visual question.

```python
from xr_ai_conversation import Conversation, ConversationExchange, RoutingAid

conversation = Conversation(
    llm,
    instructions=load_text("conversation.txt"),
    routing=RoutingAid(
        name="conversation",
        description="General conversation and current-view visual questions",
    ),
    frames=frames,
    vision=streaming_vision,
)
history = (ConversationExchange(user="Which bottle is fullest?", assistant="The left one."),)
async for text in conversation.stream("Read its label", "participant-1", history=history):
    await speak(text)
```

For a text-only destination, construct `Conversation(llm, instructions=...)`
without camera or visual-query tools. It will not register `current_view` and
its default routing description says that visual tools are unavailable.
Application instructions define the destination's domain, role, tone, and
handling policy. `RoutingAid` separately gives the front end a concise name,
description, and optional tie-breaking priority; it guides route selection and
does not enforce the application's policy.

## Multi-destination routing

Each destination declares a `RoutingAid`, checks a `ConversationTurn` with
`admit()` before speaking or starting work, and answers with source-tagged
stream chunks. `Conversation` implements that same contract for general
conversation, alongside domain destinations such as a scene editor or a
foreground guide. `ConversationFrontEnd` uses the typed `DecisionService` to
select among real destinations, lets an explicitly focused destination receive
the next turn first, and returns to route selection when it releases or declines.
A declined route is excluded from one reroute; uncertain or unavailable work
ends with a clarification instead of invoking a guessed destination. Routing
descriptions and priority are selection hints; application policy belongs in
the destination's instructions and admission check. Destinations decline or
clarify in `admit()` before work begins; after work, `answer()` returns only
`answer`, `focus`, or `release`. The front end may return `clarify` when no route
can proceed.

The front end owns per-participant reply tasks and focus assignment. Each
destination owns its persistent application state and cleanup. `interrupt()`
cancels current work while preserving focus; `participant_left()` cancels the
turn and calls `release()` on destinations that handled that participant;
`stop()` drains work and releases all tracked participant state. Output
callbacks receive the destination name and text so applications can keep speech
attribution aligned with the route.

```python
from xr_ai_conversation import ConversationFrontEnd

front_end = ConversationFrontEnd(
    {"conversation": conversation, "scene": scene_destination, "guide": guide_destination},
    decision_service,
)
result = await front_end.handle(turn, emit=publish_chunk)
```

## Public API

```{eval-rst}
.. automodule:: xr_ai_conversation
   :members:
   :undoc-members:
```
