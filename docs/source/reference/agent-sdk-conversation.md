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
retracted.

```python
from xr_ai_conversation import Conversation, ConversationExchange

conversation = Conversation(llm, frames=frames, vision=streaming_vision)
history = (ConversationExchange(user="Which bottle is fullest?", assistant="The left one."),)
async for text in conversation.stream("Read its label", "participant-1", history=history):
    await speak(text)
```

For a text-only destination, construct `Conversation(llm)` without camera or
visual-query tools. It will not register `current_view` and advertises no
current-view inspection capability. Applications can set `name`, `persona`,
`scope`, and `exclusions` to align route selection and response guidance with
their domain; the application still owns the meaning of that configuration.

## Multi-destination routing

Each destination declares a `RoutingAid`, checks a `ConversationTurn` with
`admit()` before speaking or starting work, and answers with source-tagged
stream chunks. `Conversation` implements that same contract for general
conversation, alongside domain destinations such as a scene editor or a
foreground guide. `ConversationFrontEnd` uses the typed `DecisionService` to
select among real destinations, lets an explicitly focused destination receive
the next turn first, and returns to route selection when it releases or declines.
A declined route is excluded from one reroute; uncertain or unavailable work
ends with a clarification instead of invoking a guessed destination.

The front end owns per-participant reply tasks and focus assignment. Each
destination owns its persistent application state and cleanup. `interrupt()`
cancels current work while preserving focus; `participant_left()` cancels the
turn and releases only that participant's focused destination; `stop()` drains
work and releases all focus. Output callbacks receive the destination name and
text so applications can keep speech attribution aligned with the route.

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
