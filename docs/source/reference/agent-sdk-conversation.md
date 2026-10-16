<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-conversation

`xr-ai-conversation` provides typed contracts and a participant-scoped front end
for routing turns among application destinations. It uses the typed
`DecisionService` for route selection. Destinations keep ownership of their
model calls, application state, and resources.

## Route a turn

Pass a `ConversationTurn` to `ConversationFrontEnd.handle()` with an async
callback that accepts the route name and text chunk. The front end applies
these steps:

1. It serializes turns for one participant, cancels and drains that
   participant's previous response, and bounds the completed history supplied
   to routing. It keeps the current query separate from completed history.
2. If a destination owns explicit focus, the front end checks that destination
   first. Otherwise it selects among registered destinations using their
   `RoutingAid` descriptions, eligibility, and priority.
3. The selected destination runs `admit()` before speaking or starting answer
   work. It can accept the turn, decline, ask for clarification, or release
   focus and return the same query to ordinary routing.
4. A declined route is excluded from one reroute. After selected destinations
   decline, an eligible configured default gets one answer attempt without a
   second admission check. If routing has no eligible candidate, the front end
   asks for clarification.
5. An accepted destination streams chunks through the callback, tagged with its
   registered route name. It returns an answer, keeps focus, or releases focus
   after the answer. A post-answer release consumes the current query; it does
   not send that query through routing again.

```python
from xr_ai_conversation import ConversationFrontEnd, ConversationTurn

front_end = ConversationFrontEnd(
    {"conversation": general_destination, "scene": scene_destination},
    decision_service,
)
turn = ConversationTurn(
    participant_id="participant-1",
    turn_id="turn-42",
    query="Add a blue cube.",
)
result = await front_end.handle(turn, emit=publish_chunk)
```

`ConversationExchange` records a completed user request and its user-visible
answer. The first turn for a participant uses the supplied `turn.history` as its
initial history. Later turns use history retained by the front end. Only whole
recent exchanges that fit `context_budget_bytes` are passed to route decisions;
the budget counts the serialized exchange and time label in UTF-8 bytes. An
oversized exchange is dropped whole.

`RoutingAid` is a concise selection hint. Eligibility is checked by the front
end, while each destination's `admit()` method remains responsible for deciding
whether it can accept the specific turn. Destinations decline or clarify before
answer work begins. A clarification returned after answer work starts is an
error.

## Focus, cancellation, and departure

Returning `DestinationResult("focus")` makes that destination the participant's
first route for later turns. It remains focused across answers until it returns
`DestinationResult("release")`, its admission returns `Admission("release")`,
it becomes ineligible, or the participant leaves. An admission release clears
focus and routes the same query again. A post-answer release clears focus after
consuming the query.

`interrupt(participant_id)` cancels and drains current response work while
preserving completed history and focus. Omitting the participant cancels all
active turns. Output already delivered to the callback cannot be retracted.
`participant_left()` clears that participant's history and asks every
destination used by that participant to release its state. A destination may
provide `participant_left()` for stronger departure cleanup; otherwise its
`release()` method is used. Failed releases remain tracked for a later retry.
`stop()` interrupts all active turns and attempts departure cleanup for every
tracked participant.

## Optional progress acknowledgment

`acknowledgment_llm` and `acknowledgment_delay_secs` opt into a brief
route-aware acknowledgment. Configure both together. The acknowledgment client
should default to non-reasoning generation because per-call settings cannot
override model-level behavior for every provider. A positive delay controls the
fallback threshold. Measured first-answer examples can predict a slower response
and start acknowledgment generation earlier, while destination work starts
immediately. The first nonblank answer chunk cancels acknowledgment that has
not yet been emitted. This cannot retract text already handed to the output
callback or audio already queued by the application.

The timing sample begins after destination admission and ends when the
destination produces its first nonblank text. It excludes route selection,
admission, acknowledgment-output serialization, and downstream playback.
Admission preambles suppress the optional acknowledgment for that turn.

## Package files

| File | Responsibility |
|---|---|
| `__init__.py` | Lists the routing and conversation names exported by `xr_ai_conversation` |
| `contracts.py` | Defines route metadata, turns, context text, admission results, and the destination protocol |
| `front_end.py` | Coordinates selection, admission, focus, streamed answers, history, cancellation, and cleanup |
| `_history.py` | Shares private exchange timestamps and their display format |
| `conversation.py` | Implements the application-instructed conversation destination and optional current-view tools |
| `_context.py` | Stores participant/source context and performs bounded background compaction |
| `_topics.py` | Binds configured runtime topics to participant-scoped context updates |
| `prompts/core.txt` | Provides the default conversation system instructions |

The front end and conversation engine do not depend on a voice transport or a particular application.
An application adapter translates its incoming request into a
`ConversationTurn` and connects the callback to its output path.

## Python API reference

The generated {doc}`Python reference </reference/python/xr_ai_conversation/index>`
documents the exported names and their signatures, fields, and behavior.

## Streamed conversation and visual questions

`Conversation` implements `ConversationDestination` for application-instructed text
conversation. It keeps model instructions and participant state in the conversation
engine while the front end continues to own turn selection, admission, focus,
stream attribution, and cancellation. Applications can configure an optional
current-frame and streaming visual-query tool pair. When configured, the engine
can answer present-view questions through those typed tools; it does not require a
voice transport.

```python
from xr_ai_conversation import Conversation, ConversationFrontEnd

conversation = Conversation(
    llm,
    instructions="Answer general questions and use current-view tools when needed.",
    frames=current_frame_tool,
    vision=streaming_image_query_tool,
)
front_end = ConversationFrontEnd({"conversation": conversation}, decisions)
```

The visual tools must be configured together. `Conversation` invokes the current
frame tool and supplies the returned image reference to the streaming visual
query tool. The current query and resolved visual referents are sent to that
query; application-specific destinations remain responsible for capabilities
outside the conversation engine.

## Named context topics and history budgets

`Conversation` can subscribe to named runtime topics that carry source-owned text.
The constructor fixes each topic's update policy; `ConversationContextUpdate`
contains only the non-empty `text` payload.

```python
conversation = Conversation(
    llm,
    summary_context_topics=("sample.monitor-window",),
    incremental_context_topics=("sample.notes",),
)
context_agent = conversation.context_subscriber()
```

Register the returned context agent and the conversation's runtime consumer with
the same `AgentRuntime` before it starts. The application owns registration and
shutdown; close the conversation when its owner shuts down. Each configured topic
is a separate participant-scoped source. A summary replaces that source's prior
section, while incremental text appends to its source and can be compacted in the
background. Updating one source does not replace another source.

Background context is bounded by `context_budget_bytes` after UTF-8 rendering.
Incremental input to the compactor is separately bounded by
`compaction_input_budget_bytes`. An oversized summary clears its source rather
than retaining stale information; an oversized incremental update is dropped
whole. Compaction keeps source sections independent, and an oversized compacted
summary removes only the affected source. Completed exchanges use the front-end
history budget and remain separate from background context.
