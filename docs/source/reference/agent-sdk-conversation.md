<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-conversation

`xr-ai-conversation` provides a streamed conversation destination with optional
current-view visual question answering and a typed front end for routing to
application destinations. Its key integrations are `xr_ai_models`,
`xr_ai_runtime`, `xr_ai_hub_client`, native `xr_ai_tools`, Pydantic, and
NeMo Relay; it adds no model client or transport.

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

## Named context topics and history budgets

Configure the names of the runtime topics a conversation listens to with
``context_topics``. Topic names identify source sections; they do not prescribe
an update mode. Each publisher chooses ``incremental`` or ``summary`` on each
`ConversationContextUpdate`. Call ``conversation.context_subscriber()`` to
create the typed runtime subscriber, then register that Agent with the same
`AgentRuntime` before starting it. Runtime metadata supplies participant
identity; unscoped updates are dropped. Incremental source updates are
compacted asynchronously and independently per participant and source. A
fitting summary replaces that source's active background and cancels older
compaction work. If a summary is too large for the context budget, the source's
prior section is cleared; oversized incremental updates are dropped whole.
Compaction also drops an oversized summary for that source. After a
participant leaves, updates for that session are dropped. Call
`conversation.activate_participant(participant_id)` from the participant-joined
lifecycle event before publishing that participant's context. The simple VLM
sample wires the existing `VoiceParticipantJoined` topic for this purpose;
selecting a `Conversation` destination through admission or calling
`Conversation.stream()` also activates its participant state.

`context_budget_bytes` and `compaction_input_budget_bytes` bound UTF-8 payload
bytes, not model tokens; their 12,000-byte defaults reuse the former harness's
12,000-character context policy as a conservative byte limit. The front end
and sample retain the complete transcript. When completed dialogue exceeds the
context budget, `Conversation` starts a background compaction of an older whole
prefix using a dialogue-specific prompt. It keeps the raw exchanges and sends
the generated summary with newer complete exchanges after the prefix watermark.
Until that summary is ready, a request uses the bounded recent transcript and
may omit older details. Routing and admission continue to receive their existing
bounded recent-history snapshot; they do not receive this dialogue summary.

`Conversation` keeps the current query separate from the context it renders.
When rendered context exceeds its budget, it drops the oldest whole context
items, logs the drop, and continues. It never clips an exchange or the current
request. Call `participant_left()` and `close()` at lifecycle boundaries so
context tasks cannot outlive their participant or conversation.
The budgets cover prior context text (including rendered section headings for
conversation messages), not system/application instructions, tool definitions,
transport framing, or the current request; they are not a model-window guarantee.

```python
from xr_ai_conversation import ConversationContextUpdate
from xr_ai_runtime import Topic

conversation = Conversation(llm, context_topics=("tea.status",))
topic = Topic("tea.status", ConversationContextUpdate)
runtime.register("conversation-context", conversation.context_subscriber())
await runtime.publish(
    topic,
    ConversationContextUpdate(mode="summary", text="The timer is at six minutes."),
    participant_id="participant-1",
)
```

## Multi-destination routing

Each destination declares a `RoutingAid`, checks a `ConversationTurn` with
`admit()` before speaking or starting work, and answers with source-tagged
stream chunks. `Conversation` implements that same contract for general
conversation, alongside domain destinations such as a scene editor or a
foreground guide. `ConversationFrontEnd` uses the typed `DecisionService` to
select among real destinations and lets an explicitly focused destination
receive the next turn first. A declined route is excluded from one bounded
reroute. `Admission("release")`
clears that destination's focus and sends the same query back through routing.
By contrast, `DestinationResult("release")` is returned after accepted answer
work; it consumes the current query and clears focus without rerouting it. If
selected destinations decline through those attempts, the configured eligible
default receives one final answer attempt without another admission check. If
the router chooses `clarify` but returns usable scores for eligible destinations, the front end
tries the highest-scoring candidate through normal admission. Missing or
invalid eligible scores, or no eligible candidate, still produce a clarification.
Explicit destination clarifications also remain final; if the default is
unavailable after exhausted declines, the front end asks for clarification.
Routing descriptions and priority are selection hints; application policy
belongs in the destination's instructions and admission check. Destinations
decline or clarify in `admit()` before work begins; after work, `answer()` returns
only `answer`, `focus`, or `release`.

The front end owns per-participant reply tasks and focus assignment. Each
destination owns its persistent application state and cleanup. `interrupt()`
cancels current work while preserving focus; `participant_left()` cancels the
turn and calls a destination's `participant_left()` hook when it has one,
otherwise `release()`. `stop()` drains work and releases all tracked participant
state. Output
callbacks receive the destination name and text so applications can keep speech
attribution aligned with the route. The first `handle()` call for a participant
uses the supplied `ConversationTurn.history` as its initial seed. Once the front
end has completed exchanges, later calls use its owned history instead of
merging newly supplied history.

```python
from xr_ai_conversation import ConversationFrontEnd

front_end = ConversationFrontEnd(
    {"conversation": conversation, "scene": scene_destination, "guide": guide_destination},
    decision_service,
)
result = await front_end.handle(turn, emit=publish_chunk)
```

## Source map

The package is under `agent-sdk/xr-ai-conversation/xr_ai_conversation/`:

| File | Responsibility |
|---|---|
| `__init__.py` | Lists the names exported from `xr_ai_conversation` |
| `contracts.py` | Defines destination protocols and typed turn, route, admission, result, and context records |
| `front_end.py` | Implements route selection, admission, focus, history, interruption, and cleanup coordination |
| `conversation.py` | Implements the reusable conversation destination and optional current-view tool path |
| `_context.py` | Owns private transcript, named-context, byte-budget, and compaction state |

The reusable routing front end has no voice or application-destination
dependency. Applications connect it to their input and output systems with a
small adapter that translates incoming requests to `ConversationTurn` and maps
the `Emit` callback to their output transport.

## Python API reference

The generated {doc}`Python reference </reference/python/xr_ai_conversation/index>`
documents the exported package names and their signatures.
