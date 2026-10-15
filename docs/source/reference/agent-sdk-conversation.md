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

## Named context topics and history budgets

Publish `ConversationContextUpdate(mode="incremental", text=...)` or
`ConversationContextUpdate(mode="summary", text=...)` on a named runtime
`Topic`. The topic name becomes the source section. Register
`conversation.context_subscriber(topics)` with the same `AgentRuntime` before
starting it. Runtime metadata supplies participant identity; unscoped updates
are dropped. Incremental source updates are compacted asynchronously and
independently per participant and source. A summary replaces that source's
active background exactly as published and cancels older compaction work.
After a participant leaves, updates for that session are dropped. Call
`conversation.activate_participant(participant_id)` from the participant-joined
lifecycle event before publishing that participant's context. The simple VLM
sample wires the existing `VoiceParticipantJoined` topic for this purpose;
`ConversationFrontEnd.handle()` also reopens a participant when a new turn
arrives.

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

topic = Topic("tea.status", ConversationContextUpdate)
runtime.register("conversation-context", conversation.context_subscriber((topic,)))
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
select among real destinations, lets an explicitly focused destination receive
the next turn first, and returns to route selection when it releases or declines.
A declined route is excluded from one bounded reroute. If routing selects no
eligible destination or those attempts end in declines, the configured eligible
default receives one final answer attempt without another admission check.
Explicit destination clarifications remain final; if the default is unavailable,
the front end asks for clarification. Routing descriptions and priority are
selection hints; application policy belongs in the destination's instructions
and admission check. Destinations decline or clarify in `admit()` before work
begins; after work, `answer()` returns only `answer`, `focus`, or `release`.

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
