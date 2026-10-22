<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-conversation

`xr-ai-conversation` provides an application-instructed
conversation agent with participant-scoped history and named
background-context sources. It does not contain turn selection, focus
management, or tools; routing contracts are provided by
[`xr-ai-routing`](agent-sdk-routing.md), and other capabilities, such as
questions about the camera view, belong in separate routed destinations.

## Compose a conversation

`Conversation` implements `xr_ai_routing.RoutableAgent`, so it can be passed to
`AgentRouter` alongside application-specific agents.

```python
from xr_ai_conversation import Conversation
from xr_ai_routing import AgentRouter, AgentTurn
from xr_ai_runtime import AgentRuntime
from xr_ai_voice import VOICE_OUTPUT_TOPIC

runtime = AgentRuntime()

conversation = Conversation(
    llm,
    instructions="Answer general questions about the lab.",
    runtime=runtime,
    output_topic=VOICE_OUTPUT_TOPIC,
)
front_end = AgentRouter({"conversation": conversation}, decisions, output=publish_router_control)
async with runtime:
    try:
        result = await front_end.handle(
            AgentTurn("participant-1", "turn-1", "When is the calibration meeting?"),
        )
    finally:
        await front_end.stop()
await conversation.close()
```

The agent answers from its instructions, completed history, and configured
context sources. Install `xr-ai-conversation[voice]` and configure the runtime
and a typed voice output topic to use it as a routed destination. Conversation
publishes its own response chunks with its routing name as source and the turn's
response identity and timestamp. Its first chunk interrupts older speech
unless router text such as a preamble already started the turn. It finalizes
partial responses after model failure; the router reports a failure before the
first chunk.
Cancellation drains the model without publishing further frames. Model and
publication failures propagate to the caller.

The application starts the runtime before answering and stops it after draining
routed work. `answer(turn)` is the public request handler; streaming model
iteration is private to the conversation.

## Named context topics

The constructor accepts topic names in separate summary and incremental lists.
The topic determines how its text updates that source; the payload contains only
`ConversationContextUpdate.text`.

```python
conversation = Conversation(
    llm,
    runtime=runtime,
    output_topic=VOICE_OUTPUT_TOPIC,
    summary_context_topics=("sample.monitor-window",),
    incremental_context_topics=("sample.notes",),
)
context_agent = conversation.context_subscriber()
```

Register the returned context agent with the same `AgentRuntime` that delivers
updates, before runtime start. The application owns that registration and must
close the conversation during its shutdown. Each topic is an independent,
participant-scoped source. A summary topic's update replaces its current state;
incremental text appends and is compacted in the background. A summary update
does not clear another topic, even when both topics publish for the same
participant.

## Context compaction

Dialogue and each incremental topic use the same rolling summary. Entries
accumulate verbatim; when they grow large enough, the model merges the
previous summary with the oldest entries into one new summary, and those
entries are discarded. Only the summary and the entries added since it was
written are kept, so text grows with the conversation only until the next
compaction. A short identity of every exchange is kept for the session so that
router history is not replayed.

Each section of the rendered context has its own budget, in UTF-8 bytes of
rendered text rather than model tokens:

- `dialogue_budget_bytes` (default 8,000) bounds completed dialogue.
- `topic_budget_bytes` bounds each topic: one value for every topic (default
  4,000), or a mapping with a value for each configured topic, for example
  `{"sample.monitor-window": 2_000, "sample.notes": 6_000}`.
- A section compacts when its uncompacted entries, measured as rendered,
  exceed half of its budget. Its summary is limited to a quarter of its budget.
- `compaction_input_budget_bytes` (default 12,000) bounds the previous summary
  plus the entries sent in one compaction request. It must be at least half
  the largest section budget plus 256 bytes. A larger backlog is compacted
  oldest first in several requests.
- The newest dialogue exchange always stays verbatim. Compaction runs in the
  background and never delays an answer.
- A summary topic is not compacted: each update replaces its state.

| Situation | Handling |
|---|---|
| The model returns a summary over its limit | One request asks the model to shorten it; if that is cut off or still too long, the summary keeps its start and end within the limit. |
| The model returns an empty summary, a response cut off at its output limit, or an error | The entries are kept and compaction is retried after the section's next entry. |
| One exchange or update exceeds a compaction request | The request keeps its start and end with an omission marker, so later entries still compact. |
| A section exceeds its budget while compaction is pending | The prompt leaves out that section's oldest uncompacted entries, keeping its newest, and logs a warning; they stay stored until compacted. One very large exchange or update keeps its start and end. Other sections are unaffected. |
| A summary-topic update exceeds its topic budget | The update keeps its start and end within the budget. |
| A front-end history window still contains compacted exchanges | Every exchange identity is remembered for the session, so those exchanges are not added again. |

The current request is never clipped. Compaction quality depends on the
model: a summary can omit details, so applications that must keep exact
records should publish them as a summary topic. Compaction requests use the
model's configured defaults; for reasoning models, disable thinking in the
model configuration, as the model presets do.

### Prompt layout

An answer request has three messages:

1. The system prompt with core and application instructions.
2. The context: summaries of earlier material first, then uncompacted
   exchanges, topic updates, and summary-topic states in recorded-time order,
   and the current time last.
3. The current request.

Each section compacts on its own schedule, so a summary can cover a time span
that overlaps an uncompacted entry from another section, for example a
dialogue summary covering 14:13 to 14:43 and a topic update published at
14:18. The order then cannot show which fact is newer; the model resolves such
a conflict from the times shown and can get it wrong.

The system prompt contains no clock, so it stays an identical prefix across
requests.

## Lifecycle

`release()` keeps completed participant context, as the routing contract
requires; the router calls it when focus moves to another destination.
`participant_left()` cancels the participant's compaction jobs, waits for them,
and forgets its context; `close()` does the same for every participant and
stops new answers. Other participants are unaffected by one participant's
departure. The router calls `participant_left()` only on destinations the
participant was routed to, so when topics publish for a participant who never
speaks to this conversation, the application calls
`conversation.participant_left()` itself after `router.participant_left()` or
`router.stop()` returns.

An answer still streaming when its participant leaves is not stopped. When it
completes, it records its exchange as the start of new context for that
participant. Close the conversation before closing the shared model client,
and drain caller-owned answer streams separately.

## Evaluations

`agent-sdk/xr-ai-conversation/eval` contains live-model evals: stale-summary
conflicts across sections, and long sessions with corrections and oversized
exchanges and updates. Its README describes how to run them.

## Package contents

| Module | Responsibility |
|---|---|
| `_conversation.py` | Implements the conversation engine and builds answer requests |
| `_context_update.py` | Defines the text-only context update payload |
| `_context.py` | Stores participant dialogue and source context; performs rolling-summary compaction |
| `_topics.py` | Adapts configured runtime topics to participant-scoped context updates |
| `prompts/core.txt` | Supplies the default conversation instructions |

The generated {doc}`Python reference </reference/python/xr_ai_conversation/index>`
documents the two package exports and their signatures and behavior.
