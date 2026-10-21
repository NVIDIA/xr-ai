<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-conversation

`xr-ai-conversation` provides an application-instructed conversation engine
with participant-scoped history and named background-context sources. It can
use the typed current-frame and visual-query tools when an application supplies
both. It does not contain turn selection or focus management; those contracts
are provided by [`xr-ai-routing`](agent-sdk-routing.md).

## Compose a conversation

`Conversation` implements `xr_ai_routing.RoutableAgent`, so it can be passed to
`AgentRouter` alongside application-specific agents.

```python
from xr_ai_conversation import Conversation
from xr_ai_routing import AgentRouter, AgentTurn

conversation = Conversation(
    llm,
    instructions="Answer general questions and use current-view tools when needed.",
    frames=current_frame_tool,
    vision=streaming_image_query_tool,
)
front_end = AgentRouter({"conversation": conversation}, decisions)
result = await front_end.handle(
    AgentTurn("participant-1", "turn-1", "What color is the sign?"),
    emit=publish_chunk,
)
```

The visual tools are optional, but must be configured together. When present,
the engine resolves the current frame and sends it with the current visual
question through the typed tool contract. Without them, it answers from text
instructions, completed history, and configured context sources.

## Named context topics

The constructor accepts topic names in separate summary and incremental lists.
The topic determines how its text updates that source; the payload contains only
`ConversationContextUpdate.text`.

```python
conversation = Conversation(
    llm,
    summary_context_topics=("sample.monitor-window",),
    incremental_context_topics=("sample.notes",),
)
context_agent = conversation.context_subscriber()
```

Register the returned context agent with the same `AgentRuntime` that delivers
updates, before runtime start. The application owns that registration and must
close the conversation during its shutdown. Each topic is an independent,
participant-scoped source. A summary replaces that source; incremental text
appends and can be compacted in the background. A summary update does not clear
another topic, even when both topics publish for the same participant.

## Context compaction

Dialogue and each incremental topic use the same rolling summary. Entries
accumulate verbatim; when they grow large enough, the model merges the
previous summary with the oldest entries into one new summary, and those
entries are discarded. Only the summary and the entries added since it was
written are kept, so memory grows with the conversation only until the next
compaction.

Each section of the rendered context has its own budget, in UTF-8 bytes rather
than model tokens:

- `dialogue_budget_bytes` (default 8,000) bounds completed dialogue.
- `topic_budget_bytes` bounds each topic: one value for every topic (default
  4,000), or a mapping with a value for each configured topic, for example
  `{"sample.monitor-window": 2_000, "sample.notes": 6_000}`.
- The context of one answer request is bounded by the sum of all section
  budgets.
- A section compacts when its uncompacted entries exceed half of its budget.
  Its summary is limited to a quarter of its budget, and to half of
  `compaction_input_budget_bytes`.
- `compaction_input_budget_bytes` bounds the previous summary plus the entries
  sent in one compaction request. A larger backlog is compacted oldest first
  in several requests.
- The newest dialogue exchange always stays verbatim. Compaction runs in the
  background and never delays an answer.
- A summary topic is not compacted: each update replaces that source.

| Failure | Handling |
|---|---|
| The model returns a summary over its limit | One request asks the model to shorten it; if it still overshoots, it is truncated to the limit. A summary never blocks later compaction. |
| The model returns an empty summary or fails | The entries are kept and compaction is retried on the next turn or update. After a failure, uncompacted entries beyond eight times the trigger size are dropped, oldest first, and logged. |
| One exchange or update exceeds a compaction request | The compaction request keeps its start and end with an omission marker, so later entries still compact. |
| One exchange or source exceeds its section of a prompt | The prompt keeps its start and end with an omission marker instead of dropping the rest of the context. |
| A summary-topic update exceeds its topic budget | The update keeps its start and end within the budget; the source is cleared only if not even its label fits. |
| A front-end history window still contains compacted exchanges | Recent exchange identities are remembered, so those exchanges are not added again. |
| Context still exceeds the request budget | As a last resort, the oldest newer material is dropped from the prompt (keeping the newest exchange), then source summaries, then the conversation summary, and each drop is logged. |

The current request is never clipped. Compaction quality depends on the
model: a summary can omit details, so applications that must keep exact
records should publish them as a summary topic.

### Thinking during compaction

Thinking is chosen per request. Answers use the model's configured default
(the model presets disable thinking), and `compaction_thinking_budget` lets
only background compaction requests think, with that many reasoning tokens.
Because some reasoning models overrun a thinking budget, those requests are
also capped at the budget plus the summary limit in output tokens; a request
cut off by the cap yields an empty summary and is retried later. Nemotron 3
Nano Omni does not hold to the budget: with a 512- or 2,048-token budget most
compaction requests were cut off before writing a summary, so leave the option
unset for that model.

### Prompt layout

An answer request is ordered from oldest to newest material, so a correction
always appears after the statement it supersedes:

1. The system prompt with core and application instructions. It contains no
   clock, so it stays an identical prefix across requests.
2. One context message: summaries of older material (the earlier
   conversation with this user, then each incremental topic's earlier
   updates), followed by newer material — uncompacted exchanges, topic
   updates, and summary-topic states — in recorded-time order, and the current
   time.
3. The current request.

A summary always lags behind material that arrived after it was written, so
it can contradict newer material until the next compaction. In the evals
below, the previous layout, which listed recent exchanges before all
summaries and sources, answered with the stale fact in every sample in that
situation; this order answered with the newer one.

### Background work and concurrency

Compaction runs in background tasks and never delays an answer: an answer
uses whatever summary is committed and keeps everything newer verbatim. Each
dialogue or topic section runs at most one compaction job at a time, which
continues with the next batch until the section is below its trigger. A job
commits only if the section was not replaced meanwhile (a summary-topic update
or source removal discards it), and it removes only the entries it summarized,
so entries added during the request stay. Participant departure and
conversation close cancel and join the jobs; a job's completion never affects
a later session of the same participant. The model server runs compaction
requests alongside answers; in the evals, a compaction request in flight added
about 7 ms to an answer's first-token latency.

### Evaluations

`agent-sdk/xr-ai-conversation/eval` contains live-model evals for these
behaviors: stale-summary conflicts, one long session with oversized
exchanges and updates, and compaction cost with its foreground impact. Its
README lists how to run them.

## Lifecycle

Participant departure, release, and conversation shutdown retain ownership of
compaction jobs until their cleanup finishes. Concurrent calls join that same
cleanup; cancelling a waiter does not abandon it or cancel a job's cleanup a
second time. Callers must wait for cleanup before reactivating that participant;
other participants remain independent. Close the conversation before closing
the shared model client, and drain caller-owned answer streams separately.

## Package contents

| Module | Responsibility |
|---|---|
| `_conversation.py` | Implements the conversation engine and optional current-view tools |
| `_context_update.py` | Defines the text-only context update payload |
| `_context.py` | Stores participant dialogue and source context; performs rolling-summary compaction |
| `_topics.py` | Adapts configured runtime topics to participant-scoped context updates |
| `prompts/core.txt` | Supplies the default conversation instructions |

The generated {doc}`Python reference </reference/python/xr_ai_conversation/index>`
documents the two package exports and their signatures and behavior.
