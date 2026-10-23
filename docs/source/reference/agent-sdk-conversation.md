<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-conversation

`xr-ai-conversation` provides an application-instructed
conversation agent and a `ContextManager` that holds participant-scoped
history and named background-context sources for it and for any other agent.
It does not contain turn selection, focus
management, or tools; routing contracts are provided by
[`xr-ai-routing`](agent-sdk-routing.md), and other capabilities, such as
questions about the camera view, belong in separate routed destinations.

## Compose a conversation

`Conversation` implements `xr_ai_routing.RoutableAgent`, so it can be passed to
`AgentRouter` alongside application-specific agents.

```python
from xr_ai_conversation import ContextManager, Conversation
from xr_ai_routing import AgentRouter, AgentTurn
from xr_ai_runtime import AgentRuntime
from xr_ai_voice import VOICE_OUTPUT_TOPIC

runtime = AgentRuntime()

context = ContextManager(llm)
conversation = Conversation(
    llm,
    context=context,
    instructions="Answer general questions about the lab.",
    runtime=runtime,
    output_topic=VOICE_OUTPUT_TOPIC,
)
front_end = AgentRouter(
    {"conversation": conversation},
    decisions,
    output=publish_router_control,
    cleanup_participant=context.clear,
)
async with runtime:
    try:
        result = await front_end.handle(
            AgentTurn("participant-1", "turn-1", "When is the calibration meeting?"),
        )
    finally:
        await front_end.stop()
await conversation.close()
await context.close()
```

`cleanup_participant=context.clear` is what forgets a participant's context when they
leave. The conversation does not clear it, and without this wiring a departed
participant's dialogue and summaries stay in the manager until `close()`, and a reused
participant id inherits them.

The agent answers from its prompt, completed history, and the context
manager's sources. Configure the runtime and a `ResponseOutput` topic, such as
`VOICE_OUTPUT_TOPIC` or a text front end's topic, to use it as a routed
destination. Conversation
publishes its own response chunks as a reply to the turn ID with its routing name
as source and the turn's timestamp. The voice layer plays them after router text
such as a preamble for the same turn. It finalizes
partial responses after model failure; the router reports a failure before the
first chunk.
Cancellation drains the model without publishing further response text and
finalizes an already-open stream with an empty frame. Model and publication
failures propagate to the caller.

The application starts the runtime before answering and stops it after draining
routed work. `answer(turn, started=...)` is the public request handler; streaming model
iteration is private to the conversation.

## Prompts

A request's system prompt is the conversation's `system_prompt` followed by
its `instructions`. The default system prompt describes a general assistant
that holds a conversation and answers from its own knowledge as well as from
the context. It makes no claim about tools or the camera view; an application
that needs such a policy states it in `instructions`. Pass `system_prompt` to
replace the default entirely. The compaction instructions are separate and
belong to the context manager; see [Context compaction](#context-compaction).

## The context manager

`Conversation` neither creates nor owns its `ContextManager`; the application
passes one in, and several agents can share it. The manager keeps what each
participant said, the answers they got, and the named sources applications
publish. The conversation records each completed exchange in it and renders it
into every request.

The manager's constructor accepts topic names in separate summary and
incremental lists. The topic determines how its text updates that source; the
payload contains only `ConversationContextUpdate.text`.

```python
context = ContextManager(
    llm,
    summary_context_topics=("sample.monitor-window",),
    incremental_context_topics=("sample.notes",),
)
runtime.register("context", context.subscriber())
```

Register the subscriber with the same `AgentRuntime` that delivers updates,
before runtime start. Each topic is an independent, participant-scoped source.
A summary topic's update replaces its current state; incremental text appends
and is compacted in the background. A summary update does not clear another
topic, even when both topics publish for the same participant. An agent other
than `Conversation` reads the same context with `context.blocks(participant_id)`
and writes it with `record_exchange()` or `update()`. Let one router-driven writer
record exchanges per manager: history reconciliation matches only a contiguous
tail, so two writers whose exchanges interleave in the router's history are
recorded again on later turns.

## Context compaction

Dialogue and each incremental topic use the same rolling summary. Entries
accumulate verbatim; when they grow large enough, the model merges the
previous summary with the oldest entries into one new summary, and those
entries are discarded. Only the summary and the entries added since it was
written are kept, so text grows with the conversation only until the next
compaction. A short identity of every exchange is kept for the session so that
router history is not replayed.

Each section of the rendered context has its own budget. Agents that share a
manager share its budgets, which are set where the manager is created, in UTF-8 bytes of
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

`CompactionPrompts` replaces what the model is told to keep when it compacts
dialogue, an incremental source, or an over-long summary. The manager adds the
size limit and the plain-text output format to every prompt, so custom prompts
do not state them.

The current request is never clipped. Compaction quality depends on the
model: a summary can omit details, so applications that must keep exact
records should publish them as a summary topic. Compaction requests use the
model's configured defaults; for reasoning models, disable thinking in the
model configuration, as the model presets do.

### Prompt layout

An answer request has three messages:

1. The system prompt followed by the application instructions.
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

The application owns the context manager's lifecycle. `release()` and
`participant_left()` on the conversation leave participant context in place, so
a shared manager is not cleared by one agent. Wire `context.clear` as the
router's `cleanup_participant`, as in the example above. The router runs it after
it has interrupted and drained the participant's in-flight work, so it is the one
ordering in which a streaming answer cannot record into the cleared context. It
cancels the participant's compaction jobs, waits for them, and forgets the
participant's context; it is idempotent, and other participants are unaffected.
`context.close()` does the same for every participant and drops later updates,
exchanges, and history. `conversation.close()` only stops new answers.

An answer still streaming when its participant is cleared outside the router's
cleanup is not stopped, and on completion it records its exchange as the start of
new context. After `context.close()` the exchange is dropped. Compaction tasks
call the model, so close the context manager before closing the shared model
client, and drain in-flight answers separately.

## Evaluations

`agent-sdk/xr-ai-conversation/eval` contains live-model evals: stale-summary
conflicts across sections, and long sessions with corrections and oversized
exchanges and updates. Its README describes how to run them.

## Package contents

| Module | Responsibility |
|---|---|
| `_conversation.py` | Implements the conversation engine and builds answer requests |
| `_manager.py` | Public `ContextManager`: budgets, rendering, compaction requests, and prompts |
| `_context_update.py` | Defines the text-only context update payload |
| `_context.py` | Stores participant dialogue and source context; runs the rolling-summary state machine |
| `_topics.py` | Adapts configured runtime topics to participant-scoped context updates |
| `prompts/core.txt` | Supplies the default system prompt |

The generated {doc}`Python reference </reference/python/xr_ai_conversation/index>`
documents the package exports and their signatures and behavior.
