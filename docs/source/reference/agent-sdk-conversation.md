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

Rendered background context is bounded by `context_budget_bytes`; compaction
input is separately bounded by `compaction_input_budget_bytes`. An oversized
summary clears its source, an oversized incremental record is dropped whole,
and an oversized compacted summary removes only the affected source. Completed
conversation history remains separate from background context. These budgets
bound rendered context and compaction inputs, not total transcript storage:
the exact dialogue transcript is retained until participant release, departure,
or conversation close.

## Package contents

| Module | Responsibility |
|---|---|
| `_conversation.py` | Implements the conversation engine and optional current-view tools |
| `_context_update.py` | Defines the text-only context update payload |
| `_context.py` | Stores participant dialogue and source context; performs bounded compaction |
| `_topics.py` | Adapts configured runtime topics to participant-scoped context updates |
| `prompts/core.txt` | Supplies the default conversation instructions |

The generated {doc}`Python reference </reference/python/xr_ai_conversation/index>`
documents the two package exports and their signatures and behavior.
