<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-routing

`xr-ai-routing` provides typed contracts and a participant-scoped router
for routing turns among application destinations. Each registered destination
implements the public `RoutableAgent` protocol. It uses the typed
`DecisionService` for route selection. Destinations keep ownership of their
model calls, application state, and resources.

## Route a turn

Configure `AgentRouter(output=...)` with an async callback for router-owned
admission, clarification, and acknowledgment text. The callback receives the
current `AgentTurn`, route name, and text, so it can preserve participant and
response identity when publishing. Pass each request's `AgentTurn` to `handle()`. The router applies these steps:

1. It serializes turns for one participant, cancels and drains that
   participant's previous response, and bounds the completed history supplied
   to routing. It keeps the current query separate from completed history.
2. If a destination owns explicit focus, the router checks that destination
   first. Otherwise it selects among registered destinations using their
   `RoutingAid` descriptions, eligibility, and priority.
3. The selected destination runs `admit()` before speaking or starting answer
   work. It can accept the turn, decline, ask for clarification, or release
   focus and return the same query to ordinary routing.
4. A declined route is excluded from one reroute. After selected destinations
   decline, an eligible configured default gets one answer attempt without a
   second admission check, provided it did not already decline this turn.
   If the routing model chooses `clarify`, the router tries the highest-scoring
   eligible destination through normal admission when usable scores are
   available. Equal scores preserve candidate registration order. No eligible
   candidate or usable scores produces clarification. A destination can still
   explicitly request clarification during admission; that ends routing.
5. An accepted destination delivers its output directly. It returns an answer,
   keeps focus, or releases focus after the answer. A post-answer release consumes the current query; it does
   not send that query through routing again.

```python
from xr_ai_routing import AgentRouter, AgentTurn

router = AgentRouter(
    {"conversation": general_destination, "scene": scene_destination},
    decision_service,
    output=publish_router_control,
)
turn = AgentTurn(
    participant_id="participant-1",
    turn_id="turn-42",
    query="Add a blue cube.",
)
result = await router.handle(turn)
```

`AgentExchange` records a completed user request and its destination-provided
response context. The first turn for a participant uses the supplied `turn.history` as its
initial history once. Later turns use history retained by the router. Only whole
recent exchanges that fit `context_budget_bytes` are retained and passed to route decisions;
the budget counts the serialized exchange and time label in UTF-8 bytes. An
oversized exchange is dropped whole, even when that leaves history empty; later
caller-supplied history does not reseed that participant. The budget covers
serialized completed-history entries, not the current request, destination
descriptions, or the backend's total prompt/token limit. Configure it to suit the
selected endpoint. Decision-service errors propagate to the caller; the router
does not replace a failed routing decision with an unapproved default action.

The router keeps each retained exchange's timestamp and private identity
aligned as history is trimmed. When an admission release emits text, it records
that text as a completed exchange before rerouting the same query, so the next
destination receives the updated history.

`current_turn_context()` returns an immutable snapshot for code running inside
the active routed turn. Its `exchange_id` identifies the current exchange;
`history_exchange_ids` and `history_timestamps` align with the bounded history.
Timestamps are Unix microseconds, with `None` for an unknown time. The snapshot
stays unchanged as routing advances. Outside a routed turn, the identifier is
`None` and both history tuples are empty. These opaque identifiers can be used
to correlate work with the exchanges visible to the current destination; they
do not expose or modify router state.

`RoutingAid` is a concise selection hint. Eligibility is checked by the router,
while each destination's `admit()` method remains responsible for deciding
whether it can accept the specific turn. Destinations decline or clarify before
answer work begins. A clarification returned after answer work starts is an
error.

## Response progress and routing context

Destinations can call `notify_response_started()` immediately before their
first meaningful output or application action. The notification contains no
response text. Destinations publish their output directly. It suppresses a pending optional acknowledgment and
supplies the first-response timing sample if the answer completes successfully.
It does not deliver output, complete the turn, change focus, or add history.
Return a `DestinationResult` for completion and focus; exceptions still report
failure to the caller.

The notification applies only during the current destination's `answer()` call,
including its child-task contexts. The first notification returns `True`;
duplicates and calls outside an active answer return `False`. A retained
context from a completed or interrupted answer cannot notify a later turn,
including after departure and rejoin. Destinations own and drain their child
tasks before returning. Call the function from the answer's event loop.

`DestinationResult.text` remains optional response context for routing history.
The router retains supplied text without delivering it as output. A destination
can return a concise description of an application action, a completed text
answer, or no context. Existing history budgets and route-selection behavior
apply to any supplied context.

`Conversation.answer()` notifies before publishing its first nonblank ready
chunk to its configured voice output topic. Conversation owns publication and
stream framing. Returning `DestinationResult.text` preserves
routing history; it never substitutes for direct output or explicit per-agent
`routing_aid(participant_id)` metadata. Dynamic routing aids are refreshed for
selection on each turn.

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
`participant_left()` and `stop()` each own the whole cleanup operation, not only
the caller's wait. Concurrent callers join the same operation. Canceling a caller
does not cancel cleanup; the caller's cancellation propagates after cleanup
finishes. If cleanup also fails, that failure remains the cancellation's cause.
Failed departure keeps only that participant closed until a successful
`participant_left()` retry. Other participants can continue.

`stop()` closes global admission, interrupts responses and joins departure
cleanup for every tracked participant, including history-only participants.
Failed shutdown keeps admission closed until a successful `stop()` retry. A
successful shutdown clears participant state and permits a fresh later turn.
The application owns these calls. Destination and output callbacks, including
their child tasks, must not await this router's turn or lifecycle operations:
doing so would wait for their own owned work or create a cleanup wait cycle.
Such recursive calls raise `RuntimeError`.

## Optional progress acknowledgment

`acknowledgment_llm` and `acknowledgment_delay_secs` opt into a brief
route-aware acknowledgment. Configure both together. The acknowledgment client
should default to non-reasoning generation because per-call settings cannot
override model-level behavior for every provider. A positive delay controls the
fallback threshold. Measured first-answer examples can predict a slower response
and start acknowledgment generation earlier, while destination work starts
immediately. An explicit response-start notification cancels acknowledgment that has not yet been emitted. This cannot
retract text already handed to the output callback or audio already queued by
the application.

The timing sample begins after destination admission and ends when the
destination signals its first meaningful response. It excludes route selection,
admission, acknowledgment-output serialization, and downstream playback.
Admission preambles suppress the optional acknowledgment for that turn.
The `conversation.response-started` trace records that notification. A notification does
not establish that output has been accepted or played by a downstream transport.
If prediction or acknowledgment generation fails, the optional task logs the
failure once and ends; the destination answer continues. The decision service
does not define a separate expected-prediction-failure category, so errors are
not silently converted into a delay fallback. Cancellation drains actual owned
work before replacement or resource cleanup; an operation that does not finish
its cancellation cleanup can delay the next turn, rather than being detached
behind an arbitrary timeout.

The routing engine does not depend on a voice transport or a particular
application. Configure its control output once, then call `handle()` with
`AgentTurn` values.

## Python API reference

The generated {doc}`Python reference </reference/python/xr_ai_routing/index>`
documents the exported names and their signatures, fields, and behavior.
