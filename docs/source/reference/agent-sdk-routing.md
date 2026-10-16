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

Pass an `AgentTurn` to `AgentRouter.handle()` with an async
callback that accepts the route name and text chunk. The router applies
these steps:

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
   second admission check. If routing has no eligible candidate, the router
   asks for clarification.
5. An accepted destination streams chunks through the callback, tagged with its
   registered route name. It returns an answer, keeps focus, or releases focus
   after the answer. A post-answer release consumes the current query; it does
   not send that query through routing again.

```python
from xr_ai_routing import AgentRouter, AgentTurn

router = AgentRouter(
    {"conversation": general_destination, "scene": scene_destination},
    decision_service,
)
turn = AgentTurn(
    participant_id="participant-1",
    turn_id="turn-42",
    query="Add a blue cube.",
)
result = await router.handle(turn, emit=publish_chunk)
```

`AgentExchange` records a completed user request and its user-visible
answer. The first turn for a participant uses the supplied `turn.history` as its
initial history. Later turns use history retained by the router. Only whole
recent exchanges that fit `context_budget_bytes` are passed to route decisions;
the budget counts the serialized exchange and time label in UTF-8 bytes. An
oversized exchange is dropped whole.

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

The router does not depend on a voice transport or a particular application.
An application adapter translates its incoming request into a
`AgentTurn` and connects the callback to its output path.

## Python API reference

The generated {doc}`Python reference </reference/python/xr_ai_routing/index>`
documents the exported names and their signatures, fields, and behavior.
