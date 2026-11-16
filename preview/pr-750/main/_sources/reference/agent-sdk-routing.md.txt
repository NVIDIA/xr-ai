<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-routing

## Overview

Route user turns among specialized agents for conversation, scene inspection,
or note-taking. Routing lets those agents share one interaction and keep a
task in focus, so a multi-step activity can continue across user turns.

## Route a turn

Each registered destination implements the public `RoutableAgent` protocol;
the router uses the typed `DecisionService` for route selection.

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
descriptions, or the backend's total prompt/token limit. The decision service owns
that limit, so the router does not size requests against it. When the service
rejects a routing request with `DecisionInputTooLargeError`, the router retries with
the newest half of the history, down to none, and logs a warning and a
`conversation.destination-route-history-trimmed` Relay event for each retry.
Only the routing request is trimmed; destinations still receive the router's
retained history. A participant whose history stays over the limit pays these
retries on each routed turn, up to log2(n) + 1 decision calls for n exchanges;
lower `context_budget_bytes` if the warning recurs. If the request is still too
large with no history, the routing prompt, destination descriptions, and current
request alone exceed the limit, and the error propagates. Other
decision-service errors also propagate to the caller; the router does not
replace a failed routing decision with an unapproved default action.

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

The router calls `answer(turn, started=...)`. Destinations await `started()`
immediately before their first meaningful output or application action. The
signal contains no response text; destinations publish their output directly.
It suppresses a pending optional acknowledgment and supplies the first-response
timing sample if the answer completes successfully. It does not deliver output,
complete the turn, change focus, or add history. Return a `DestinationResult`
for completion and focus; exceptions still report failure to the caller. A turn
that fails before its destination calls `started()` sends one brief failure
message through the router's output; after that call, the destination owns any
failure output. Callers should not publish another.

If an acknowledgment is already committed, `started()` waits until it is
delivered, so direct answer output never precedes it. `started()` returns
`None`; destinations do not choose whether their output interrupts. Router
control text and destination replies share the turn ID as their request ID, so
the voice layer plays them in order. Only the first call changes router state.
After `answer()` returns or is interrupted, the signal has no effect.

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

## Register the router with a runtime

Install `xr-ai-routing[runtime]` to use the typed `xr_ai_interaction` messages.
The direct `handle()` path uses its configured control callback and imports
neither the runtime nor those messages. Runtime queries need no callback. Configure the query and
output topics explicitly, then register the router's runtime subscriber:

```python
from xr_ai_interaction import ResponseOutput, UserQuery
from xr_ai_runtime import AgentRuntime, Topic

queries = Topic("my-app.query", UserQuery)
outputs = Topic("my-app.output", ResponseOutput)
runtime = AgentRuntime()
router = AgentRouter(
    {"conversation": conversation, "scene": scene_destination},
    decision_service,
    query_topic=queries,
    output_topic=outputs,
)
runtime.register("router", router.runtime_subscriber())
```

Use the same query topic on the input producer and register a consumer for the
chosen output topic. With `VoiceAgent`, use its `VOICE_OUTPUT_TOPIC` or an
existing aggregation input topic for output. Destinations publish their answer
streams independently under their own producer identity and close their own
streams. `AgentTurn.turn_id` carries the input publication identity;
`timestamp_us` carries its originating input time. The router publishes finite
control messages under its own producer identity, so an acknowledgment never
holds an open stream ahead of a direct answer.

Optional join, leave, and interruption topics must also be supplied explicitly.
Each uses the corresponding `ParticipantJoined`, `ParticipantLeft`,
or `Interrupted` model. No query or lifecycle topics are subscribed by
default. Incorrect payload classes and incomplete topic pairs fail during
construction. A leave topic requires a join topic, because only a join readmits
a participant who left.

Install `xr-ai-routing[runtime]` and supply explicit query and output topics to
use runtime delivery; optional participant and interruption topics also require
explicit configuration. Runtime queries use the configured output topic for
router-owned control text; direct `handle(turn)` calls use the constructor
`output` callback.

The runtime delivers queries concurrently; the router owns its detached query
and destination tasks. Initialization runs once before the participant's first
query, including when no join event preceded it; an interrupted initialization
runs again on the next turn. Application initialization and idempotent cleanup
callbacks own application resources that exist before a route is used. Runtime
turns carry no caller-supplied history, and these callbacks cannot seed routing
history. Keep conversation context subscribers registered separately. A
participant's initialization or cleanup does not block another participant.
Leave closes admission before cleanup; late queries cannot recreate state, and a
new join reopens admission after cleanup, first retrying a departure whose
cleanup failed. A runtime query refused because its participant left or the
router is stopping is logged and dropped. `set_status` reports `processing` and
`idle` around each runtime turn; its failures are logged and never fail the turn.
Interruption retains
completed history and focus. Await `router.stop()` before runtime shutdown and
resource closure. Successful stop permits a fresh later lifecycle.

Runtime turns use the same failure rule as `handle()`: a turn that fails before
its destination calls `started()` gets one brief failure message through the
router's output, unless that output itself failed. After `started()`, the
destination owns failure output: preserve a partial answer without adding a
duplicate fallback, and finalize any open stream on failure. Cancellation drains owned work and
closes an open stream without adding text. Optional progress notifications
can describe nonvoice actions and are never treated as evidence of voice
publication.

## Focus, cancellation, and departure

Returning `DestinationResult("focus")` makes that destination the participant's
first route for later turns. It remains focused across answers until it returns
`DestinationResult("release")`, its admission returns `Admission("release")`,
it becomes ineligible, another destination returns `DestinationResult("focus")`,
or the participant leaves. An admission release clears focus and routes the
same query again. A post-answer release clears focus after consuming the query.

A focused destination can return `Admission("decline")` to let another destination
handle the current query, then receive the next turn again. For example, a
foreground note-taking agent can decline a general question without ending its
note session. The router keeps its focus and does not call its `release()`.
The declined destination is excluded from selection and default fallback for
the rest of this turn, within the existing bounded routing attempts.

An ordinary answer, clarification, failure, or cancellation during that turn
retains the original focus. If another destination explicitly returns
`DestinationResult("focus")`, the router releases the previous owner and gives
focus to that destination instead. This keeps a single foreground owner;
it does not stack suspended destinations. Returning `Admission("release")`
explicitly ends focus and reroutes the current query as before.

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
`participant_left()` retry. With runtime topics, a participant who left also
needs a join event, which retries any failed cleanup itself. Other participants
can continue.

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
admission, waiting for committed acknowledgment delivery, and downstream playback.
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

Without runtime topics, routing imports neither the agent runtime nor voice
packages and requires no particular application. Configure its control output once, then call `handle()`
with `AgentTurn` values.

## Python API reference

The generated {doc}`Python reference </reference/python/xr_ai_routing/index>`
documents the exported names and their signatures, fields, and behavior.
