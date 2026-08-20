<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Agent SDK

The SDK is split by dependency boundary. Directory names match the Python
imports developers use; distribution names remain stable for package
installation.

| Directory | Import | Responsibility |
|---|---|---|
| `xr-ai-hub` | `xr_ai_hub` | Minimal msgpack/ZMQ IPC with XR-Media-Hub |
| `xr-ai-models` | `xr_ai_models` | Typed model protocols, profiles, and OpenAI-compatible clients |
| `xr-ai-runtime` | `xr_ai_runtime` | Agent registration and typed participant-scoped fan-out |
| `xr-ai-tools` | `xr_ai_tools` | Relay-managed tools and model tool-call helpers |
| `xr-ai-voice` | `xr_ai_voice` | Voice agent, transport, and private media pipeline |
| `xr-ai-web-events` | `xr_ai_web_events` | Bounded live browser views over selected application events |

Package guides are versioned with this site:
{doc}`/reference/agent-sdk-hub`, {doc}`/reference/agent-sdk-models`,
{doc}`/reference/agent-sdk-runtime`, {doc}`/reference/agent-sdk-tools`,
{doc}`/reference/agent-sdk-voice`, and {doc}`/reference/agent-sdk-web-events`.
The generated {doc}`/reference/python/index`
is the source of truth for public names, signatures, types, defaults, fields,
and method-level behavior.

## Ownership boundaries

An `Agent` owns its state, resources, background tasks, lifecycle, and
concurrency policy. It exposes ordinary `Tool` or `AsyncTool` instances for
direct execution and model tool-call loops. Bounded turns use `ToolSet` and
`run_tool_loop()`; callers retain model configuration, conversation policy,
retries, participant context, cancellation, and task ownership. `AgentRuntime`
owns only typed publication and delivery tasks; it does not own model loops,
planning, memory, media, or agent-created work. A publication waits for its
subscribers, so a subscriber that performs lengthy work should hand it to its
own bounded queue.

Workers construct model clients with the factories in `xr_ai_models` and depend
on the corresponding typed service protocols. Adapter behavior, endpoint
connectivity, deployment ownership, credentials, and model-specific wire
details remain inside that package. See {doc}`ai-services` for deployment and
operational guidance.

`VoiceAgent` owns readiness, hub transport, voice gating, its private media
pipeline, signal handling, and cleanup. It publishes final pre-gate transcripts
separately from gate-accepted user queries, optionally publishes typed
participant lifecycle events, and consumes typed voice output. Join and leave
publication stays non-blocking for the media pipeline while preserving order per
participant. Application agents subscribe to the events they own and clean up
their own participant state. Pipecat and the media session remain implementation
details.

Applications may publish selected compact payloads to `WEB_EVENT_TOPIC` for a
`WebEventsAgent` to display. The viewer owns only a bounded live history and a
loopback HTTP listener. It is not persistence, does not inspect every runtime
topic, and does not enter model, voice, media, or hub authentication paths. The
application selects which typed events to forward explicitly.

Applications with multiple speech producers may place
`VoiceAggregationAgent` before `VoiceAgent`. Producers publish candidate
finite or incremental responses to `voice.contribution`; the aggregator owns
participant-scoped ordering, preserves a lone stream, coalesces simultaneous
finite updates through the configured `LLMService`, publishes completed raw or
rewritten text immediately, then reserves a bounded, open-loop estimate of its
spoken duration only for scheduling subsequent speech. Consequently the
client's completed-response data echo does not wait for playback pacing.
Pending work is capacity-bounded with a
priority-aware drop policy: routine work never displaces an alert, while a new
alert replaces the oldest pending routine update or, if necessary, the oldest
alert. Urgent output bypasses coalescing and rewriting, retains displaced
routine work encountered during coalescing or rewrite for a later batch, and
interrupts active speech. Interrupted stream IDs remain quarantined through
their terminator or idle expiry, and the agent enforces its rewrite deadline
independently of the model transport. This
policy remains outside the private media pipeline so applications opt in
explicitly and retain ownership of which events should become speech.

`ProcessorEndpoint` is the minimal agent-side hub boundary. It receives data,
audio, frame signals, and participant events and sends participant-routed return
traffic. Video tools acquire frames on demand; raw pixels and media stay on the
hub path rather than entering agent APIs.

## Documentation boundary

Public API membership comes from literal `__all__` declarations. Sphinx parses
the source without importing SDK packages, then renders co-located docstrings,
annotations, and defaults. Package READMEs retain installation, quickstarts,
and cross-call behavioral guidance; this component page records only shared
architecture and ownership. Guides, troubleshooting, and migrations remain
handwritten because they describe workflows and decisions rather than API
shape.
