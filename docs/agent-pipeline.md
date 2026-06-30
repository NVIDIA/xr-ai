<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Agent pipeline

The unified voice pipeline (`xr-ai-pipecat`) assembles a chain of pipecat
`FrameProcessor`s.  Read this when writing a new agent brain, adding a
capability or MCP toolset, or debugging frame-level behavior.

For system-level topology (hub, IPC, LiveKit) see `docs/architecture.md`.
For adding a new sample from scratch see `docs/adding-a-sample.md`.

---

## Pipeline shape

```
XRMediaHubInputTransport
        │  InputAudioRawFrame (transport_source = pid)
        ▼
VadSttProcessor
        │  UserStartedSpeakingFrame (transport_source = pid)
        │  UserStoppedSpeakingFrame (transport_source = pid)
        │  TranscriptionFrame       (user_id = pid, transport_source = pid)
        │  InterruptionFrame        (transport_source = pid)  ← early STOP probe only
        ▼
VoiceGateProcessor
        │  GatedQueryFrame  (participant_id, text, fresh_match, pts_us)
        │  InterruptionFrame (transport_source = pid)         ← gate's STOP handler
        │  TextFrame        ("Okay, I will stop.", transport_destination = pid)
        ▼
BrainProcessor  (subclass)
        │  TextFrame        (transport_destination = pid)     ← one per yielded chunk
        │  BrainResponseEndFrame (pid, text, pts_us)          ← end of turn
        ▼
StreamingTtsProcessor
        │  OutputAudioRawFrame (transport_destination = pid)
        ▼
XRMediaHubOutputTransport
        │  AudioChunk → hub → client
```

`make_voice_pipeline()` assembles this chain and returns a ready-to-run
`PipelineWorker`.  Pass a `BrainProcessor` subclass; configure VAD knobs
via `VadConfig` and gate behavior via `VoiceGateConfig`.

---

## Processor contracts

### `XRMediaHubInputTransport`

Receives `AudioChunk` and `FrameSignal` from the hub over ZMQ and converts
them to pipecat frames.  Sets `transport_source = participant_id` on every
`InputAudioRawFrame` so downstream processors can route by speaker.

### `VadSttProcessor`

One `VadDetector` per participant (keyed by `transport_source`).  Emits:

- `UserStartedSpeakingFrame` — when VAD accumulates ≥ `min_speech` seconds.
  `transport_source` is set to the speaker's pid.
- `UserStoppedSpeakingFrame` — at utterance end.  `transport_source` = pid.
- `TranscriptionFrame` — STT result.  `user_id` and `transport_source` both
  carry the pid.
- `InterruptionFrame` — only from the early STOP probe (a fast-path that runs
  STT on the partial buffer after `stop_probe_after_s`).  `transport_source` =
  pid so the brain cancels only that speaker's task.

### `VoiceGateProcessor`

Feeds each `TranscriptionFrame` into a per-pid `VoiceGate` state machine.
Emits one of:

| Event | Frame emitted |
|---|---|
| Magic phrase + query | `GatedQueryFrame(participant_id, text, fresh_match=True, pts_us)` |
| Follow-up window open | `GatedQueryFrame(participant_id, text, fresh_match=False, pts_us)` |
| "Stop" detected | `InterruptionFrame(transport_source=pid)` + `TextFrame(stop ack)` |
| Magic phrase only | nothing (opens follow-up window) |
| Neither | nothing |

`UserStartedSpeakingFrame` passes through unconditionally — the gate does not
check its state before forwarding it.  `on_user_started_speaking` therefore
fires on every VAD onset, including utterances that the gate will ultimately
drop.

### `BrainProcessor`

Base class for all agent brains.  Owns:

- **Per-pid in-flight task** — at most one `handle_query` coroutine running
  per participant; a new `GatedQueryFrame` cancels the prior task.
- **`InterruptionFrame` handling** — cancels the task for the pid in
  `frame.transport_source`; falls back to cancelling all tasks when no pid is
  set (global interrupt).
- **Hook dispatch** — `on_user_started_speaking(pid)`, `on_query_superseded(pid)`,
  `on_participant_joined(pid)`, `on_participant_left(pid)`.

**Subclass contract:** implement `handle_query(pid, text, fresh_match)` and
return either a string or an `AsyncIterator[str]`.  Each yielded chunk becomes
a `TextFrame` routed to `pid`.

```python
class MyBrain(BrainProcessor):
    async def handle_query(self, pid, text, fresh_match):
        return await self._vlm.ask(...)            # single string
        # or:
        return self._vision.ask(pid, text)         # AsyncIterator[str]
```

### `StreamingTtsProcessor`

Consumes `TextFrame`s, calls the TTS service, and emits
`OutputAudioRawFrame`s.  Echoes the completed turn text on the
`agent.response` data channel once the `BrainResponseEndFrame` arrives.
Also observes TTS WAVs from the voice gate chime.

### `XRMediaHubOutputTransport`

Converts `OutputAudioRawFrame`s back to `AudioChunk`s and sends them to the
hub, addressed to the participant in `frame.transport_destination`.

---

## Frame types (`xr_ai_pipecat.frames`)

| Frame | Fields | Who emits | Who consumes |
|---|---|---|---|
| `GatedQueryFrame` | `participant_id`, `text`, `fresh_match`, `pts_us` | `VoiceGateProcessor` | `BrainProcessor` |
| `BrainResponseEndFrame` | `pid`, `text`, `pts_us` | `BrainProcessor` | `StreamingTtsProcessor` |
| `ParticipantJoinedFrame` | `participant_id` | `XRMediaHubInputTransport` | `VoiceGateProcessor`, `BrainProcessor` |
| `ParticipantLeftFrame` | `participant_id` | `XRMediaHubInputTransport` | `VadSttProcessor`, `VoiceGateProcessor`, `BrainProcessor` |

Pipecat's built-in frames (`UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`,
`InterruptionFrame`, `TranscriptionFrame`, `TextFrame`, `InputAudioRawFrame`,
`OutputAudioRawFrame`) are used directly.  `transport_source` and
`transport_destination` on the pipecat base `Frame` carry the participant id;
our processors always set these on the frames they emit.

---

## Brain building blocks (`xr-ai-capabilities`)

Two composable abstractions let a brain expose tools to an LLM without
hard-coding routing logic.

### `AgentCapability`

A **brain-local** tool — executes inside the agent process, no network hop.
Implement two methods:

```python
class MyCapability(AgentCapability):
    def as_tool_defs(self) -> list[ToolDef]:
        return [ToolDef(name="my_tool", description="...", parameters={...})]

    async def execute(self, name, args, pid, *, onset_pts_us=0, end_pts_us=0) -> dict:
        ...
```

`VisionModule` is the canonical example: it exposes `look_at_current_frame`
and handles camera access, frame fetch, and VLM call entirely in-process.

### `MCPToolset`

A **remote** tool server reachable via FastMCP.  Pairs a client with the
tool names it owns:

```python
MCPToolset(oxr_client, frozenset({"get_head_pose", "position_ahead", ...}))
MCPToolset(render_client)   # tools=None → catch-all
```

`route_tool(toolsets, name)` returns the first toolset whose set contains
`name`, or the catch-all.  `collect_tool_defs(toolsets)` queries all clients
once and returns `ToolDef` objects for the LLM's tool list.

### Wiring them into a brain

```python
brain = MyBrain(
    toolsets=[
        MCPToolset(oxr,    _OXR_TOOLS),
        MCPToolset(render),               # catch-all
    ],
    capabilities=[
        VisionModule(transport.endpoint, vlm),
    ],
)
```

`BrainProcessor` (M2, planned) will route tool calls automatically: checks
capabilities first (by name), then `route_tool` for MCP servers.  Until then,
subclasses call `route_tool` / `capability.execute` directly in their
`handle_query` implementation.

---

## `VisionModule`

`VisionModule` implements `AgentCapability` and exposes one brain-local tool:

| Tool | Constant | What it does |
|---|---|---|
| `look_at_current_frame` | `VISION_TOOL_NAME` | Fetches the latest live frame and runs VLM Q&A inline |

Camera streaming is always-on; `VisionModule` never sends `startCamera` /
`stopCamera`.  It waits up to `frame_timeout_s` for a fresh frame and raises
`VisionUnavailable` if none arrives.

Two call styles over the same frame-acquisition path:

- `ask(pid, query)` — `AsyncIterator[str]` for TTS streaming.
- `perceive(pid, query)` — `str` for agentic tool loops; raises
  `VisionUnavailable` on failure.
- `execute(name, args, pid)` — `AgentCapability` interface; wraps `perceive`,
  returns `{"answer": text}` or `{"spoken": message}`.

---

## Adding a new brain

1. Subclass `BrainProcessor`.
2. Implement `handle_query(pid, text, fresh_match)`.
3. Use `MCPToolset` + `collect_tool_defs` to wire MCP servers.
4. Use `VisionModule` (or a custom `AgentCapability`) for brain-local tools.
5. Pass to `make_voice_pipeline(transport, stt, tts, brain, vad_cfg, voice_gate_cfg)`.

See `agent-samples/simple-vlm-example/worker/agent.py` (minimal VLM brain)
and `agent-samples/xr-render-demo/worker/processors.py` (full agentic loop
with MCP tool calling).
