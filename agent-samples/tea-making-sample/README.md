<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making guidance

For an adaptation-oriented architecture guide, see
[`docs/source/reference/tea-making-sample.md`](../../docs/source/reference/tea-making-sample.md).

This sample combines a foreground tea guide with independent background
observers. The foreground guide identifies tea, retrieves brewing guidance,
watches visible preparation steps, and maintains one deterministic workflow per
participant. Background tasks can record transcripts, watch for requested visual
changes, and write periodic video observations without replacing the active tea
guide.

Foreground replies and workflow notices publish candidate speech through
`VoiceAggregationAgent`. It preserves one participant's active response,
combines non-urgent updates that arrive while that response is being spoken,
and forwards interrupting output immediately.

The sample uses native `xr_ai_runtime` agents and `xr_ai_tools`; it does not use
NAT, PydanticAI, or MCP. Nemotron-3-Nano-Omni on port 8108 supplies both language
reasoning and visual inference. STT, embedding, and RAG remain separate typed
services. Selected runtime events appear in a generic live browser viewer while
durable operational records remain JSON Lines files under `artifacts/`.

## Run it

Start the reusable model services in one terminal:

```bash
uv run --project agent-samples/model-servers model_servers
```

Then start the tea stack from the repository root:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample \
  --tts-mode piper

# Allow direct event-viewer access from a trusted private network.
uv run --project agent-samples/tea-making-sample tea_making_sample \
  --tts-mode piper --expose-web-events
```

Open `https://localhost:8080`, accept the self-signed certificate on first use,
allow microphone and camera access, and connect.
Open `http://127.0.0.1:8092` on the XR-AI host to watch selected foreground,
guidance, background, and participant events grouped by topic. The viewer starts
with the worker and does not read or tail the JSONL files. Pass
`--expose-web-events` to listen on all IPv4 interfaces and browse
`http://<xr-host>:8092`; allow TCP port 8092 through the host and cloud
firewalls only from your trusted development network.
In wake-word mode, begin commands with “Agent” or “Hey Agent.” For example:

```text
Agent, help me make tea.
Agent, what do you see?
Agent, start watching for spills.
Agent, start recording the transcript.
```

Wake-word mode is enabled by default:

- The default `--voice-mode wake-word` requires the configured wake phrase and
  allows a short follow-up window. Pass `--voice-mode always-on` explicitly to
  dispatch every finalized utterance.
- `--tts-mode piper` runs lightweight CPU speech on port 8105. `magpie` runs
  neural speech on port 8104 and uses CUDA when available.

The connection page is provided by XR-Media-Hub and remains part of the sample.
Only the old monitoring-specific UI is omitted.
Spoken agent responses are also published on the connection client's
`agent.response` text channel so the page can display accessible captions. Raw
streamed text is finalized when its content completes, and aggregated text is
finalized when its rewrite completes; playback pacing does not delay the Agent
panel. An urgent interruption may stop audio after the complete intended text
has already appeared.

## Behavior

The foreground agent sees only the current user query and compact workflow
state. Its native tool loop can answer directly, inspect the current image,
retrieve tea references, control the tea workflow, or start and stop background
tasks. Participant identity is injected by the application and is never exposed
as a model-selected argument.

When that loop selects `current_view`, the sample uses the same direct streaming
path as `simple-vlm-example`: one current frame goes to
`StreamingImageQueryTool`, whose chunks go straight to participant voice. Omni's
visual response is not passed back through the foreground LLM for rewriting.

The tea workflow advances only after explicit user commands. Visual observations
may update evidence-backed state, but they do not silently move to the next step.
Participant joins create fresh state; leaving cancels participant-owned work.

Background task output is retained under the sample's `artifacts/` directory:

- change-watch sessions and observations;
- finalized transcripts and periodic summaries;
- periodic visual captions and material-change records;
- foreground turns and Relay events, when enabled by the worker.

Each record includes participant routing information or lives in a
participant-specific file. The files are the integration boundary for manual
inspection and can be replaced by typed runtime subscribers in a downstream
application.

The browser viewer is an independent, bounded in-memory presentation of
selected events. It is useful while a session is live but is not durable and
does not replace the JSONL artifacts. Its listener has no application
authentication or TLS. Keep the default loopback binding unless direct access
is needed. Do not expose it to the public Internet; restrict access to a trusted
development network or put it behind an authenticated TLS proxy.

## Configuration

- `yaml/models.local.json` maps both `llm` and `vlm` to Omni on port 8108 and
  declares STT, embedding, and TTS endpoints. The shared Omni launcher accepts
  one image per request by default; this sample's visual tools intentionally
  preserve that contract. Configure the model server before adopting a
  multi-image tool such as `query_images`.
- `yaml/tea_making_worker.yaml` controls VAD, frame timeouts, observation
  intervals, artifact output, the web-events host/port/history, RAG,
  and workflow paths.
- `yaml/workflow.yaml` defines the tea steps, typed state, evidence gates, and
  user-facing messages.
- `worker/tea_making_worker/prompts/` is the default source for model prompts;
  explicit inline YAML values override those files.
- `yaml/voice_gate.yaml` is the wake-word profile. The launcher selects the
  always-on profile when requested.
- `yaml/rag_service.yaml` indexes Markdown and text files under
  `rag-documents/` and exposes retrieval over typed msgpack/ZMQ RPC.

Model profiles declare the heavy services as reusable. Their health checks must
pass before voice input becomes ready. The sample owns its selected TTS process,
RAG service, hub, and worker.

## Safety

This is a demonstration, not a safety controller. Keep hot vessels stable,
follow appliance and tea-package instructions, and do not rely on visual
inference as the sole protection against burns, spills, or electrical hazards.
