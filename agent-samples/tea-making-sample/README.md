<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making guidance

For an adaptation-oriented architecture guide, see
[`docs/source/reference/tea-making-sample.md`](../../docs/source/reference/tea-making-sample.md).

This sample combines a foreground tea guide with independent background
observers (transcripts, visual-change watching, periodic video observations).
It uses native `xr_ai_runtime` agents and `xr_ai_tools`; it does not use NAT,
PydanticAI, or MCP. See the linked guide above for architecture, agent
responsibilities, and how to adapt the sample.

## Run it

Start the reusable model services, including Piper TTS:

```bash
uv run --project agent-samples/model-servers model_servers
```

Then start the tea stack from the repository root:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample

# Allow direct event-viewer access from a trusted private network.
uv run --project agent-samples/tea-making-sample tea_making_sample \
  --expose-web-events
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

Wake-word behavior comes from the `voice_gate_yaml` selected in
`yaml/tea_making_worker.yaml`; the checked-in configuration requires the wake
phrase and allows a short follow-up window. Piper TTS is reused on port 8105.

The connection page is provided by DeviceIOHub and remains part of the sample.
Only the old monitoring-specific UI is omitted.
Spoken agent responses are also published on the connection client's
`agent.response` text channel so the page can display accessible captions. Raw
streamed text is finalized when its content completes, and aggregated text is
finalized when its rewrite completes; playback pacing does not delay the Agent
panel. An urgent interruption may stop audio after the complete intended text
has already appeared.

## Foreground behavior

While tea guidance is active, foreground requests remain limited to the current
step, tea-guide status and controls, and the independent background
applications. An unrelated request receives exactly “I can only help with the
active tea guide right now.” without calling a tool or changing guide state.
When guidance is idle, ordinary questions are not subject to that refusal.

## Foreground routing eval

The live-model eval checks idle and active first actions, exact unrelated-query
refusal, valid guide controls, current-step questions, visual routing, and
background-application controls. Start `model-servers` so the configured Omni
LLM endpoint on port 8108 is healthy, then run from the repository root:

```bash
uv run --project agent-samples/tea-making-sample/worker \
  python agent-samples/tea-making-sample/eval/eval.py
```

The command prints one `PASS` or `FAIL` line per case and exits nonzero when a
tool choice, argument schema, or response contract does not match.

## File outputs

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
- `yaml/voice_gate.yaml` is the checked-in wake-word configuration. Set
  `voice_gate_yaml: voice_gate.always-on.yaml` in the worker YAML to dispatch
  every finalized utterance without a wake phrase.
- `yaml/rag_service.yaml` indexes Markdown and text files under
  `rag-documents/` and exposes retrieval through typed RPC using msgpack over
  ZMQ.

The model configuration declares LLM, VLM, STT, embedding, and TTS as reusable.
Their health checks must pass before voice input becomes ready. The sample owns
only its RAG service, hub, and worker.

## Safety

This is a demonstration, not a safety controller. Keep hot vessels stable,
follow appliance and tea-package instructions, and do not rely on visual
inference as the sole protection against burns, spills, or electrical hazards.
