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

The sample uses native `xr_ai_runtime` agents and `xr_ai_tools`; it does not use
NAT, PydanticAI, or MCP. Nemotron-3-Nano-Omni on port 8108 supplies both language
reasoning and visual inference. STT, embedding, and RAG remain separate typed
services. There is no monitoring dashboard: operational records are written as
JSON Lines files under `artifacts/`.

## Run it

Start the reusable model services in one terminal:

```bash
uv run --project agent-samples/model-servers model_servers
```

Then start the tea stack from the repository root:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample \
  --voice-mode wake-word \
  --tts-mode piper
```

Open `https://localhost:8080`, accept the self-signed certificate on first use,
allow microphone and camera access, and connect.
In wake-word mode, begin commands with “Agent” or “Hey Agent.” For example:

```text
Agent, help me make tea.
Agent, what do you see?
Agent, start watching for spills.
Agent, start recording the transcript.
```

The launcher requires both behavior choices:

- `--voice-mode wake-word` requires the configured wake phrase and allows a
  short follow-up window. `always-on` dispatches every finalized utterance.
- `--tts-mode piper` runs lightweight CPU speech on port 8105. `magpie` runs
  neural speech on port 8104 and uses CUDA when available.

The connection page is provided by XR-Media-Hub and remains part of the sample.
Only the old monitoring-specific UI is omitted.

## Behavior

The foreground agent sees only the current user query and compact workflow
state. Its native tool loop can answer directly, inspect the current image,
retrieve tea references, control the tea workflow, or start and stop background
tasks. Participant identity is injected by the application and is never exposed
as a model-selected argument.

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

## Configuration

- `yaml/models.local.json` maps both `llm` and `vlm` to Omni on port 8108 and
  declares STT, embedding, and TTS endpoints.
- `yaml/tea_making_worker.yaml` controls VAD, frame timeouts, observation
  intervals, artifact output, RAG, and workflow paths.
- `yaml/workflow.yaml` defines the tea steps, typed state, evidence gates, and
  user-facing messages.
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
