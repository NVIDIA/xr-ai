<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# tea-making-sample

`tea-making-sample` is a YAML-driven guided workflow sample. It guides the
wearer through making tea by periodically running a step-specific VLM prompt,
feeding that caption into a tool-calling LLM loop, and updating step context.

The code is intentionally generic. The main customization points are:

- `yaml/workflow.yaml`: step IDs, names, descriptions, VLM prompts,
  agent prompts, per-step tools, context output fields, completion rules,
  skip defaults, and optional timers.
- `rag-documents/`: Markdown reference documents indexed by the native RAG
  service and returned through `rag_lookup`.
- `worker/tea_making_worker/prompts/system.txt`: user-question answering behavior.

## Workflow Shape

Visual steps follow the same loop:

1. The worker periodically captures the latest live camera frame.
2. The step's `vlm_prompt` captions only the visual facts needed for that step.
3. The step enters a mini state machine: `started`, `needs_input`, or `complete`.
4. The LLM receives the caption, current context, generated context schema, and
   the step's `agent_prompt`.
5. The LLM can call only the tools enabled by that step's `agent_tools`, then
   returns JSON context updates.
6. The worker marks the step ready when `advance_when` is satisfied.
7. The worker waits for a semantic proceed command before moving to the next
   step. If the step is incomplete, it applies that step's `skip_defaults` first.

The worker gives the step's `on_enter_message` once at the beginning. Periodic
VLM/LLM passes silently update context. If the step still lacks required
information after `runtime.reminder_interval_s`, it sends one delayed reminder
by default. It does not repeat every VLM caption or rephrase the same opening
instruction.

Tools that must run for correctness can set `auto_invoke: true` in
`agent_tools`, with `when_context_empty` and `context_outputs` mappings. The
worker then invokes the tool as soon as its optional VLM verdict policy matches
and merges mapped results authoritatively, while prompts remain responsible for
task-specific interpretation and guidance.

A step may define a YAML `timer` instead of a `vlm_prompt`. Timer-only steps do
not capture frames or invoke the step agent. The worker derives elapsed and
remaining time from context, answers timer questions deterministically, and
sends the configured completion notice when time expires.

A visual step can declaratively map structured VLM output into context with
`state_updates`:

```yaml
state_updates:
  - context_field: measured_value
    observation_key: MEASURED_VALUE
    states: [started, needs_input, complete]
    value_map:
      not_visible: "not visible"
```

The worker parses the named observation line, coerces the value using the
context field's YAML type, and applies it authoritatively before and after the
step agent runs. `states` controls which step mini-states accept the update. A
mapping that includes `complete` keeps that field current while the completed
step waits for the wearer to say next, without reopening the step or repeating
guidance. `value_map` is optional and can translate VLM labels into typed
context values such as booleans. This mechanism is task-independent; changing
the workflow YAML is sufficient to reuse it for other observed state.

Completing or skipping the final step ends the session and resets it to idle.
The guide does not continue from the completed context; a phrase such as
"start making tea" creates a fresh workflow at step 1.

Navigation commands use a short LLM classifier, with a YAML-backed local
fast path for obvious start, stop, status, and proceed commands. The classifier
can recognize conversational commands such as "carry on to the next part";
ordinary progress reports do not advance the workflow.

The worker logs guide decisions to `worker.log`: user query text, classifier
intent, step transitions, VLM observations, context patch keys, dropped updates
that contradict VLM verdict lines, tool calls/results, reminders, notices, and
final response text.

Transient web-client disconnects pause visual monitoring without clearing the
active step or accumulated context. Monitoring resumes when the same
participant ID reconnects.

VLM captions are internal evidence. They are stored in the per-participant
observation log and returned to the LLM through `get_recent_vlm_observations`
when visual evidence is needed, but they are not posted directly to the user.

The included tea workflow has these steps:

- `0` Idle
- `1` Identify tea information
- `2` Fill water and start heating
- `3` Wait for water to boil
- `4` Steeping the tea
- `5` Wait for steeping to finish (timer only; no VLM captions)

After steeping starts, advance to step 5 with wording such as "next", "let's
proceed", or "carry on". While the timer is active, questions such as "how much
time has passed?" and "how long do I still need to wait?" return elapsed and
remaining time. The guide announces when the steeping time is up.

## Run

The tea sample is a consumer of the persistent shared model stack. It does not
launch model servers or define GPU memory allocations of its own.

First, from the repository root, start `model-servers`:

```bash
uv run --project agent-samples/model-servers model_servers
```

That command selects the repository's hardware profile, starts the persistent
STT, LLM, VLM, and embedding services, waits for them to become healthy, and
then exits. Next, start the tea sample:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample
```

The tea launcher verifies that all four required shared endpoints are running:

| Shared service | Role | Port |
|---|---|---|
| Parakeet STT | Speech recognition | 8103 |
| Nemotron-3 Nano | Agent reasoning and tool calling | 8107 |
| Cosmos-Reason1-7B | Live-frame scene understanding | 8100 |
| Llama Nemotron Embed | Tea-document indexing and retrieval | 8109 |

It then starts only the sample-owned native RAG service, media hub, Piper TTS,
and tea worker. The model roles remain intentionally separate:

- Nemotron-3 Nano is the agent LLM. It classifies workflow navigation, reasons
  over YAML-defined state, answers the wearer, and performs native tool calls
  such as `rag_lookup`.
- Cosmos-Reason1-7B is the scene-understanding VLM. It receives the latest
  camera image and the active step's `vlm_prompt`, then returns visual evidence
  to the agent loop.
- Llama Nemotron Embed indexes and retrieves the local Markdown documents. It
  is not used for agent generation.

`yaml/models.yaml` declares these roles independently through the shared
`xr-ai-models` presets. The worker constructs a separate `LLMService` and
`VLMService`; no multimodal Omni model or hand-written HTTP model client is
used.

All hardware detection, model selection, container lifecycle, and GPU memory
allocation live under `agent-samples/model-servers/`; there are no duplicated
model-server YAMLs in this sample. Configure model access as described in
[`docs/credentials.md`](../../docs/credentials.md), then open the web client
served by the media hub and say "help me make tea".

`yaml/tea_making_worker.yaml` bounds camera acquisition with
`frame_timeout_s` and each Cosmos scene request with `vlm_timeout_s`. The
worker log records separate frame-wait and VLM-request timings when diagnosing
a delayed observation.

When the tea sample exits, the shared models stay running for reuse by other
samples. Stop them explicitly when they are no longer needed:

```bash
uv run --project agent-samples/model-servers model_servers --stop
```

## Adapting The Sample

To build a different guided task, copy `yaml/workflow.yaml` and replace the
step prompts, context fields, and RAG documents. The worker does not know about
tea-specific fields; it loads the YAML, generates the context-output prompt,
runs visual or timer steps, applies `skip_defaults`, and evaluates
`advance_when`. Keep task-specific behavior in `vlm_prompt`, `agent_prompt`,
`agent_tools`, and timer mappings rather than adding task branches to the
worker.
