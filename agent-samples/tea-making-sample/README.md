<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# tea-making-sample

`tea-making-sample` is a YAML-driven guided workflow sample. It guides the
wearer through making tea by periodically running a step-specific VLM prompt,
feeding that caption into a tool-calling LLM loop, and updating step context.

The code is intentionally generic. The main customization points are:

- `yaml/workflow.yaml`: shared context fields, step reads and writes, VLM prompts,
  agent prompts, mechanism names, per-step tools, completion rules, and skip defaults.
- `rag-documents/`: Markdown reference documents indexed by the native RAG
  service and returned through `rag_lookup`.
- `worker/tea_making_worker/prompts/system.txt`: user-question answering behavior.

## Workflow Shape

Every active tea step uses the default `caption_agent` mechanism. A step only
needs a `mechanism` key when it overrides that default:

1. The guide resolves the step's YAML `mechanism` through a mechanism registry.
2. If the step has a `vlm_prompt`, the mechanism captures one fresh frame and
   produces one internal caption. A step without that prompt skips vision.
3. The LLM receives the optional caption, projected context, writable-field
   schema, and `agent_prompt`.
4. The LLM can call only the tools listed in `agent_tools`, then returns one
   partial JSON context patch.
5. The guide applies that patch and marks the step ready when `advance_when` is
   satisfied.
6. The guide waits for a semantic proceed command before moving to the next
   step. If incomplete, it applies that step's `skip_defaults` first.

The guide gives `on_enter_message` once. Periodic mechanism iterations silently
update context. If information is still missing after
`runtime.reminder_interval_s`, it sends one delayed reminder by default.

There is no separate caption-to-state mapping. The caption always goes through
the agent, and the agent's patch is the only state update. New captions can
replace writable observations during the same incomplete step:

```yaml
mechanism: caption_agent
vlm_prompt: "Read the current display."
agent_prompt: "Replace measured_value with the newest visible reading."
writes: [measured_value]
```

Time is handled like other external information. Step 4 enables
`get_current_time` so the agent can record when steeping starts. Step 5 has no
`vlm_prompt`; its periodic agent iteration calls `get_timer_status` and sets
`steeping_complete` when the tool reports expiration. User timer questions use
the same tool through the answer agent.

`CaptionAgentStepMechanism` is separate from `WorkflowGuide` in
`step_mechanism.py`. Another workflow can implement the small `StepMechanism`
protocol, register it in `StepMechanisms`, and select it by name for any YAML
step. Session, reminder, completion, and navigation lifecycle stays shared.

Workflow state is persistent and sparse. Declare reusable fields once under
`context.fields`; then give each step a `reads` list and a `writes` list:

```yaml
context:
  fields:
    measured_value:
      type: number
    check_complete:
      type: boolean

steps:
  - id: 1
    reads: [measured_value]
    writes: [check_complete]
```

Only populated fields are carried in a session. A field is created at startup
only when its declaration has an `initial` value. Steps do not need to repeat
carried field schemas, and the state agent may return any subset of its writable
fields. Writes are mutable by default, so newer observations can replace older
ones within the same step. For a small workflow, a step may also declare a new
field inline by making `writes` a mapping of field names to schemas. The older
per-step `context_output.fields` format remains supported.

Completing or skipping the final step ends the session and resets it to idle.
The guide does not continue from the completed context; a phrase such as
"start making tea" creates a fresh workflow at step 1.

Navigation commands use a short LLM classifier, with a YAML-backed local
fast path for obvious start, stop, status, and proceed commands. The classifier
can recognize conversational commands such as "carry on to the next part";
ordinary progress reports do not advance the workflow.

The worker logs guide decisions to `worker.log`: user query text, classifier
intent, mechanism iterations, step transitions, VLM observations, agent context
patch keys, tool calls/results, reminders, notices, and final response text.

Transient web-client disconnects pause visual monitoring without clearing the
active step or accumulated context. Monitoring resumes when the same
participant ID reconnects.

VLM captions are internal evidence. They are stored in the per-participant
observation log and returned to the LLM through `get_recent_vlm_observations`
when historical evidence is needed, but they are not posted directly to the
user. For present-tense visual questions, `inspect_current_view` captures a new
frame using the wearer's actual question. This also works in no-caption steps;
it does not turn periodic VLM monitoring on for those steps.

Messages support speech-oriented template filters: `| duration` turns seconds
into natural durations, `| local_time` renders epoch or ISO timestamps as local
clock times, and `| spoken` expands common temperature units. Final responses
also normalize temperature notation before reaching text to speech.

The included tea workflow has these steps:

- `0` Idle
- `1` Identify tea information
- `2` Fill water and start heating
- `3` Wait for water to boil
- `4` Steeping the tea
- `5` Wait for steeping to finish (agent tool only; no VLM captions)

After steeping starts, advance to step 5 with wording such as "next", "let's
proceed", or "carry on". While the timer is active, questions such as "how much
time has passed?" and "how long do I still need to wait?" return elapsed and
remaining time. The guide announces when the steeping time is up.

## Run

Run the sample:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample
```

The launcher starts STT, Nemotron-Omni, the embedding server, native RAG
service, media hub, TTS, and this worker. Nemotron-Omni handles both visual
captioning and agent reasoning; no Llama-Nemotron chat model is used. The
embedding model is used only to index and retrieve the local Markdown
documents.

The launcher uses `nvidia-smi` to select one of two deployment profiles before
starting any model service:

| Hardware | Profile | Allocation |
|---|---|---|
| 1 x 96 GiB Blackwell | `yaml/96G_blackwell/` | NVFP4 Omni, STT, and embedding on GPU 0. Omni is capped at 80%, embedding at 5%, and RTX Pro uses the Triton MoE backend. |
| 2 x 48 GiB L40/L40S | `yaml/dual_48G_ada/` | FP8 Omni alone on GPU 0; STT and embedding on GPU 1. |

DGX Spark detection reuses the Blackwell profile. The Omni server starts first
so vLLM profiles its reservation on an otherwise empty GPU. It pins
`vllm/vllm-openai:v0.20.0`, the minimum release that registers the model
architecture, and limits Omni to 8 sequences and a 32K context because this
interactive workflow does not need the model card's server-scale defaults.
Open the web client served by the media hub and say "help me make tea".

`yaml/tea_making_worker.yaml` bounds camera acquisition with
`frame_timeout_s` and each Omni caption request with `vlm_timeout_s`. The
worker log records separate frame-wait and VLM-request timings when diagnosing
a delayed observation.

If Omni fails with a message like "Free memory on device ... is less than
desired GPU memory utilization", another GPU model service is still loaded.
Stop the shared model-server stack first:

```bash
uv run --project agent-samples/model-servers model_servers --stop
```

Then rerun `tea_making_sample`.

## Adapting The Sample

To build a different guided task, copy `yaml/workflow.yaml` and replace the
field registry, step projections, prompts, and RAG documents. The worker does
not know about tea-specific fields; it loads the YAML, generates write schemas,
runs the selected step mechanism, applies `skip_defaults`, and evaluates
`advance_when`. Keep task-specific behavior in `vlm_prompt`, `agent_prompt`, and
the plain `agent_tools` list. Add a new `StepMechanism` only when a workflow
step genuinely needs a different execution strategy.
