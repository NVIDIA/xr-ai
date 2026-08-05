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

The selected mechanism receives both kinds of active-step trigger:

- A `periodic` event may run the step VLM prompt and return a writable context
  patch.
- A `voice` event receives the same projected context and step procedure, but
  has a read-only response contract. It can inspect a fresh frame, query RAG,
  read timers, and read the observation log, but it cannot update context or
  complete a step. Navigation is classified before either path.

This capability split is enforced in code: voice events are offered only tools
marked `read_only`, return no state schema, and any context patch returned by a
custom mechanism is discarded by the guide. This keeps one mechanism and one
tool-calling loop without allowing an answer to mutate workflow state.

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

Time is handled like other external information. The final steeping step enables
`get_current_time` and `get_timer_status`. It uses `vlm_stop_when` to caption
frames only until the steeping start timestamp exists; later periodic iterations
are tool-only until the timer expires. `suppress_reminders_when` prevents the
missing-immersion reminder after the timestamp is recorded. These conditions
are generic YAML rules and can be used by other workflows.

`CaptionAgentStepMechanism` is separate from `WorkflowGuide` in
`step_mechanism.py`. Another workflow can implement the small `StepMechanism`
protocol, register it in `StepMechanisms`, and select it by name for any YAML
step. Session, reminder, completion, and navigation lifecycle stays shared.
`StepEvent` and `StepIteration` are task-neutral trigger and result contracts,
so a future mechanism can use a sensor, deterministic service, or different
agent while preserving the same guide lifecycle.

Agent tools use `AgentTool` and `ToolCatalog` from `tools.py`. A tool pairs its
model-facing schema with an async handler and a `read_only` capability. Tools
created for one request, such as `inspect_current_view`, use the same contract
as long-lived tools such as RAG and timers. This removes tool-name branches from
the generic LLM loop and gives future agents one place to select tools by name
and capability.

Voice prompts include only the current step projection and the previous user
request, not prior assistant measurements. Repeated timer questions therefore
call `get_timer_status` again instead of anchoring on an earlier spoken value.
Future-step details are absent from ordinary answer prompts; an explicit request
uses the read-only `get_next_workflow_step` tool to retrieve the exact YAML step.

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
"Agent, start making tea" creates a fresh workflow at step 1. Canceling in
the middle uses the same reset path and clears context, observations, reminders,
and conversation history.

Navigation commands use a short LLM classifier, with a YAML-backed local
fast path for obvious start, stop, status, and proceed commands. The classifier
can recognize conversational commands such as "carry on to the next part";
ordinary progress reports do not advance the workflow.

The worker logs guide decisions to `worker.log`: user query text, classifier
intent, mechanism iterations, step transitions, VLM observations, agent context
patch keys, tool calls/results, reminders, notices, and final response text.

Disconnecting clears the participant's active workflow and removes the session.
Reconnecting starts from a fresh idle state; unfinished context and visual
observations are never resumed.

`yaml/voice_gate.yaml` requires every spoken command to begin with "Agent"
or "Hey agent". The follow-up grace window is disabled, so an ungated second
utterance is ignored. On connection, the shared voice gate speaks the configured
welcome message explaining that the user should say "Agent, help me make
tea." Worker-generated reminders and completion notices bypass the wake gate.

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
- `2` Fill water
- `3` Start heating and reach the target temperature
- `4` Steep the tea (VLM until immersion, then timer tool only)

While the final timer is active, questions such as "Agent, how much time has
passed?" and "Agent, how long do I still need to wait?" return a fresh
elapsed or remaining value. The guide announces when the steeping time is up and
returns to idle.

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
Open the web client served by the media hub. After the welcome message, say
"Agent, help me make tea".

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
step genuinely needs a different execution strategy. The event, mechanism, and
tool-catalog contracts are deliberately sample-local for now; move them into a
shared SDK package after a second workflow validates that their APIs cover more
than this example.
