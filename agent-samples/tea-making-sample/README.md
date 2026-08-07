<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Tea-making guidance sample

This sample is a configuration-driven physical-task guide. A participant has
one outer workflow state machine; its active step runs one repeated inner loop:

```text
    YAML NAT trigger -> plain caption -> step NAT agent -> workflow__commit
```

Voice uses foreground-process semantics:

```text
idle -> root agent -> answer | vision/RAG | start tea guide
active -> current tea-step agent -> answer | quick tool | lifecycle transition
```

There is no custom tool-call loop. NeMo Agent Toolkit builds the root agent,
every observation agent, and one foreground variant for each tea step. Current
and historical conversation turns are not added to prompts. Session state
selects exactly one foreground function before the LLM call, so the inactive
agent is never invoked and no model delegates to another model. Root exposes
start, vision, and RAG. A tea variant exposes lifecycle tools plus only its
step's quick tools. The exact utterance reaches that agent directly.

Starting an active guide remains an idempotent status response at the function
boundary, and only reset or restart may clear state. Explicit imperative
next/continue/advance/skip commands select the advance function; questions that
merely contain those words remain direct answers in the current foreground.

Every observation agent calls the same commit function exactly once. Completed,
unsupported, or unclear observations use an empty commit. Step prompts name the
specific evidence required for each readiness write; a reported action, repeated
caption, tool target, or unrelated retrieval passage is never completion by
itself. The shared agent prompt asks for one short spoken message when a commit
changes state without completing the step. No-op and completion commits remain
silent, avoiding repeated progress and duplicate completion announcements.

Each voice policy contains the minimum procedure needed to resolve references
such as “how do I do that?” without conversation history. The `current_view`
tool exposes only a question to the model; participant identity is injected
from the active invocation, preventing a small model from inventing an ID and
looking at the wrong camera stream.

## Configure a workflow

Edit `yaml/workflow.yaml`. Each step declares:

- `reads` and `writes`: its complete state boundary.
- `trigger`: any registered NAT function, interval, arguments, and optional
  result field. `$participant_id` and `$state.<field>` resolve at invocation.
- `evidence`: an optional caption pattern and required consecutive match count
  that must hold before a completion commit is accepted.
- `agent`: the observation policy and additional NAT tools. The state commit
  function is always present.
- `voice`: a read-only policy and its NAT tools.
- `complete_when` and `next`: readiness and the destination of an explicit
  voice advance.
- `state_on_skip` and `messages`: deterministic management behavior.

User-facing message templates support generic presentation filters:
`{{ value | temperature_c }}` speaks a full “degrees Celsius” value and
`{{ value | duration }}` converts seconds to natural minutes and seconds.
Internal state remains numeric for comparisons and tools. The shared voice
contract gives both voice answers and background agent notices one
workflow-independent final-output rule: rewrite abbreviations, symbols, unit
notation, machine formats, and compact tool or state text as complete spoken
words and familiar quantities without changing meaning.

The worker contains no tea-specific branch. The five supplied steps use the
same engine even though four are triggered by live vision and one by the native
clock timer.

Visual trigger prompts are short focus guides, not output schemas. The VLM
returns an ordinary plain-text caption describing only relevant visible facts.
The configurable evidence gate must pass before the observation agent can
complete a visual step. Identification accepts one readable OCR caption for
responsiveness and explicitly rejects dark, unreadable, or absent-text
captions. Filling uses the stricter threshold: three consecutive captions must
name the vessel, visible water, and a concrete cue such as its surface or level.
An active-heater indicator or unit-bearing temperature display records the
durable heating-started milestone; automatic observation does not track water
readiness. Once heating is detected, the user may advance manually or ask
whether the water is hot enough. For that question, the voice agent reads the
current display and calls `temperature__verify`. The tool obtains the target
from session state, converts Fahrenheit when needed, and returns the observed
and target values with the comparison result without changing workflow state.
Statements that hazards or heater indicators are not visible do not invalidate
an otherwise clear, unit-bearing temperature reading.
Steeping starts only after the same vessel shows both water and tea-water
contact; a tea bag in a dry or visually ambiguous cup is not enough.

The identify step prefers brewing values visible on the package. When the tea
name is visible but its temperature or time is missing, its agent calls the
native `rag_lookup` tool over the sample's `rag-documents/` corpus. Retrieved
guidance never overrides package instructions, and `guidance_source` records
which source produced the values. A frame without a specific visible tea name
never triggers generic retrieval, and the sample-local alias caps retrieval at
two passages to keep the second agent call small. Retrieval queries contain the
exact visible name plus the missing brewing facts, and compact chunks reduce
neighboring-variety contamination. Voice answers obtain tea
identity only from committed state or live vision; RAG supplies brewing values
for a known name and is never an identity source. Identification becomes ready
in the same atomic commit that records an evidence-backed name, temperature,
duration, and source; generic retrieval results cannot fill missing values.
Vision transcribes only legible printed words in reading order and does not
classify tea from package artwork. A retrieved passage must contain the visible
variety before it can support missing brewing values.
The identity frame focuses on front-label brand and tea/blend text while
excluding slogans, package count, weight, and badges. Identification state has
no draft form: an incomplete turn commits nothing, and a complete turn writes
identity, brewing values, source, and readiness atomically from the fresh
caption.

State is sparse and typed. A step sees only its `reads + writes` projection and
can update only `writes`. A commit is rejected atomically when a field or type
is invalid. Completion is derived from `complete_when`; the model cannot assert
completion separately.
Observation calls label prior status as `already_complete` and pair the compact
input with a generated contract containing each writable field's YAML meaning
and completion condition. This keeps status distinct from writable state while
giving a small model enough semantics to produce a supported patch.
Completing a step never changes the active step. Observation continues through
the same trigger-agent-commit loop; commits become revision-free no-ops while
the completion predicate remains true. Only an explicit user “next,”
“continue,” or “skip” request lets the active tea agent advance it.
Participant state is ephemeral. A real client join, including the roster replay
after an app restart, resets guidance to idle; leaving and reconnecting creates
a fresh session. Duplicate roster replays for an already-known connection are
ignored so they cannot erase an active walkthrough.

## Model modes

The launcher requires model, voice, and speech behavior explicitly. With no
arguments it prints the available choices and exits without starting services:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample
```

Start the shared model servers once:

```bash
uv run --project agent-samples/model-servers model_servers
```

Select Nemotron-3-Omni for both vision and agent reasoning with:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample \
  --model-mode omni --voice-mode always-on --tts-mode piper
```

Select Cosmos Reason1 for vision and Nemotron-3-Omni for agents with:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample \
  --model-mode cosmos --voice-mode wake-word --tts-mode magpie
```

`--voice-mode always-on` accepts every transcribed utterance.
`--voice-mode wake-word` requires each command to start with “Agent” or “Hey
Agent”; saying only the wake phrase opens a five-second follow-up window. The
`--tts-mode piper` selects lightweight CPU speech on port 8105;
`--tts-mode magpie` selects NeMo Magpie speech on port 8104 and uses CUDA when
available. The sample uses Magpie's native speaking rate to preserve voice
quality. The launcher writes temporary model, worker, and RAG configs, so
switching modes never edits source files and every process uses the selected
profile.

While idle, the root foreground can answer directly, inspect the current view,
retrieve tea knowledge, or launch tea guidance. While active, the current tea
variant alone handles the turn with next/skip/restart/status/exit plus that
step's tools. Lifecycle calls carry the `tea_guide` scope and return directly;
quick tool results remain in the same NAT agent loop. Direct answers are valid
but cannot mutate state.

Both profiles disable hidden reasoning for Omni agent calls. The Omni vision
profile also caps continuous caption generation; agent calls retain their
tool-loop budget. A timed-out caption is
logged as `vision.timeout` and the next frame continues the same observation
loop. Both model modes reuse Parakeet STT and Nemotron Embed while the sample
manages the selected Piper or Magpie TTS service, the typed RAG service, the
media hub, and its worker.

Both modes use the embedding endpoint at port 8109. The RAG service reads
`yaml/rag_service.yaml`, indexes the local Markdown corpus, and exposes only the
shared `rag__retrieve` capability; the worker narrows it to the step-selectable
`rag_lookup` NAT function.

Open the web client shown by the hub. In wake-word mode, say, “Agent, help me
make tea.” In always-on mode, say, “Help me make tea.”

## Observability

All decisions use structured `event {...}` records in the standard worker log.
The most useful event names are:

- `trigger.request` / `trigger.response`: exact NAT function inputs, outputs,
  and latency.
- `agent.observe.retry` / `agent.observe.skipped`: malformed model-generated
  tool arguments retried once or deferred to the next observation frame.
- `step.evidence`: caption, match decision, consecutive count, and threshold.
- `rag.lookup.request` / `rag.lookup.response`: exact retrieval query, latency,
  scores, sources, and returned passages when package instructions are incomplete.
- `agent.observe.request` / `agent.observe.response`: the complete compact
  state-agent context and result.
- `step.commit`, `step.commit_noop`, and `step.commit_rejected`: state deltas,
  completed-step no-ops, or validation failures.
- `agent.foreground.request`, `agent.foreground.response`, and
  `agent.foreground.retry`: selected root/tea foreground, exact compact input,
  lifecycle operation or direct answer, resulting foreground state, and
  corrected tool arguments.
- `step.ready`, `step.enter`, `workflow.start_noop`, `workflow.reset`,
  `workflow.complete`, and `notice.queued`: readiness, protected repeated
  starts, lifecycle resets, explicit transitions, and speech.

Use the log directory printed at startup, then filter a human test with:

```bash
rg '"participant_id":"<id>"' <worker.log>
rg '"trace_id":"<trace>"|step.commit' <worker.log>
rg 'step.evidence|commit_rejected|vision.unavailable' <worker.log>
```

Unexpected failures propagate out of the monitor or voice task and stop the
worker with their traceback. Missing live frames and invalid model patches are
handled because another observation can repair them.

## Development

Read [AGENT_GUIDE.md](AGENT_GUIDE.md) before changing behavior. It records the
state-machine invariants, prompt budgets, log-driven human-test loop, and the
sample-local pieces that should move to `xr-ai-nat` after a second application
proves their reusable contract.

Run the lightweight checks without models:

```bash
uv run --project agent-samples/tea-making-sample/worker \
  python -m unittest discover -s agent-samples/tea-making-sample/worker/tests -v
uv run --project agent-samples/tea-making-sample/worker \
  python agent-samples/tea-making-sample/eval/check.py
```
