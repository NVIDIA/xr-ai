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

Voice uses a separate hierarchy:

```text
utterance -> router NAT agent -> management function OR current step voice agent
```

There is no custom tool-call loop. NeMo Agent Toolkit builds the router, every
step agent, every voice agent, and their step-specific tool lists. Current and
historical conversation turns are not added to prompts. Each call receives only
the active step's projected state and current input. The router reserves
management functions for explicit start, next/continue/skip, stop/reset, and
workflow-step status requests. Task questions, action reports, correctness
checks, current readings, and timer questions always delegate to the active
step voice agent. Explicit next/continue/advance/skip commands always select the
advance function and are never delegated as task questions.

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
Heating comparisons normalize a visible Fahrenheit reading to Celsius before
checking whether it meets or exceeds the Celsius target; the raw numbers are
never compared, and a target is never reused as the current reading. A reading
below target leaves readiness unchanged. An active-heater indicator or
unit-bearing temperature display immediately records the durable
heating-started milestone, even when the water is below target.
The observation agent delegates conversion and comparison to the deterministic
`temperature__verify` NAT tool and follows its boolean result.
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
“continue,” or “skip” request lets the top-level router advance it.
Participant state is ephemeral. A real client join, including the roster replay
after an app restart, resets guidance to idle; leaving and reconnecting creates
a fresh session. Duplicate roster replays for an already-known connection are
ignored so they cannot erase an active walkthrough.

## Model modes

The launcher requires both model and voice behavior explicitly. With no
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
  --model-mode omni --voice-mode always-on
```

Select Cosmos Reason1 for vision and Nemotron-3-Omni for agents with:

```bash
uv run --project agent-samples/tea-making-sample tea_making_sample \
  --model-mode cosmos --voice-mode wake-word
```

`--voice-mode always-on` accepts every transcribed utterance.
`--voice-mode wake-word` requires each command to start with “Agent” or “Hey
Agent”; saying only the wake phrase opens a five-second follow-up window. The
launcher selects dedicated gate profiles and writes temporary worker and RAG
configs, so switching modes never edits source YAML and both processes always
use the same model profile.

Both profiles disable hidden reasoning for Omni agent calls. The Omni vision
profile also caps continuous caption generation; agent calls retain their
tool-loop budget. A timed-out caption is
logged as `vision.timeout` and the next frame continues the same observation
loop. Both model modes reuse Parakeet STT and Nemotron Embed while the sample
manages Piper TTS, the typed RAG service, the media hub, and its worker.

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
- `step.evidence`: caption, match decision, consecutive count, and threshold.
- `rag.lookup.request` / `rag.lookup.response`: exact retrieval query, latency,
  scores, sources, and returned passages when package instructions are incomplete.
- `agent.observe.request` / `agent.observe.response`: the complete compact
  state-agent context and result.
- `step.commit`, `step.commit_noop`, and `step.commit_rejected`: state deltas,
  completed-step no-ops, or validation failures.
- `agent.router.*`, `voice.delegate`, and `agent.voice.*`: voice routing and
  step answer traces.
- `step.ready`, `step.enter`, `workflow.reset`, `workflow.complete`, and
  `notice.queued`: readiness, lifecycle resets, explicit transitions, and speech.

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
