<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Packet and executable-guide contract

## Recording packet

`packet.json` is the entry point. It contains session status, recording counts,
relative source-file paths, and an Activity → Phase hierarchy.

- `frames/index.jsonl`: frame ID, source timestamp, image path, source sequence,
  width, and height.
- `captions.jsonl`: a periodic observation linked to an exact frame, with
  activity, phase, caption, and delta.
- `transcript.jsonl`: final STT utterances on the same Unix-microsecond timeline.
- `summary.md`: live Activity → Phase table for fast orientation.
- `errors.jsonl`: optional failures. A gap is not proof that nothing happened.

Align speech and vision using source timestamps. `generated_at_us` is caption
inference time, not the pictured event time.

## Executable guide

The authoritative file is YAML ending in `.guide.yaml` or `.guide.yml`. It has
exactly four root fields: `schema_version`, `task`, `state`, and `steps`.
Unknown fields are rejected at every structural level.

`task` requires:

- `id`: stable lowercase identifier matching `^[a-z][a-z0-9_-]*$`.
- `name`: succinct human title of 1–5 words, separated by spaces, with no
  punctuation. Summarize the overall task or intended result rather than
  concatenating step descriptions. For example, use `Assemble toy vehicle`,
  not `Place case, wait, and attach shovel`. This naming rule applies to
  `task.name`, not the stable `task.id` or filename.
- `version`: positive integer, incremented for revisions.
- `status`: always `draft` when generated; only `approved` guides can run.
- `source_session`: packet session ID.
- `start_step`, `foreground_prompt`, and `complete_message`.

`state` maps identifiers to `{type, description, initial?}`. Supported types are
`boolean`, `integer`, `number`, and `string`. Initial and committed values must
match exactly; booleans are not integers.

`steps` is a non-empty, linear, acyclic list. IDs are unique, all steps must be
reachable from `task.start_step`, and `next` is either another step ID or null.
Each step requires:

- `reads` and `writes`: declared state field names. `writes` cannot be empty.
- `trigger`: `function`, positive `interval_s`, `arguments`, and optional
  `result_field`. Trigger arguments may reference visible state as
  `$state.<field>`.
- `agent` and `voice`: a non-empty `prompt` and a `tools` list.
- `complete_when`: a non-empty mapping using only fields in `writes`.
- `complete_on_skip`, `state_on_skip`, and enter/complete/skip `messages`.

Supported triggers are `current_view` and `clock__timer`. Supported policy tools
are `current_view`, `clock__now`, and `clock__timer`. Participant identity is
bound by the engine and never appears in model-authored arguments.

For `current_view`, `arguments.question` must contain 1–500 characters,
including spaces and punctuation. The limit applies to the parsed YAML string,
including folded text (`>-`), not to individual source lines. Shorten wording
without removing required visual checks or changing the accepted answer labels.

Optional `evidence` contains a full-match regex, a positive `consecutive` count,
and optional `commit` values. When `commit` is present, the engine applies it
directly after the evidence threshold, avoiding a second model judgment. Its
fields must be writable and typed. Without `commit`, completion proposed by the
observation agent remains gated by the same evidence count.

### Timer steps

Separate visual setup from elapsed-time checking, as in the tea-making sample.
In the preceding setup step, confirm the required visual prerequisite, call
`clock__now`, and save its exact positive `epoch_us` value in a declared integer
state field. Do not invent a timestamp or use a timestamp from the recording.
Do not route a skipped setup step into a timer whose start time is unset.

The waiting step must use `trigger.function: clock__timer`, with
`arguments.started_at_us: $state.<saved_start_field>` and a positive integer
`arguments.duration_s` taken from the demonstrated instructions or an explicit
user requirement. Use `result_field: expired`, `evidence.pattern: '^true$'`,
`evidence.consecutive: 1`, and `evidence.commit` to set the declared completion
boolean to true. Use that boolean in `complete_when`. This directly enforces
the timer result without an observation-model judgment; keep `agent.tools: []`.
For spoken timer questions, use `voice.tools: [clock__timer]` and instruct the
agent to report its fresh result using the same saved start time and duration.

Timer steps check elapsed time only. Do not add `current_view`, visual evidence
requirements, or visual resets to the waiting step. Put any required visual
checks before or after it. Timing runs from the saved timestamp, not from the
user saying next. Explain that this does not continuously verify the scene
during the wait; do not claim uninterrupted placement or contact. If the
recording requires continuous visual monitoring, flag that requirement as
unsupported by this elapsed-time-only pattern rather than silently dropping it.

### Runtime invariants

- One active guide per participant.
- The session pins guide ID, version, and SHA-256 content generation at start.
- Model work happens outside the state lock and commits require the expected
  step and revision.
- The model can only mutate the active step's declared write set.
- Completion never advances automatically; the user says `next` or `skip`.
- Catalog reloads affect new sessions only. Invalid and duplicate guides are
  indexed with errors and are never runnable.
