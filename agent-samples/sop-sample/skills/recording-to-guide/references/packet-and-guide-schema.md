<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Packet and guide contract

## Capture input

Start with `packet.json`: `session_id`, `status`, `narration_status`, `counts`,
`files`, and `hierarchy`. Resolve `files` paths relative to the
packet directory. A complete packet can still contain caption errors or no
narration; inspect the contents, not only its status.

| Source | Fields used for evidence |
|---|---|
| `frames/index.jsonl` | `frame_id`, `timestamp_us`, `path`, `sequence`, `width`, `height` |
| `captions.jsonl` | `caption_id`, `frame_id`, `frame_timestamp_us`, `frame_path`, `activity`, `phase`, `caption`, `delta` |
| `transcript.jsonl` | `transcript_id`, `timestamp_us`, `text` |
| `summary.md` and `hierarchy` | Derived orientation aids, not an exhaustive step inventory |
| `errors.jsonl` if present | Capture gaps; absence of observations is not proof of inactivity |

Source timestamps are Unix microseconds. Caption `generated_at_us` measures
inference completion, not event time. STT timestamps are alignment hints, not
precise action boundaries. Inspect nearby frames if timing is uncertain.
SOP snapshots, captions, and narration cover the spoken start/stop recording
boundary. Optional `--capture` media covers the whole participant connection;
it is not required for guide generation and may span multiple SOP packets.
Use packet-local narration, not the whole shared transcript. New packets have
no `media_capture` field. Older packets may link a bundle through
`media_capture.manifest`; retain that bundle if present. Raw audio may help review
missing narration, but generation does not automatically retranscribe it.
Do not fabricate speech for silent recordings.

## Executable YAML

Exactly four root fields: `schema_version: 1`, `task`, `state`, `steps`. Unknown
structural fields are rejected. Use block mappings and quote prose; commas in
unquoted flow mappings (`{...}`) can silently create unintended fields.

`task` requires `id`, `name`, `version`, `status`, `source_session`, `start_step`,
`foreground_prompt`, `complete_message`. Use a stable lowercase identifier
matching `^[a-z][a-z0-9_-]*$`, a 1–5 word punctuation-free name, a positive integer
version, and `status: draft`. Only human-reviewed `approved` guides may replay.
Name, ID, and filename are separate. Use the same intended task in all prompts
and completion messages; do not copy the worked example's wording.

`state` maps identifiers to `{type, description, initial?}`. Types are `boolean`,
`integer`, `number`, `string`. Initial and committed values must match the type;
`true` is not an integer and `"true"` is not a boolean.

`steps` is a nonempty linear list of unique IDs, all reachable from
`task.start_step`, without cycles. Each step contains:

| Field | Contract |
|---|---|
| `id`, `title` | Stable identifier and short instruction title |
| `reads`, `writes` | Lists of declared state fields; `writes` cannot be empty |
| `trigger` | `function`, positive `interval_s`, `arguments`, optional `result_field` |
| `agent`, `voice` | Each has a nonempty `prompt` and an allowed `tools` list |
| `evidence` | Full-match regex `pattern`, positive integer `consecutive`, optional typed `commit` mapping |
| `complete_when` | Nonempty typed mapping using only this step's writable fields |
| `next` | Next step ID, or `null` for the last step |
| `complete_on_skip` | Boolean; true ends the run if this incomplete step is skipped |
| `state_on_skip` | Typed mapping using only writable fields; normally `{}` |
| `messages` | Nonempty `enter`, `complete`, and `skip` strings |

`evidence.commit` uses writable fields, not a literal placeholder like
`done_field`. With it, matching evidence commits directly without an observation
LLM; that agent's prompt and tools do not execute. Without it, the observation
agent may call the engine-provided
`workflow__commit`, still gated by the evidence threshold. Do not list that
internal tool in `agent.tools`. Completion does not advance; messages ask the
user to say next. A skip is not verified success.

Give each step its own completion boolean, initially false. Do not reuse a
previous step's completion field: that would make the later step complete on
entry without its own observation. Capture success never initializes replay
completion to true. The visual example links two independent checks; it does
not prescribe two steps for every recording.

## Closed tool catalog

| Name | Trigger | Policy tool | Arguments and result |
|---|---|---|---|
| `current_view` | Yes | Yes | `question`: 1–500 characters; result field `text` |
| `clock__now` | No | Yes | No arguments; returns positive integer `epoch_us` |
| `clock__timer` | Yes | Yes | Positive integers `started_at_us`, `duration_s`; result fields `elapsed_s`, `remaining_s`, `expired` |

No participant ID, image path, sequence window, or `max_frames` argument is
accepted. The engine binds the participant. The 500-character question limit
applies to the parsed string including spaces and folded YAML, not each line.
Use `result_field: text` for visual checks and `result_field: expired` for waits.
Visual policies normally use `agent.tools: []`, `voice.tools: [current_view]`.
`agent` handles background observations; `voice` answers current-step questions.
Declaring a tool does not call it; the prompt must give the needed arguments.

Trigger arguments can use `$state.<field>`. The field must be declared in state
and in that step's `reads` or `writes`, with the argument's required type.
These references are not general template substitutions for message text.

For waits, read [the full timer example](timer.guide.yaml). Setup observes the
prerequisite, calls `clock__now`, and commits its exact returned timestamp plus
a separate setup boolean. Do not use `evidence.commit` on setup, since its clock
value is dynamic. Set setup `complete_on_skip: true` so a skip cannot enter an
uninitialized timer. The wait uses `clock__timer`, `result_field: expired`,
`pattern: '^true$'`, `consecutive: 1`, and direct commit of its own boolean.
Keep wait `agent.tools: []`, `voice.tools: [clock__timer]`; no visual requirement.
Timing starts at setup, not when the user says next. It does not prove continuous
contact. Unknown, ranged, or fractional durations require review instead of
guessing an accepted positive integer.

## Evidence gaps and review-blocked steps

Use normal live checks for clearly specified observable results, even when a
saved image is missing or a caption is stale. Record that authoring gap in the
review notes; it does not make the replay tool incapable of checking the result.
This includes an explicitly instructed count or orientation: absent visual
corroboration is not an unresolved instruction. Decide from the required result.
For example, an explicit placement instruction with an unavailable saved image
can still have a `current_view` placement check with `^ready$` evidence.

Reserve a blocked step for an unresolved required condition or an unsupported
verification, such as proving past ordering or a hidden fit from one image.
Retain the action, not a fictitious verification.
An unclear object identity can require this pattern without preventing a draft
of the known action and target. Refer to
[the complete blocked example](review-blocked.guide.yaml). Do not substitute a
request for clarification for both requested output files when an action is known.
Use a bounded `current_view` question about the observable context, a separate
false-initialized completion field, and this never-matching evidence gate:

```yaml
evidence:
  pattern: '(?!)'
  consecutive: 1
```

Omit `evidence.commit`; set `agent.tools: []` and instruct the agent not to commit
the step because its requirement cannot be verified. The gate prevents completion
even if the model proposes it. Set `complete_on_skip: true` if later steps depend
on this unresolved prerequisite. Explain the exact gap in the step prompts and
review notes. Do not approve until it is resolved; schema validity alone is not
approval. Never put invented triggers or extra `review_blockers` keys in the YAML.
An ordinary ordered step list does not itself require a temporal tool: final
visible states can verify ordinary placements. Block when the required success
condition is the past motion or order itself, rather than the visible result.

Store evidence mappings, caption corrections, and approval blockers in the
separate `.review.md` file. It is a human review aid, not executable guide state.
Keep authoring changes in those notes. The
[evaluation skill](../../evaluate-guide/SKILL.md) writes its observed outcomes,
untested checks, and revision findings separately in `<slug>.evaluation.md`.
Do not add test results to YAML.
Write these outputs under `agent-samples/sop-sample/guides/` from the repository
root, unless a different output directory was requested. Links between these
references resolve relative to their containing files, not the working directory.
