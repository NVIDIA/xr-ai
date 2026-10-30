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

A finalized `complete` packet contains `packet.json`, `summary.md`, and
`transcript.jsonl`; the transcript can be empty. The frame index and caption log
are created only when observations are saved. An absent `frames/index.jsonl`
with `counts.frames: 0`, or absent `captions.jsonl` with `counts.captions: 0`,
means no evidence of that kind was saved. A missing log with a nonzero count
is inconsistent: report it rather than treating it as empty. `errors.jsonl`
is optional; its absence does not establish sufficient evidence. An `incomplete`
packet can lack required files. Check status, counts, and file availability
before using the packet; `complete` means writes finished, not that the
demonstration succeeded or contains every required action.

| Source | Fields used for evidence |
|---|---|
| `frames/index.jsonl` | `frame_id`, `timestamp_us`, `path`, `sequence`, `width`, `height` |
| `captions.jsonl` | `caption_id`, `frame_id`, `frame_timestamp_us`, `frame_path`, `activity`, `phase`, `caption`, `delta` |
| `transcript.jsonl` | `transcript_id`, `timestamp_us`, `text` |
| `summary.md` and `hierarchy` | Derived orientation aids, not an exhaustive step inventory |
| `errors.jsonl` if present | Capture gaps; absence of observations is not proof of inactivity |

Fields ending in `_us` use Unix microseconds; `started_at` and `ended_at`
use UTC ISO 8601 timestamps. Caption `generated_at_us` measures
inference completion, not event time. STT timestamps are alignment hints, not
precise action boundaries. Inspect nearby frames if timing is uncertain.
Newer captions also contain pipeline-generated `delta_reference_frame_id` and
`delta_reference_timestamp_us`: the earlier image actually supplied to the VLM.
Both are `null` for an initial single-image observation. Resolve the reference
through the frame index. Sampling can outpace captioning, and failed captions
do not advance the comparison image, so do not subtract one from `frame_id`.
Older packets may omit these fields; do not invent an exact comparison pair.
Reference metadata identifies inputs, not the truth of the model's delta.
Keep comparison IDs and source times in review notes, not executable guide state.
No meaningful visible change differs from an indeterminate comparison: neither
establishes inactivity throughout the interval. Uncertain labels are not actions
and must not erase actions supported by narration or inspected images.
Capture commands take effect after speech recognition and command processing,
not at exact spoken-word boundaries. `started_at` records the accepted start
command's transcript timestamp, not a guarantee that sampling began then.
`stop_command.timestamp_us` records the accepted stop utterance's transcript
timestamp; saved frames can extend beyond it while the command is processed.
`ended_at` is null during capture and records the effective closure after
producers stop and evidence admission closes, before pending writes and summaries
finish draining. `stop_command` is null if no spoken stop was accepted, such as
when disconnecting finalizes capture. These boundaries do not prove task completion.
Commands are excluded from narration; a late transcript can miss admission even
if the user spoke before stopping. Inspect nearby evidence and report gaps rather
than assuming exact alignment or discarding frames after the stop timestamp.

Optional session media is not required for guide generation and may span
multiple SOP packets.
Use packet-local narration, not the whole shared transcript. New packets have
no `media_capture` field. Older packets may link a bundle through
`media_capture.manifest`; retain that bundle if present. Raw audio may help review
missing narration, but generation does not automatically retranscribe it.
Do not fabricate speech for silent recordings.

## Executable YAML

Exactly four root fields: `schema_version: 2`, `task`, `state`, `steps`. Unknown
structural fields are rejected. Use block mappings and quote prose; commas in
unquoted flow mappings (`{...}`) can silently create unintended fields.

`task` requires `id`, `name`, `version`, `status`, `start_step`,
`foreground_prompt`, `complete_message`. Use a stable lowercase identifier
matching `^[a-z][a-z0-9_-]*$`, a 1–5 word punctuation-free name, a positive integer
version, and `status: draft`. Only human-reviewed `approved` guides may run in guide mode.
Name, ID, and output path are separate. Save the guide as `<slug>-guide/guide.yaml`,
with a lowercase, hyphen-separated slug derived from its name, for example
`assemble-toy-guide/guide.yaml`. Keep `review.md` and `evaluation.md` in that
same directory, outside the executable YAML.
Use the same intended task in all prompts
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
| `voice` | Nonempty `prompt` and allowed `tools` list for conversation |
| `agent` | Optional for timer steps, required without a direct commit; visual steps reject it |
| `evidence` | Visual steps require a text `condition` for CLEF; timer steps use a full-match `pattern`. Both use positive integer `consecutive` and an optional typed `commit` mapping |
| `complete_when` | Nonempty typed mapping using only this step's writable fields |
| `next` | Next step ID, or `null` for the last step |
| `complete_on_skip` | Optional boolean, default false; true ends the run if this incomplete step is skipped |
| `finish_on_complete` | Optional boolean, only for a terminal step; true ends the run automatically after verified completion |
| `state_on_skip` | Optional typed mapping using only writable fields; defaults to `{}` |
| `messages` | Nonempty `enter`, `complete`, and `skip` strings |

For `current_view`, ask Cosmos for a focused description of the relevant region,
objects, contacts, labels, and hidden details. State the completion requirements
in `evidence.condition`. CLEF receives that condition and the description, not
the image, conversation history, or a suggested answer. Only yes counts toward
`consecutive`. Repeated frames do not count twice, and failures break the streak.
Do not use `pattern` or `agent` on visual steps. Unavailable or uncertain evidence
must not be treated as success, but both models can still misinterpret a scene.

`evidence.commit` uses writable fields, not a literal placeholder like
`done_field`, and must satisfy `complete_when`. After the required confirmations,
the runtime applies this bounded patch; CLEF cannot invent state updates.
Approved visual steps require a commit. A review-blocked draft may omit it;
then no decision, even yes, can complete that step. Timer `pattern` matching and
runtime-owned elapsed-time enforcement are unchanged. A timer without a direct
commit may still use an observation `agent`; do not list its internal
`workflow__commit` in `agent.tools`. Ordinary completion does not advance; messages ask the
user to say next. A terminal `finish_on_complete: true` step ends the run without
Next; its message must not request another advance. A skip is not verified success.

Give each step its own completion boolean, initially false. Do not reuse a
previous step's completion field: that would make the later step complete on
entry without its own observation. Capture success never initializes guide
completion to true. The visual example links two independent checks; it does
not prescribe two steps for every recording.

## Closed tool catalog

| Name | Trigger | Policy tool | Arguments and result |
|---|---|---|---|
| `current_view` | Yes | Yes | `question`: 1–500 characters; result field `text` |
| `clock__now` | No | Yes | No arguments; returns positive integer `epoch_us` |
| `clock__timer` | Yes | Yes | Trigger: positive integer `duration_s`; policy tool: no arguments, reads the active runtime-owned timer. Result fields `elapsed_s`, `remaining_s`, `expired` |

No participant ID, image path, sequence window, or `max_frames` argument is
accepted. The engine binds the participant. The 500-character question limit
applies to the parsed string including spaces and folded YAML, not each line.
Use `result_field: text` for visual checks and `result_field: expired` for waits.
Visual steps omit `agent`; CLEF handles background decisions.
`voice.tools: [current_view]` lets the conversation LLM answer current-step questions.
Declaring a tool does not call it; the prompt must give the needed arguments.

Trigger arguments can use `$state.<field>`. The field must be declared in state
and in that step's `reads` or `writes`, with the argument's required type.
These references are not general template substitutions for message text.

For waits, read [the full timer example](timer.yaml). Setup observes the
prerequisite and commits its own boolean. Set setup `complete_on_skip: true`
when a skip must not enter a wait without that prerequisite. The wait uses
`clock__timer` with only `duration_s`, `result_field: expired`,
`pattern: '^true$'`, `consecutive: 1`, and direct commit of its own boolean.
Keep wait `agent.tools: []`, `voice.tools: [clock__timer]`; no visual requirement.
The runtime starts a monotonic timer on entry to the wait step, not at the earlier
placement check. The policy tool takes no arguments and cannot change that start.
Do not supply `started_at_us`, save a start in model-writable state, or use
`clock__now` to initialize a timer. Timing does not prove continuous contact.
Unknown, ranged, or fractional durations require review instead of
guessing an accepted positive integer.
If an instruction requires timing from a physical event rather than wait-step
entry, report the mismatch: user-controlled Next can introduce a delay. Do not
claim the runtime measured from the event or silently change that requirement.

## Evidence gaps and review-blocked steps

Use normal live checks for clearly specified observable results, even when a
saved image is missing or a caption is stale. Record that authoring gap in the
review notes; it does not make the guide tool incapable of checking the result.
This includes an explicitly instructed count or orientation: absent visual
corroboration is not an unresolved instruction. Decide from the required result.
For example, an explicit placement instruction with an unavailable saved image
can still have a `current_view` placement check with an explicit visible placement condition.

Reserve a blocked step for an unresolved required condition or an unsupported
verification, such as proving past ordering or a hidden fit from one image.
Retain the action, not a fictitious verification.
An unclear object identity can require this pattern without preventing a draft
of the known action and target. Refer to
[the complete blocked example](review-blocked.yaml). Do not substitute a
request for clarification for both requested output files when an action is known.
Use a bounded `current_view` question about the observable context, a separate
false-initialized completion field, and a condition without a commit:

```yaml
evidence:
  condition: The adapter identity and required fit must be resolved by human review.
  consecutive: 1
```

State the unresolved requirement in `evidence.condition` and omit
`evidence.commit`. Do not add an `agent` or a regex gate. No CLEF decision can
complete a step without a commit, and the parser rejects approving such a guide.
Set `complete_on_skip: true` if later steps depend
on this unresolved prerequisite. Explain the exact gap in the step prompts and
review notes. Do not approve until it is resolved; schema validity alone is not
approval. Never put invented triggers or extra `review_blockers` keys in the YAML.
An ordinary ordered step list does not itself require a temporal tool: final
visible states can verify ordinary placements. Block when the required success
condition is the past motion or order itself, rather than the visible result.

Store evidence mappings, caption corrections, and approval blockers in the
separate `review.md` file. It is a human review aid, not executable guide state.
For consequential assumptions, explain their effect on the procedure or success
criteria and the question needing human confirmation. Distinguish uncertainty
about requirements from recording coverage gaps and unsupported verification;
do not silently resolve the former or weaken requirements to fit the recording.
Continue drafting sufficiently understood actions without seeking confirmation
for ordinary organization and wording.
Keep authoring changes in those notes. The
[evaluation skill](../../evaluate-guide/SKILL.md) writes its observed outcomes,
untested checks, and revision findings separately in `evaluation.md`.
Do not add test results to YAML.
Runtime evaluation uses the guide runtime and configured local models with
recorded camera input. The authoring model observes actual traces and revises
the guide; it does not replace visual inference, tool results, or state commits.
Test artifacts and recording timestamps stay outside the executable guide.
Write these outputs under `agent-samples/procedure-capture-guide/<slug>-guide/`
from the repository root, unless a different output directory was requested. Links between these
references resolve relative to their containing files, not the working directory.
