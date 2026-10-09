<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP sample

The SOP sample records narrated demonstrations between spoken commands. It saves
source evidence for a later procedural guide; it does not generate, approve,
or execute guides. Guide replay is not implemented in this branch; an external coding agent can
generate a draft from saved packets using the instructions below.

## Run a demonstration

Run commands from `agent-samples/sop-sample/`:

```bash
uv run --project ../../model-server-samples/model-servers model_servers
uv run main.py
```

Wait for the shared models to be ready before starting the sample. Open the
authenticated web-client URL printed by DeviceIOHub, enable the microphone,
and connect. Say **start recording**, demonstrate and narrate, then say
**stop recording**. Say **start recording** again for another SOP packet without
reconnecting. The worker logs each finalized packet's path and status.

Live video and on-demand image capture are both supported. Camera-off does not
stop recording. The worker makes no spoken replies, including during narration;
there is no wake phrase or command acknowledgement.

To additionally save participant-wide video, bidirectional audio, and data-channel
traffic, use the same optional flag as Simple VLM and XR Render:

```bash
uv run main.py --capture
```

This flag enables the shared capture service in participant mode. Its bundle
starts on participant connection and ends on disconnect or service shutdown,
independently of SOP recording commands. One bundle may span several SOP packets.
The default demo profile produces MP4 output when video is available and retains
the shared source media and timeline. With no streamed video, it may contain only
audio, transcripts, and metadata. On-demand stills are not synthesized into video.

Refer to {doc}`/getting_started/requirements` for model and hub deployment and
{doc}`/components/server-runtime` for shared media capture. Optional video
recording uses the shared NVENC encoder and requires supported NVIDIA hardware.
Without `--capture`, the SOP worker saves snapshots, captions, and narration
without launching that encoder. STT and VLM endpoints are configured in
`yaml/models.json`. The worker creates an unused TTS client to satisfy the shared
voice runtime contract, but makes no TTS synthesis requests.

## Recording boundaries

- Connecting does not start SOP frame sampling, captioning, or narration storage.
- **start recording** starts a fresh SOP packet. Repeated starts while recording
  are ignored; **stop recording** finalizes the active packet.
- Commands are standalone utterances, ignoring case, surrounding whitespace, and
  trailing sentence punctuation. There is no wake phrase. Other speech, including
  “stop,” “stop capture,” and “do not stop recording,” is narration only while
  recording. The two control commands are excluded from SOP narration.
- Camera mute, unmute, publication, and frame gaps never start or stop a packet.
  With no fresh live frame, the existing current-frame tool requests a still from
  clients that allow on-demand capture; this can briefly activate the camera.
- Unavailable images do not end recording or narration. Captioning resumes only
  when another distinct image has been saved.
- Disconnect and graceful shutdown finalize open packets. A reconnect is idle
  until another **start recording** command. Stale events from an old connection
  cannot stop a new recording.

Speech controls take effect after final recognition, not at the first spoken
word. Keep the microphone enabled and pause after the last utterance before
disconnecting so STT can finish. With `--capture`, raw audio can remain available
even if a late transcript misses its SOP boundary.

Stop drains pending filesystem writes and finalizes the packet before the next
queued start. Narration and commands are ordered per participant, so speech
following a queued start belongs to that new packet. Snapshots during the restart
gap are not buffered. Departure cancels queued starts but drains accepted
recordings and their queued narration. No raw-media manifest or encoder cleanup
is needed to complete an SOP packet.

## Outputs

Generated data remains local and Git-ignored. Defaults are relative to the sample:

| Location | Contents |
|---|---|
| `artifacts/sessions/<session-id>/packet.json` | SOP status, participant, timestamps, counts, and activity hierarchy |
| `artifacts/sessions/<session-id>/frames/` | JPEG samples targeting 2 fps and timestamped `index.jsonl` |
| `artifacts/sessions/<session-id>/captions.jsonl` | Timestamped captions, activities, phases, and visible changes |
| `artifacts/sessions/<session-id>/transcript.jsonl` | Final user transcripts inside the SOP recording boundary |
| `artifacts/sessions/<session-id>/summary.md` | Human-readable activity and phase summary |
| `artifacts/sessions/<session-id>/errors.jsonl` | Frame, caption, or persistence errors, when present |
| `artifacts/captures/<bundle>/` | Optional participant-wide media, transcripts, timeline, and manifest from `--capture` |

The captioner checks for a new sampled frame every 5 seconds by default, with
the previous caption as context. It does not recaption the same saved image.
On-demand cadence depends on client response time; 2 fps is a target, not a
guarantee. Captions are observations, not verified SOP steps.

Narration reuses `VoiceAgent`'s final transcript events, also published to the
optional shared capture service. There is no second STT implementation. The
shared transcript retains speech outside SOP boundaries and the spoken commands;
the SOP transcript contains only narration between them.

An accepted stop is recorded in `packet.json.stop_command`. Packets no longer
contain a per-SOP `media_capture` manifest link: shared bundles use independent
participant-wide boundaries. Match them using participant identity and timestamps
when needed. Packet `complete` means its local writes finished, not that every
frame or utterance was available or the procedure was correct. Review errors and
counts before generating a guide. Retention is disabled by default; monitor disk
usage and remove unwanted recordings manually.

## Generate a draft guide

After the packet is finalized, use a coding agent with local file and image
access to follow
[`recording-to-guide`](https://github.com/NVIDIA/xr-ai/blob/main/agent-samples/sop-sample/skills/recording-to-guide/SKILL.md).
This is the manual generation stage. `recording-to-guide` owns direct YAML
authoring and revisions. The companion
[`evaluate-guide`](https://github.com/NVIDIA/xr-ai/blob/main/agent-samples/sop-sample/skills/evaluate-guide/SKILL.md)
checks a fixed draft against the original capture and returns findings without
editing it. Both use the same schema and tool contract. The workflow is generate,
evaluate, revise, and reevaluate before handoff for human review.
There is no Python YAML generator, plan compiler, automatic approval, or new
model-server requirement. Selecting GPT Luna happens in the coding agent, not
in the capture sample's `models.json`.

From the repository root, give the coding agent this request, replacing the
session placeholder:

```text
Read agent-samples/sop-sample/skills/recording-to-guide/SKILL.md and follow it.
Use agent-samples/sop-sample/artifacts/sessions/<session-id>/packet.json.
Use this packet's narration; optional full-session media is not required.
Keep the repository root as your working directory for reads, writes, and validation.
Read the schema and applicable examples linked relative to the skill directory.
Write the demonstrated procedure as guide YAML directly, plus evidence
review notes, under agent-samples/sop-sample/guides/. Reuse captions and narration;
inspect selected saved images when needed. For each step, combine its instructions
and chronological captions to extract the action, object, target, and required
count, orientation, order, or duration. Distinguish in-progress states from the
completed result. Merge repeated views of one action without losing distinct
actions or required instances. Use this evidence mapping for both the user
instruction and completion question. Distinguish a missing capture image from an
unresolved requirement: the former needs a review note, the latter may need a
blocked draft step.
Run the validator, then use
agent-samples/sop-sample/skills/evaluate-guide/SKILL.md to evaluate the fixed draft
against the original capture. Prefer a fresh evaluator context when available;
give it the guide and original evidence, not the author's self-review or expected
answers. The evaluator writes findings without changing the guide. Use those
findings, including its literal dispatch and state trace, to revise the guide's
prompts and steps. Use at most three evaluation-and-update rounds by default,
stopping early when no correctable failures remain. Revalidate every update and
evaluate the final revision before handoff; the final check is not another update
round. Never weaken requirements to fit the demonstration. Report the actual
evaluation-and-update round count, guide, authoring review, and evaluation paths,
actual
results, changes, untested checks, and approval blockers. Keep the guide draft;
do not approve it, launch live replay, alter source recordings, or generate a
YAML-writing program.
```

No skill installation is required for this explicit-file workflow. To install
the skills into an agent's skill directory, copy the **entire** `recording-to-guide/`
and `evaluate-guide/` directories as siblings, including their references and
agent metadata, rather than only `SKILL.md`. The evaluation skill links to the
authoring skill's shared contract and examples.
Keep the target repository and sample directory explicit when using an installed
copy. An agent without an image tool can draft from available evidence, but it
must report uninspected visual ambiguities rather than claim they are resolved.

The outputs are local and Git-ignored:

- `guides/<slug>.guide.yaml`: authoritative, model-authored draft guide.
- `guides/<slug>.review.md`: the author's evidence mapping, caption corrections,
  revisions, and unresolved requirements.
- `guides/<slug>.evaluation.md`: evaluation findings, guide identity, observed
  check results, untested checks, and approval blockers. Neither Markdown file
  is executable guide state.

Generated task names use 1–5 punctuation-free words. The skill preserves narrated
actions, prerequisites, counts, and orientations; it does not use activity labels
as an exhaustive list of steps. It combines chronological captions into actions,
not one step per caption, and maps each action's object, target, and constraints
to an instruction and a completion question. It inspects selected original
frames when necessary. Missing narration or occluded actions remain evidence
gaps. A schema-valid guide can still omit a physical requirement, so human review
must compare both the instructions and completion questions with the recording.

### Validate and review

From the repository root:

```bash
uv run --project agent-samples/sop-sample/worker python -m sop_sample_worker._validate_guide agent-samples/sop-sample/guides/<slug>.guide.yaml
```

The read-only validator checks the same guide contract used by the SOP replay
work: state types and write sets, linked steps, allowed tools, trigger arguments,
result fields, and the 500-character visual question limit. Text fields must be
actual strings. Duplicate mapping keys are rejected at every level, including
overrides introduced through YAML merges, instead of silently replacing earlier
requirements. It reports errors
with a nonzero exit status. It does not generate, rewrite, approve, or execute
the guide, and it does not certify evidence coverage or physical correctness.

Before handoff, `evaluate-guide` checks the fixed draft against the saved
recording, including incomplete and completed states, action coverage, tool
dispatch, transitions, and timer boundaries. It does not revise the guide.
`recording-to-guide` applies justified corrections, validates, and requests
another evaluation. The default is at most three evaluation and revision passes,
stopping early when no correctable failures remain. The latest YAML must be
checked after edits; missing evidence and unresolved failures are reported.

For evaluation alone, provide the original packet and guide paths to
`evaluate-guide`. Prefer a fresh evaluator context without the author's
self-review or claimed test results. If the same context is used, disclose it.
The evaluator reports the guide's content hash and task version to distinguish
findings about different revisions. Evaluation does not grant approval.

This is an offline authoring-model image/state walkthrough, not a built-in
recorded-media replay runner or a measurement of the production VLM. It does not
launch the live sample or make a draft approved. Timer boundary calculations are
labeled as simulations, and unavailable visual tests are marked untested.
Recorded replay cannot prove generalization to other demonstrations.

Review the evidence notes and resolve every blocker before manually changing
`task.status` from `draft` to `approved`. Preserve the original capture. When
revising a guide, increment `task.version` and check that the title, step prompts,
and completion messages still describe the same procedure. Execute the approved
guide by launching the sample in replay mode with `--replay "Guide Name"`.

### Verification limits

The generation contract exposes `current_view`, `clock__now`, and
`clock__timer`. A current view verifies visible state, counts, and orientation;
it cannot prove a hidden fit or the order of past actions. Timer setup captures
the live clock after a visual prerequisite, followed by an elapsed-time-only
waiting step. Neither this pattern nor repeated positive images proves
uninterrupted contact.

The skill does not emit experimental sequence triggers or arbitrary tool names.
For requirements the contract cannot verify, it preserves the action in a draft
step with a never-matching evidence gate and records the missing capability for
review. Comparing several saved images during generation is not the same as
verifying several replay images at runtime. Do not approve a guide by simply
removing such a blocker or weakening the required condition.

### Check skill changes

Treat the request above, both skills, the shared schema, and the worked examples
as one generation package. Review and evaluate them together; a skill-only test does
not qualify the documented workflow. Preserve a snapshot or hashes of all these
files with each run, and rerun the same cases when any part changes.

The skill incorporates lessons from smaller-model evaluations: explicit action
coverage, selective image inspection for stale captions, concrete verification
questions, consistent state-field names, and block-style YAML. These instructions
reduce known failure opportunities; they are not a measured accuracy guarantee
for a particular model version.

Synthetic evaluation cases live in
`tests/fixtures/sop_guide_generation/cases.yaml`. They cover missed narrated
additions, multi-caption action grouping, prerequisites and counts, uncertain
transcription, elapsed waits,
unsupported temporal verification, and incomplete demonstrations during recorded
replay. Each contains capture excerpts and a
semantic review checklist, not a ready-made guide or third-party recording.

For an end-to-end comparison, copy the package, unchanged validator, and one
capture into an isolated workspace with the documented paths. Use the exact
request above, changing only `<session-id>`. Allow file reads, selective image
inspection, direct YAML edits, and validator-driven repairs. Synthetic cases
can be materialized as packet, caption, and transcript files from `input` only;
keep `checks` outside the model workspace. Include real saved captures when
evaluating visual inspection. Do not replace this request with an inline
"return YAML only" prompt or a plan compiler.

Validate the output independently, then review action coverage, actual trigger
questions, timer initialization, uncertainty gates, and evidence references.
Check recorded-replay coverage, reported versus actual observations, whether
changes were retested, and whether failures were hidden by weakening requirements.
Record model and reasoning setting, package hashes, prompt, raw tool trace,
validation attempts, approval blockers, elapsed time, and reported token usage.
Count the complete authoring turn, including inspection and repair; cached
input is a subset of total input. Separate environment failures from guide
failures. Schema validity is not semantic accuracy, and a shorter prompt is not
an improvement if it loses actions or disables otherwise usable checks.

Run the offline schema and validator regressions from the repository root:

```bash
uv --config-file uv.toml run --project tests pytest tests/test_sop_guide_generation.py
```

## Composition and configuration

The orchestrator starts the hub, optional `device_io_capture`, and SOP worker.
Shared capture uses `session_mode: participant` and `profile: demo`, as in the
other samples. The worker never sends `CaptureTools` start or stop commands.

The worker subscribes only to existing participant events and final voice
transcripts. It has no camera lifecycle subscription. `VoiceAgent` owns STT;
the sample handles recording commands and narration in per-participant order.
Filesystem writes and finalization tasks drain before the frame endpoint closes.
No conversational agent or recording speech producer is added.

The voice gate disables optional STOP handling, and the worker disables early
VAD transcript probes. Ordinary STOP words cannot trigger spoken acknowledgements
or early STOP interruptions. Other samples keep their existing defaults.

`yaml/worker.yaml` owns SOP output paths, sampling, caption cadence, and speech
detection. `yaml/media_capture.yaml` independently configures the optional shared
bundle. Paths resolve relative to their owning YAML. `yaml/device_io_hub.yaml`
owns the room and web server; `yaml/models.json` owns model endpoints. Restart
after edits. Refer to {doc}`configuration` and {doc}`command-line` for fields and CLI
options.
