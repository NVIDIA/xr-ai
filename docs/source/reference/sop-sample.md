<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP sample

The SOP sample records narrated demonstrations between spoken commands. It saves
source evidence for a later procedural guide. Use `--replay "Guide Name"` to run
one approved guide instead. Optional `--capture` works with either worker.

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

## Replay an approved guide

Place a reviewed `*.guide.yaml` or `*.guide.yml` file under `guides/` in the
sample directory. Guides remain local and ignored by Git. Set `task.status` to
`approved` only after reviewing the instructions, completion checks, timers,
and safety constraints. The CLI matches an exact, case-insensitive `task.name`
or `task.id`; it rejects missing, draft, invalid, and ambiguous selections.
Quote names containing spaces:

```bash
uv run main.py --replay "Guide Name"
```

Start the shared models first, as for capture. Replay uses the LLM, VLM, STT,
and TTS endpoints in `yaml/models.replay.json`. Connect using the hub's web
client URL and enable microphone and camera access. Each participant starts at
step one with independent progress. No wake phrase or spoken start command is
required. Turning off live video does not reset the guide. The shared current
frame tool may request a still image from clients that support on-demand capture;
if no fresh image is available, visual verification waits. Timer checks continue
without a camera. Disconnect releases
that participant's progress and pending work; reconnect starts a fresh run.

Like {doc}`tea-making-sample`, replay separates foreground conversation from
background verification. The foreground answers guide questions using the
current step and its allowed tools. It cannot directly commit completion.
Periodic observations use a fresh camera view or a deterministic elapsed-time
check. The observer can commit only declared writable fields, with any required
consecutive evidence. Late observations from a previous step or run cannot
change current progress. The agent announces step completion once, without
requiring a user question, but does not automatically advance to the next step.

| Request | Behavior |
|---|---|
| “What should I do?” | Explain the current step |
| “Does this look right?” | Use the current step's visual tool if available |
| “How much time remains?” | Use the current step's timer tool if available |
| “Next” or “continue” | Advance only if the current step is verified complete |
| “Skip” | Explicitly bypass the current step; this is not evidence of success |
| “Status” | Report current progress |
| “Restart,” “reset guide,” or “stop guide” | Clear progress and return to step one of the selected guide |

After “next” on the completed final step, the agent announces workflow
completion, restores the guide's initial state and timer values, and presents step one again.
No relaunch or reconnect is needed. Explicitly skipping to the end also resets
the run, with an explicit skipped-steps message rather than a verified-completion
claim. Disconnect to stop receiving guidance. Recording commands and selecting
another guide are not supported inside replay mode.

The worker pins the validated guide and SHA-256 from the same captured bytes at
startup. File edits do not change a running replay; restart the sample to use a
new version or revoke approval. Approval is a local review convention, not a
cryptographic signature. Model-based visual checks can be wrong; reviewers must
describe observable evidence, including orientation and count where relevant.
This demo is not a safety interlock. A still image cannot prove a hidden action.

### Guide structure

The declarative shape follows the tea-making workflow: a task, typed sparse
state, and ordered steps with separate observation and voice policies. Existing
schema-version-1 guides from the workflow-recorder sample are supported.

Text fields must be actual strings. Duplicate mapping keys are rejected at
every level, including overrides introduced through YAML merges, instead of
silently replacing earlier requirements. Invalid guides cannot start replay,
even when their `task.status` is `approved`.

- `schema_version` must be `1`.
- `task` declares `id`, `name`, positive `version`, `status` (`draft` or
  `approved`), `source_session`, `start_step`, `foreground_prompt`, and
  `complete_message`.
- Each `state` field declares a type (`boolean`, `integer`, `number`, or
  `string`), a description, and optionally an initial value.
- Each step declares `id`, `title`, `reads`, `writes`, `trigger`, `agent`,
  `voice`, a non-empty `complete_when` mapping, `next`, and `messages` containing
  `enter`, `complete`, and `skip`. The `next` chain must visit every step once
  and end with `null`.
- `agent` and `voice` each contain a prompt and a tool list. Allowed step tools
  are `current_view`, `clock__now`, and `clock__timer`. Only the observer receives
  `workflow__commit`. Foreground workflow controls are supplied by the runner,
  not listed in the guide's step tools.
- Optional `evidence` contains a full-match regex `pattern`, positive
  `consecutive` count, and optional `commit` mapping. With `commit`, matching
  observations directly apply that bounded patch without an observation LLM.
  Without `commit`, the observation LLM proposes state changes and the evidence
  count gates completion. Use a precise closed-set VLM answer contract if a
  regex is used. Unavailable observations never provide positive evidence.
- Optional `state_on_skip` writes only declared step fields;
  `complete_on_skip: true` ends the run when explicitly skipped.

`current_view` triggers require `arguments.question` (1–500 characters), a
positive `interval_s`, and optionally `result_field: text`. Timer triggers use
`function: clock__timer` with `arguments.started_at_us` (positive Unix
microseconds) and `arguments.duration_s` (positive integer seconds). They may
select `elapsed_s`, `remaining_s`, or `expired` as `result_field`.
Arguments can reference declared readable or writable fields using
`$state.field_name`. A preceding observation can call `clock__now` and commit
the start timestamp; replay does not invent a timestamp if it is missing.
Timer-only steps use elapsed-time evidence, not a visual substitute. Completion
commits for timer triggers are rejected until their configured duration expires.

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

For replay, the launcher selects `sop_sample_replay` and uses `yaml/replay.yaml`,
`yaml/models.replay.json`, and `yaml/voice_gate.replay.yaml`. The replay worker
does not create SOP packets or accept recording commands. Add `--capture` to
save participant-wide replay media independently of guide execution.

## Validation

From the repository root, run the CPU capture and replay regressions with:

```bash
uv --config-file uv.toml run --project tests pytest -q \
  tests/test_sop_sample.py tests/test_sop_replay.py -m "not gpu"
```

The replay tests cover approval and ambiguity, pinned guide content, repeated
runs, participant isolation, stale observations, departure races, elapsed-time
guards, and actual runtime-to-voice-aggregation delivery. They use synthetic
guide data, not committed user recordings or guides.

With the configured local language-model server running, evaluate foreground
intent routing separately:

```bash
uv --config-file uv.toml run --project tests pytest -q \
  tests/test_sop_replay.py -m gpu
```

These opt-in cases call the real LLM to check direct controls, explanations,
reported and negated actions, unrelated requests, and recording requests. They
do not measure camera-model accuracy or exercise a physical assembly.
