<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP sample

The SOP sample has two mutually exclusive modes. `--capture` records narrated
demonstrations as evidence for a later procedural guide. `--replay "Guide Name"`
runs one approved local guide with spoken guidance and verification. The sample
does not generate or approve guides automatically, and cannot switch modes
while running.

## Run a demonstration

Run commands from `agent-samples/sop-sample/`:

```bash
uv run --project ../../model-server-samples/model-servers model_servers
uv run main.py --capture
```

Wait for the shared models to be ready before starting the sample. Use the
authenticated web-client URL printed by DeviceIOHub, allow microphone and
camera access, connect, and wait for readiness. Enable the microphone and live
video, then demonstrate and narrate the procedure. Pause after your last
sentence so speech recognition can finish, then turn live video off. The
worker logs the finalized packet's path and status. Starting live video again
creates a separate recording without requiring reconnection.

Refer to {doc}`/getting_started/requirements` for the model and hub deployment
requirements and {doc}`/components/server-runtime` for shared media capture.
Video recording uses the shared NVENC capture service and requires a supported
NVIDIA GPU. External STT and VLM endpoints are configured in `yaml/models.json`.
Capture mode creates a TTS client to satisfy the shared voice runtime contract,
but has no speech-output producer and makes no TTS synthesis requests.

## Recording boundaries

- Connecting with the camera off does not begin recording.
- The first live camera frame starts a participant-local recording. Unmuting
  an existing camera track also starts recording.
- Muting or unpublishing the last active camera track finalizes that recording.
  Disconnect and worker shutdown also finalize open recordings.
- A temporary pause in incoming frames does not end a recording. The boundary
  comes from the camera lifecycle, not an inactivity timeout.
- On-demand still images and screen-sharing tracks do not start a recording.
- Repeated active-track notifications do not create duplicate sessions. A new
  recording waits for the previous recording's finalization before starting.
- There are no voice controls or agent replies. All user speech during capture,
  including the words “start recording” or “stop recording,” is narration.

The microphone must be enabled to capture spoken narration. Stopping only the
microphone does not stop the camera-controlled recording. STT completion is
asynchronous: turn the camera off after the last utterance has been transcribed.
Raw audio remains available even if a transcript is missing. During rapid
off/on toggles, media arriving before the previous packet finishes finalizing
is not buffered for the next packet.
If shared capture takes longer than 10 seconds to finalize, the worker logs a
warning and continues waiting; it does not start an unbacked packet. The wait
also applies during shutdown. If no manifest arrives, restart and graceful
shutdown remain blocked until capture finalization can complete.
The worker also waits for the new bundle's recorded start event, retrying the
start command if capture is still closing the previous bundle. Frames and
captions for the new packet begin only after that acceptance check succeeds.

## Outputs

All generated data is local and ignored by Git. Defaults are relative to the
sample directory:

| Location | Contents |
|---|---|
| `artifacts/sessions/<session-id>/packet.json` | Packet status, counts, activity hierarchy, and shared media manifest reference |
| `artifacts/sessions/<session-id>/frames/` | JPEG samples at a target of 2 fps and timestamped `index.jsonl` |
| `artifacts/sessions/<session-id>/captions.jsonl` | Timestamped visual captions, activity, phase, and visible changes |
| `artifacts/sessions/<session-id>/transcript.jsonl` | User narration projected from the shared capture transcript |
| `artifacts/sessions/<session-id>/summary.md` | Human-readable activity and phase summary |
| `artifacts/sessions/<session-id>/errors.jsonl` | Capture or caption errors, when present |
| `artifacts/captures/<session-id>/<bundle>/` | Shared raw video, audio, transcript, event timeline, and final manifest |

The captioner observes the latest sampled frame every 5 seconds by default,
with the previous caption as context. Captions are
observations, not verified SOP steps. Narration is derived from the shared
capture service's transcript, not a second STT implementation. The media profile
is `raw`; this sample does not create a captioned demo MP4 automatically.

`packet.json` becomes `complete` after the shared manifest is available and
narration export succeeds. Failed media finalization or narration export marks
the packet `incomplete` and records the reason. Keep the packet and referenced
capture bundle together. Media retention is disabled by default, so monitor
disk usage and remove unwanted recordings manually.

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

In capture mode, the orchestrator starts the hub, shared `device_io_capture` service, then the
sample worker. The capture service uses `session_mode: explicit`; camera events
cause the worker to invoke the existing participant-scoped `CaptureTools`.
There is no second audio or video encoder in the sample.

The hub emits typed camera lifecycle events and replays active tracks with its
participant roster. The worker serializes boundaries per participant and keeps
an independent capture-control endpoint alive until all recordings finalize.
`VoiceAgent` owns speech recognition and publishes the shared transcript without
a conversational agent. Filesystem writes and finalization tasks are drained
before the capture endpoint closes.

The voice gate disables its optional STOP command handling, and the sample
disables early VAD transcription probes. STOP words therefore do not produce
early STOP interruptions or spoken acknowledgements. Final utterances still
use the normal speech-recognition and narration path. The defaults for other
samples remain unchanged.

`yaml/worker.yaml` owns frame frequency, caption interval, speech detection, and
the SOP artifact root. It resolves `yaml/media_capture.yaml` to locate shared
manifests. Paths are relative to their owning YAML file. Keep the shared capture
service in explicit mode. `yaml/device_io_hub.yaml` owns the room and web server;
`yaml/models.json` owns model endpoints. Restart the sample after edits. Refer
to {doc}`configuration` for checked-in fields and {doc}`command-line` for CLI
options.

Replay starts only the hub and replay worker. It reuses `VoiceAgent`,
`VoiceAggregationAgent`, typed model clients, `CurrentFrameTool`, `ImageQueryTool`,
native `ToolSet`, and `run_tool_loop`. It does not start shared capture, create
capture packets, or import the recording worker. `yaml/replay.yaml` owns replay
paths and timeouts, resolved relative to that file. Its model and voice-gate
settings are independent of capture settings. Operational logs still use the
shared launcher logging mechanism.

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
