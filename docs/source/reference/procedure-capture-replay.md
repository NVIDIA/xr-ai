<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(task-capture)=
# Procedure capture and replay

The procedure-capture-replay sample records narrated demonstrations between
spoken commands. It saves snapshots, visual captions, and transcribed narration for review and downstream
processing. Replay mode guides participants through a locally approved procedure
and verifies progress with visual observations and elapsed-time checks.

## Run a demonstration

Run commands from `agent-samples/procedure-capture-replay/`:

```bash
uv run --project model-servers procedure_capture_replay_model_servers
uv run main.py --mode capture
```

The sample's `model-servers` launcher starts batch Parakeet STT on port 8103,
Pocket TTS on 8105, Nemotron Omni LLM on 8108, and Cosmos VLM on 8100.
Batch STT accepts narration without speaker enrollment. The capture worker uses
STT, TTS, and Cosmos; the LLM is available but does not participate in capture
or generate guides automatically. Replay uses Omni for guide questions and
model-driven observations, alongside the same STT, TTS, and Cosmos services.
The launcher combines shared server defaults with
`model-servers/yaml/<gpu-profile>.yaml`; use `--gpu-profile NAME` to select a
profile explicitly. The worker's `yaml/models.json` configures its client
adapters and endpoints, not server deployment. Keep those endpoints aligned
with server ports when changing configuration. Refer to
{doc}`/guides/customizing-model-servers` for the shared launch workflow.

Wait for the shared models to be ready before starting the sample. Open the
authenticated web-client URL printed by DeviceIOHub, enable the microphone,
and connect. Say **start capture**, demonstrate and narrate, then say
**stop capture**. Say **start capture** again for another task packet without
reconnecting. The worker logs each finalized packet's path and status.

Use live video or on-demand image capture. Speak commands directly, without a
wake phrase. After **start capture**, the worker says "Capture started."
and stays silent during narration. After **stop capture**, it says
"Capture stopped." The evidence packet, transcript, and summary then finish
saving silently; processing errors are logged. Disconnecting finalizes the packet
without announcements.

To stop the sample, press Ctrl+C. The shared model servers remain running.
To stop the services selected by this sample's deployment:

```bash
uv run --project model-servers procedure_capture_replay_model_servers --stop
```

Stop the model stack before changing server settings, switching samples, or
updating the checkout. If you changed ports in a GPU profile, pass the same
`--gpu-profile NAME` on stop.

## Output destination

Choose an output root at launch without editing configuration:

```bash
uv run main.py --mode capture --output-root /path/to/task-evidence
```

Packets are saved under `PATH/sessions/<session-id>/`. Relative command-line
paths resolve against the launch directory; `~` is expanded. Without this option,
`artifacts_dir` from `yaml/capture-agent-config.yaml` is used, relative to that YAML. The override
does not relocate model or voice configuration and does not edit the YAML.

The sample saves JPEG snapshots and text evidence. Audio is transcribed by STT;
captions use the configured VLM, and recording acknowledgements use TTS. Refer to
{doc}`/getting_started/requirements` for model and hub deployment. The sample
does not launch a raw audio/video recorder or require NVENC for its snapshots.

## Recording boundaries

- After connecting, the worker waits for **start capture** to open a packet.
  **stop capture** finalizes it. Repeat the pair to record another demonstration
  on the same connection; duplicate starts during recording are ignored.
- Say each command as a standalone utterance. Matching ignores case, surrounding
  whitespace, and trailing sentence punctuation. Other speech is saved as
  narration while recording, including phrases such as “stop” or “stop recording.”
  The two control commands are excluded from narration.
- Recording and narration continue while the camera is off or images are
  unavailable. When a fresh live frame is unavailable, the worker requests a
  still from clients supporting on-demand capture; this can briefly activate
  the camera. Captioning resumes when a distinct image is saved.
- Disconnect or graceful shutdown finalizes the active packet. Reconnecting
  returns the worker to waiting for **start capture**.

Commands take effect after speech recognition finishes. Keep the microphone
enabled and pause after the final narration before saying **stop capture** or
disconnecting. A late transcript can miss the recording boundary; source audio
is not retained by this sample.

A new recording waits for the preceding packet to finish saving. Narration after
its start command is queued in order, but snapshots during this brief restart
gap are not buffered.

## Outputs

Generated data remains local and Git-ignored. Defaults are relative to the sample:

Frame, caption, and transcript logs append during recording. Packet counts and
the summary are sealed at stop after pending writes drain; they are not live
progress indicators.

| Location | Contents |
|---|---|
| `artifacts/sessions/<session-id>/packet.json` | Recording status, participant, timestamps, counts, and activity hierarchy |
| `artifacts/sessions/<session-id>/frames/` | JPEG samples targeting 2 fps and timestamped `index.jsonl` |
| `artifacts/sessions/<session-id>/captions.jsonl` | Timestamped captions, activities, phases, and visible changes |
| `artifacts/sessions/<session-id>/transcript.jsonl` | Final user transcripts inside the task recording boundary |
| `artifacts/sessions/<session-id>/summary.md` | Human-readable activity and phase summary |
| `artifacts/sessions/<session-id>/errors.jsonl` | Frame, caption, or persistence errors, when present |

### Evidence-packet contract

`packet.json` uses `schema_version: 1`. A finalized `complete` packet contains
`packet.json`, `summary.md`, and `transcript.jsonl` (which can be empty). The
`frames/` directory exists, but `frames/index.jsonl` and `captions.jsonl` are
created only when records are appended. An absent log with a zero corresponding
count means no observations of that kind were saved. A missing log with a
nonzero count is inconsistent and must not be silently interpreted as empty.
`errors.jsonl` is optional; its absence is not proof of sufficient evidence.
Cancelled startup or failed storage can leave an `incomplete` packet with fewer
files; consumers must check status and file availability before conversion.

JSONL files contain one object per nonempty line. `files` paths in the packet,
image `path` values in the frame index, and caption `frame_path` values are
relative to the packet directory. Preserve these paths when copying a packet.
Frame IDs and caption reference IDs are local to that packet. Timestamps ending
in `_us` are Unix microseconds; human-readable timestamps are UTC ISO 8601.

`started_at` is the accepted start command's transcript timestamp. During a
recording, `ended_at` is `null`. At finalization it records the effective end,
after recording producers stop and further evidence admission closes, before
pending writes and derived summaries finish draining. `stop_command.timestamp_us`
preserves the accepted stop utterance's transcript timestamp, which can precede
the last captured frames because speech recognition takes time. `stop_command`
is `null` when no spoken stop was accepted. Neither
timestamp is a claim that the demonstrated task itself finished successfully.

Activity/phase grouping, captions, and `summary.md` are fallible model-derived
interpretations. Use the frame index and saved images to inspect their evidence.
Downstream coding-agent conversion consumes the packet as immutable input and
writes generated replay material separately. A replay capability consumes that
generated package rather than importing capture internals.

The captioner starts when the first sampled frame is available, then waits
5 seconds after each caption attempt before checking for a new frame. Each
request compares two saved images in chronological order: the previous
successfully captioned frame, then the current frame. The first caption of each
recording uses only the initial image. Requests describe visible actions, object
details, and changes, and run when a distinct saved image is available. Model
latency adds to the caption interval; on-demand sampling also depends on client
response time. Review captions against the images: they are model observations,
not verified procedural steps.

`caption` describes the current image; `delta` compares earlier and current
states. Previous activity and phase names are supplied only to stabilize summary
grouping, not as visual evidence. Grouping is a navigation aid, not a step list.
Malformed responses are recorded as `caption_error` in `errors.jsonl`, including
the raw response, and are excluded from captions and summaries. The next request
still compares against the last successfully captioned image.

Each saved caption includes pipeline-generated `delta_reference_frame_id` and
`delta_reference_timestamp_us` for the earlier image actually supplied to the
model. Both are `null` on the first observation. Resolve the reference ID through
`frames/index.jsonl`; it need not be the preceding sampled frame, especially
after slow or failed inference. Current-image provenance remains in `frame_id`,
`frame_timestamp_us`, and `frame_path`. These references identify the evidence
pair; they do not verify the model's interpretation or establish intervening
actions. Older packets without the reference fields do not explicitly record
that comparison provenance.

The transcript contains admitted narration between the spoken commands. The
commands themselves are recorded as boundaries rather than narration.

`packet.json.stop_command` records an accepted spoken stop. Packet status
`complete` means local writes finished; evidence can still be missing or
incorrect. Review the counts, errors, and source images before using the packet.
Persistence failures mark a packet `incomplete` when storage permits saving that
status; otherwise the process log reports the failure. Failed packets are closed
so subsequent recording commands create separate packets.
Recordings are retained until manually removed, so monitor disk usage.

## Caption evaluation

With the configured VLM running, run the synthetic visual controls from the
sample directory. Choose a new output directory for each evaluation:

```bash
PYTHONPATH=worker uv run --project worker python eval/caption.py --output artifacts/caption-eval
```

The cases exercise initial observations, chronological direction, unchanged
images, object appearance and disappearance, and grouping labels that disagree
with visual evidence. The runner uses the production query, system prompt,
multi-image tool, and parser. It saves source images, raw responses, and expected
observations for inspection. The command exits nonzero if any case fails the
response contract or cannot obtain a model response; all cases are still saved
for inspection. Its automatic result measures response structure,
not visual accuracy: compare every output with its images and expected
observations. Synthetic controls supplement, rather than replace, evaluations on
recorded demonstrations.

## Replay an approved guide

Place a reviewed `*.guide.yaml` or `*.guide.yml` file under `guides/` in the
sample directory. User-created guides remain local and ignored by Git; the
maintained tea-making example is the only checked-in exception. Set `task.status` to
`approved` only after reviewing the instructions, completion checks, timers,
and safety constraints. The CLI matches an exact, case-insensitive `task.name`
or `task.id`; it rejects missing, draft, invalid, and ambiguous selections.
Quote names containing spaces:

```bash
uv run main.py --mode replay "Guide Name"
```

Start the shared models first, as for capture. Replay uses the LLM, VLM, STT,
and TTS endpoints in `yaml/models.json`. Connect using the hub's web
client URL and enable microphone and camera access. The command line selects
the guide, but connecting does not start it. Say **start replay** to begin step
one with independent progress for that participant. No wake phrase is required.
Until then, no verification or timer is running. Say **stop replay** at any time
to cancel guidance, then **start replay** to begin a fresh run of the same guide.
Turning off live video does not reset the guide. The shared current
frame tool may request a still image from clients that support on-demand capture;
if no fresh image is available, visual verification waits. Timer checks continue
without a camera. Disconnect releases
that participant's progress and pending work; reconnect waits for start replay.

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
| “Start replay” | Begin a fresh run of the command-line-selected guide from step one |
| “Stop replay” | Cancel guidance and verification, clear progress, and keep the selected guide loaded but inactive |
| “What should I do?” | Explain the current step |
| “Does this look right?” | Use the current step's visual tool if available |
| “How much time remains?” | Use the current step's timer tool if available |
| “Next” or “continue” | Advance only if the current step is verified complete |
| “Skip” | Explicitly bypass the current step; this is not evidence of success |
| “Status” | Report current progress |
| “Stop guide” or “reset guide” | Same as stop replay |
| “Restart” | Same as start replay |

After “next” on the completed final step, the agent announces workflow
completion and leaves the guide inactive, ready for another start replay.
No relaunch or reconnect is needed. Explicitly skipping to the end also leaves
the guide inactive, with an explicit skipped-steps message rather than a
verified-completion claim. While inactive, status and start replay remain
available, but there is no active step or timer. Recording commands and
selecting another guide are not supported inside replay mode.

The worker pins the validated guide and SHA-256 from the same captured bytes at
startup. File edits do not change a running replay; restart the sample to use a
new version or revoke approval. Approval is a local review convention, not a
cryptographic signature. Model-based visual checks can be wrong; reviewers must
describe observable evidence, including orientation and count where relevant.
This demo is not a safety interlock. A still image cannot prove a hidden action.

### Included tea-making guide

`guides/tea-making.guide.yaml` is a hand-authored, approved example with task
name **Tea Making** and ID `tea_making`. It follows the five stages of
{doc}`tea-making-sample`: identify tea, fill the heating vessel, heat water,
immerse the tea, and wait for steeping. It is not generated from a user recording.

Start the sample's shared models, then run:

```bash
uv run main.py --mode replay "Tea Making"
```

Connect and say **start replay**. Say **next** after each verified step and
**stop replay** to cancel. After completion, say **start replay** to repeat.

Use green tea with compatible package instructions, a stable heat-safe vessel,
and a kettle or thermometer with a readable **current** water-temperature
display. The example uses 80°C (176°F) water and a three-minute steep, within
the green-tea ranges in the tea sample's brewing reference. Its visual target
check accepts 78–82°C or 172–180°F. If the package requires a different recipe,
stop and adapt a local copy before using it. Do not tilt or lift hot vessels
to satisfy a camera check.

The example is adapted to replay's existing tools, not a drop-in copy of the
tea sample's workflow YAML:

- It uses a fixed green-tea recipe, not package-dependent settings or RAG lookup.
- Its four visual steps require two distinct fresh frames whose VLM responses
  exactly match `YES`. `NO`, `UNKNOWN`, missing frames, and malformed responses
  cannot complete a step. These bounded evidence commits do not use an
  observation LLM; the foreground LLM still handles questions and controls.
- The heating check verifies a displayed current temperature in the recipe
  range, rather than the tea sample's “heating underway” threshold. Replay
  does not expose the tea sample's typed temperature tools: this is a VLM
  judgment, not a deterministic temperature measurement or safety guarantee.
- Immersion detection does not start the timer. Say **next** promptly after
  immersion is confirmed to enter the timer step. The runtime then measures a
  full 180 seconds; time spent waiting at the immersion step is not subtracted.
  Timer verification does not inspect images or prove continued immersion.
- Skipping a visual prerequisite ends the run as skipped instead of assuming
  the prerequisite was satisfied. Skipping the timer also cannot produce a
  verified-completion announcement.

Regression tests load this exact example and exercise its state transitions,
distinct-frame evidence, skip behavior, and timer with simulated observations.
Those tests do not establish accuracy on real tea-making images or appliances.

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
  regex is used. Visual evidence counts distinct source frames, identified by
  participant, track, timestamp, and sequence. Repeated or older frames do not
  increment the count or run another observation-model update. Unavailable
  observations reset the consecutive evidence count.
- Optional `state_on_skip` writes only declared step fields;
  `complete_on_skip: true` ends the run when explicitly skipped.

`current_view` triggers require `arguments.question` (1–500 characters), a
positive `interval_s`, and optionally `result_field: text`. Timer triggers use
`function: clock__timer` with `arguments.duration_s` (positive integer seconds). They may
select `elapsed_s`, `remaining_s`, or `expired` as `result_field`.
Arguments can reference declared readable or writable fields using
`$state.field_name`. On entering a timer step, replay resolves and freezes its
duration and starts a monotonic runtime clock. Missing or invalid durations
leave the step blocked. Start timestamps are not writable timer state;
`started_at_us` in timer trigger arguments is rejected. Model commits and
wall-clock changes cannot backdate the timer or shorten an active wait.
The foreground `clock__timer` tool takes no arguments and reads the active
runtime timer. Restarting the guide starts a fresh timer on the next timer-step
entry; `clock__now` is informational and cannot start or modify timers.
Timer-only steps use elapsed-time evidence, not a visual substitute. Completion
commits for timer triggers are rejected until their configured duration expires.

## Composition and configuration

The outer `main.py` launches the hub and worker. The worker's `__main__.py`
accepts configuration and readiness arguments; `procedure_capture_replay_worker/app.py` constructs
the hub endpoint, model clients, voice runtime, and frame/image tools.
The worker uses `VoiceAgent` for transcripts, `CurrentFrameTool` for snapshots,
and `MultiImageQueryTool` for captions.

The registered `CaptureAgent` in `procedure_capture_replay_worker/capture/agent.py` owns commands,
narration admission, participant generations, and acknowledgements. Its private
`_Capture` helper in `capture/capturing.py` owns sampling, caption tasks,
persistence, and finalization.
`capture/captioning.py` defines the request and response contract used by both
recording and evaluation. Application cleanup explicitly drains capture work
before closing its frame endpoint and VLM client.

The worker keeps shared model, voice, image, and lifecycle wiring in `app.py`
and shared settings in `config.py`. The sibling `capture/` and `replay/` packages
own their capability-specific agents, settings, and helpers. Only the selected
agent is registered. With no mode flag, capture remains the default.

`replay/agent.py` handles connection events, foreground questions, and conversational
controls. Its `_Replay` backend in `replay/replaying.py` owns guide progress,
runtime timers, frame evidence, and background verification. Transitions clear
both local notices and pending voice-aggregation output under the session lock,
so obsolete completion notices cannot follow the new step's instruction.
The foreground receives detached guidance context and scoped tools captured
under one execution revision. The backend rechecks that revision under its own
lock when controls execute and answers finish. It owns accepted control tasks
through completion even if a new utterance cancels the foreground turn; departure
and shutdown drain those tasks. Stop replay cancels the observer and clears timer
state without unloading the selected guide. Connecting only loads an inactive
session; start replay activates it and starts its observer and any first-step timer.
Replay uses `VoiceAggregationAgent`, `ImageQueryTool`, native tools, and the
shared tool loop. The conversation prompt governs intent; an English wording
filter is only a conservative routing heuristic, not an authorization boundary.
State-changing tools independently check session freshness and completion.
Guide loading and validation live in `replay/guides.py`
and `replay/_workflow_spec.py`. Replay never constructs the capture backend or
writes demonstration packets. Both modes start the same hub and worker processes
and use the same separately launched model stack and `yaml/models.json`.

The always-on voice gate passes commands and narration through the same accepted
input path. Recording acknowledgements follow voice-input handling, so a command
cannot cancel its own response.

| Configuration | Settings |
|---|---|
| `yaml/worker.yaml` | Shared frame and speech detection settings; capability configuration paths |
| `yaml/capture-agent-config.yaml` | Task output directory, sampling rate, and caption interval |
| `yaml/replay-agent-config.yaml` | Local guide directory and visual verification timeout |
| `yaml/models.json` | Shared batch STT, LLM, VLM, and TTS services and endpoints |
| `yaml/voice_gate.yaml` | Silent, wake-word-free input with sample-owned recording commands |
| `yaml/voice_gate.replay.yaml` | Wake-word-free replay questions and guide controls |
| `yaml/device_io_hub.yaml` | Room and web server |

`worker.yaml` selects the capture YAML through `capture_config`, resolved relative
to the worker YAML. `artifacts_dir` resolves relative to the selected capture
YAML, even when it is in another directory. Model and voice configuration remain
beside the worker YAML. Absolute configuration and output paths are also supported.
The worker loads shared settings into `WorkerConfig` and capture-specific settings
into `CaptureConfig` under `capture/config.py`; `app.py` supplies each to its
owning component. Replay selects its YAML through `replay_config` and loads
`ReplayConfig` from `replay/config.py`; its guide directory resolves relative to
that YAML. The required positional `GUIDE_NAME` selects `task.name` or `task.id`,
not a filesystem path. `--output-root` applies only to capture. Restart after
configuration edits.
Refer to {doc}`configuration` and {doc}`command-line` for fields and CLI options.

## Replay regression tests

From the repository root:

```bash
uv --config-file uv.toml run --project tests pytest -q \
  tests/test_procedure_capture_replay.py tests/test_sop_replay.py -m "not gpu"
```

These tests cover mode isolation, approval, guide validation, participant-local
progress, repeated runs, stale observations, timers, and runtime-to-voice delivery.
They use synthetic guide fixtures, not committed user guides. Opt-in tests in
`test_sop_replay.py` with `-m gpu` exercise the configured local LLM's tool routing;
they do not measure visual accuracy or verify a physical assembly. Start the
sample's model servers before running those evaluations. They are also marked
`external_models`: they query the existing deployment without starting, stopping,
or deleting its containers.
