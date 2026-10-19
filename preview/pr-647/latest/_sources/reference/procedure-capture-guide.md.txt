<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(task-capture)=
# Procedure capture and guidance

The procedure-capture-guide sample records narrated demonstrations between
spoken commands. It saves snapshots, visual captions, and transcribed narration for review and downstream
processing. Guide mode guides participants through a locally approved procedure
and verifies progress with visual observations and elapsed-time checks.

## Run a demonstration

Run commands from `agent-samples/procedure-capture-guide/`:

```bash
uv run --project model-servers procedure_capture_guide_model_servers
uv run main.py --mode capture
```

The sample's `model-servers` launcher starts batch Parakeet STT on port 8103,
Pocket TTS on 8105, Nemotron Omni LLM on 8108, and Cosmos VLM on 8100.
Batch STT accepts narration without speaker enrollment. The capture worker uses
STT, TTS, and Cosmos; the LLM is available but does not participate in capture
or generate guides automatically. Guide mode uses Omni for guide questions and
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
uv run --project model-servers procedure_capture_guide_model_servers --stop
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
writes generated guide material separately. A guide capability consumes that
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

## Follow an approved guide

Place a reviewed `*.guide.yaml` or `*.guide.yml` file under `guides/` in the
sample directory. User-created guides remain local and ignored by Git; the
maintained tea-making example is the only checked-in exception. Set `task.status` to
`approved` only after reviewing the instructions, completion checks, timers,
and safety constraints. The CLI matches an exact, case-insensitive `task.name`
or `task.id`; it rejects missing, draft, invalid, and ambiguous selections.
Quote names containing spaces:

```bash
uv run main.py --mode guide "Guide Name"
```

Start the shared models first, as for capture. Guide mode uses the LLM, VLM, STT,
and TTS endpoints in `yaml/models.json`. Connect using the hub's web
client URL and enable microphone and camera access. The command line selects
the guide, but connecting does not start it. Say **start guide** to begin step
one with independent progress for that participant. No wake phrase is required.
Until then, no verification or timer is running. Say **stop guide** at any time
to cancel guidance, then **start guide** to begin a fresh run of the same guide.
Turning off live video does not reset the guide. The shared current
frame tool may request a still image from clients that support on-demand capture;
if no fresh image is available, visual verification waits. Timer checks continue
without a camera. Disconnect releases
that participant's progress and pending work; reconnect waits for start guide.

Like {doc}`tea-making-sample`, guide separates foreground conversation from
background verification. The foreground answers guide questions using the
current step and its allowed tools. It cannot directly commit completion.
Periodic observations use a fresh camera view or a deterministic elapsed-time
check. The observer can commit only declared writable fields, with any required
consecutive evidence. Late observations from a previous step or run cannot
change current progress. The agent announces step completion once, without
requiring a user question, but does not automatically advance to the next step.

| Request | Behavior |
|---|---|
| “Start guide” | Begin a fresh run of the command-line-selected guide from step one |
| “Stop guide” | Cancel guidance and verification, clear progress, and keep the selected guide loaded but inactive |
| “What should I do?” | Explain the current step |
| “Does this look right?” | Use the current step's visual tool if available |
| “How much time remains?” | Use the current step's timer tool if available |
| “Next” or “continue” | Advance only if the current step is verified complete |
| “Skip” | Explicitly bypass the current step; this is not evidence of success |
| “Status” | Report current progress |
| “Reset guide” | Same as stop guide |
| “Restart” | Same as start guide |

After “next” on the completed final step, the agent announces workflow
completion and leaves the guide inactive, ready for another start guide.
No relaunch or reconnect is needed. Explicitly skipping to the end also leaves
the guide inactive, with an explicit skipped-steps message rather than a
verified-completion claim. While inactive, status and start guide remain
available, but there is no active step or timer. Recording commands and
selecting another guide are not supported inside guide mode.

The worker pins the validated guide and SHA-256 from the same captured bytes at
startup. File edits do not change a running guide; restart the sample to use a
new version or revoke approval. Approval is a local review convention, not a
cryptographic signature. Model-based visual checks can be wrong; reviewers must
describe observable evidence, including orientation and count where relevant.
This demo is not a safety interlock. A still image cannot prove a hidden action.

### Included tea-making guide

`guides/tea-making.guide.yaml` is a hand-authored, approved example with task
name **Tea Making** and ID `tea_making`. Its four stages guide brewing
chamomile tea: read the packet, verify the water temperature, put the tea in
water, and start the steeping timer. It is not generated from a user recording.

Start the sample's shared models, then run:

```bash
uv run main.py --mode guide "Tea Making"
```

Connect and say **start guide**. Say **next** after each verified visual step
and **stop guide** to cancel. The guide finishes automatically when the
five-minute timer expires. Say **start guide** to repeat.

Use a tea packet or box labeled **chamomile**, a stable heat-safe vessel,
and a kettle or thermometer with a readable **current** water-temperature
display. The example uses the chamomile settings in the tea sample's
[brewing reference](https://github.com/NVIDIA/xr-ai/blob/main/agent-samples/tea-making-sample/rag-documents/tea-brewing.md):
100°C (212°F) water and a 5–7 minute steep. It selects five minutes and verifies
a displayed 100°C (212°F) current reading. If the packet requires a different
recipe or local boiling temperature cannot reach that target, stop and adapt a
local copy before using it. Do not tilt or lift hot vessels to satisfy a camera
check.

The example is adapted to guide's existing tools, not a drop-in copy of the
tea sample's workflow YAML:

- It accepts the printed word **chamomile** on a tea packet or box, including
  blend names such as **Chamomile Lemon**, regardless of brand. The brand does
  not need to be visible. The brewing settings are fixed in the guide; the
  brewing reference is not retrieved at runtime.
- Its three visual steps require two distinct fresh frames. The first reads
  the tea name and matches the word `CHAMOMILE`; the second reads a current
  `LABEL | NUMBER UNIT` reading and accepts `CURRENT`, `ACTUAL`, or `THERMOMETER`
  with `100 C` or `212 F`. It preserves `TARGET` and `SET` labels in the visual
  result so those readings cannot satisfy completion. Unlabeled displays are
  unavailable rather than assumed to show current temperature. Matching tolerates
  equivalent letter case, spacing, degree symbols, and trailing decimal zeros,
  not different temperatures. Immersion requires `YES` for
  visible bag contact with visible water, including a floating bag.
  Other readings, `NO`, `UNKNOWN`, missing frames, and malformed responses
  cannot complete a step. These bounded evidence commits do not use an
  observation LLM; the foreground LLM still handles questions and controls.
- The heating check verifies a displayed current temperature at the brewing
  target, not a target setting or a “heating underway” threshold. Guide
  does not expose the tea sample's typed temperature tools: this is a VLM
  judgment, not a deterministic temperature measurement or safety guarantee.
- Immersion detection does not start the timer. Say **next** promptly after
  immersion is confirmed to enter the timer step. The runtime then measures a
  full 300 seconds; time spent waiting at the immersion step is not subtracted.
  Timer verification does not inspect images or prove continued immersion.
- The final step sets `finish_on_complete: true`, so expiry ends the active run,
  clears progress, and announces completion without another **next** command.
- Skipping a visual prerequisite ends the run as skipped instead of assuming
  the prerequisite was satisfied. Skipping the timer also cannot produce a
  verified-completion announcement.

Regression tests load this exact example and exercise its state transitions,
distinct-frame evidence, skip behavior, and timer with simulated observations.
Those tests do not establish accuracy on real tea-making images or appliances.

#### Local-model guide evaluation

`eval/guide.py` exercises a manifest's selected guide through
the actual guide agent, runtime, voice aggregation, frame selection, and model
tools. It uses the sample's configured LLM and VLM. Fixture images replace the
hub camera endpoint, and typed requests enter the same runtime topic as
transcribed speech. Microphones, STT, network camera delivery, and audible TTS
are not covered. The script never manages external model-server processes.

The checked-in `eval/guide_cases.yaml` and `eval/fixtures/indicator.guide.yaml`
provide a synthetic visual-and-timer scenario. `eval/guide_routing.yaml`
exercises the composed foreground prompt with controls, polite requests,
quoted commands, negation, requests to explain guide content, unsupported capture
requests, and visual questions.
Both run with the full permitted tool set; the evaluator does not preselect
tools based on the utterance.

A YAML or JSON manifest selects `guides_dir` (relative to the manifest) and
`guide` (name or ID), and lists ordered `actions`:

- `image`: show an image `path`, or a named synthetic `scene`. Optional
  `scenes` map names to a rectangle color or display text.
- `observe`: wait for `count` visual results (default two) from the displayed
  image and compare them to the allowed `answers`.
- `say`: send `text` as a user turn. Optional `expect` checks the delivered
  answer against a regex; `tools` and `arguments` check model tool calls.
  Status assertions use **What is my status?** and `workflow__status`.
- `wait_speech`: wait for an `expect` regex in speech delivered since the last
  user turn, with optional `timeout_s`. This checks completion notices, including
  automatic timer completion, without reading private backend state.

The runner makes no assumptions about step names, step count, or a final timer.
Timers use real elapsed time. Include negative observations before positive
ones and explicitly test early advance, completion, stop, and restart as needed.

From the sample directory, with its model servers already running:

```bash
uv run --project worker python eval/guide.py \
  --output artifacts/guide-eval
uv run --project worker python eval/guide.py \
  --manifest eval/guide_routing.yaml --output artifacts/guide-routing-eval
```

Use a new output directory for each run. Supply `--manifest /path/to/scenario.yaml`
to evaluate another guide and its images.
Outputs include the exact guide, manifest, image hashes, model tool calls, visual
answers, speech text, and result JSON. Synthetic controls can isolate label or
setpoint mistakes but do not establish accuracy on a user's actual camera views.

### Guide structure

The declarative shape follows the tea-making workflow: a task, typed sparse
state, and ordered steps with separate observation and voice policies. This is
a guide-specific schema inspired by that design, not a compatible loader for
tea-making or legacy workflow-recorder YAML. Adapt guides to the contract below.

Text fields must be actual strings. Duplicate mapping keys are rejected at
every level, including overrides introduced through YAML merges, instead of
silently replacing earlier requirements. Invalid guides cannot be used in guide mode,
even when their `task.status` is `approved`.

- `schema_version` must be `1`.
- `task` declares `id`, `name`, positive `version`, `status` (`draft` or
  `approved`), `start_step`, `foreground_prompt`, and
  `complete_message`.
- Each `state` field declares a type (`boolean`, `integer`, `number`, or
  `string`), a description, and optionally an initial value.
- Each step declares `id`, `title`, `reads`, `writes`, `trigger`,
  `voice`, a non-empty `complete_when` mapping, `next`, and `messages` containing
  `enter`, `complete`, and `skip`. The `next` chain must visit every step once
  and end with `null`.
- `agent` and `voice` each contain a prompt and a tool list. `agent` is optional
  when `evidence.commit` supplies a deterministic completion patch; that path
  does not run an observation LLM. Otherwise `agent` is required. Allowed step tools
  are `current_view`, `clock__now`, and `clock__timer`. Only the observer receives
  `workflow__commit`. Foreground workflow controls are supplied by the runner,
  not listed in the guide's step tools.
- Optional `evidence` contains a full-match regex `pattern`, positive
  `consecutive` count, and optional `commit` mapping. With `commit`, matching
  observations directly apply that bounded patch without an observation LLM.
  The patch must satisfy every `complete_when` requirement. Each step must have
  at least one initially unmet completion field that no preceding step can write;
  use a dedicated completion flag rather than reusing a preceding step's flag.
  Without `commit`, the observation LLM proposes state changes and the evidence
  count gates completion. Use a precise closed-set VLM answer contract if a
  regex is used. Visual evidence counts distinct source frames, identified by
  participant, track, timestamp, and sequence. Repeated or older frames do not
  increment the count or run another observation-model update. Unavailable
  observations reset the consecutive evidence count.
- Optional `state_on_skip` writes only declared step fields;
  `complete_on_skip: true` ends the run when explicitly skipped.
- Optional `finish_on_complete: true` is valid only on a terminal step. Once
  verified, it ends the run automatically and announces the task's completion
  message. If an earlier step was skipped, it reports skipped steps instead.
  The default is `false`, preserving explicit advancement after verification.

`current_view` triggers require `arguments.question` (1–500 characters), a
positive `interval_s`, and optionally `result_field: text`. Timer triggers use
`function: clock__timer` with `arguments.duration_s` (positive integer seconds). They may
select `elapsed_s`, `remaining_s`, or `expired` as `result_field`.
Arguments can reference declared readable or writable fields using
`$state.field_name`. On entering a timer step, guide resolves and freezes its
duration and starts a monotonic runtime clock. Missing or invalid durations
leave the step blocked. Start timestamps are not writable timer state;
`started_at_us` in timer trigger arguments is rejected. Model commits and
wall-clock changes cannot backdate the timer or shorten an active wait.
The foreground `clock__timer` tool takes no arguments and reads the active
runtime timer. Restarting the guide starts a fresh timer on the next timer-step
entry; `clock__now` is informational and cannot start or modify timers.
An unavailable or superseded timer returns `available: false` and a reason,
not an invented elapsed time or a tool exception.
Timer-only steps use elapsed-time evidence, not a visual substitute. Completion
commits for timer triggers are rejected until their configured duration expires.

## Composition and configuration

The outer `main.py` launches the hub and worker. The worker's `__main__.py`
accepts configuration and readiness arguments; `procedure_capture_guide_worker/app.py` constructs
the hub endpoint, model clients, voice runtime, and frame/image tools.
The worker uses `VoiceAgent` for transcripts, `CurrentFrameTool` for snapshots,
and `MultiImageQueryTool` for captions.

The registered `CaptureAgent` in `procedure_capture_guide_worker/capture/agent.py` owns commands,
narration admission, participant generations, and acknowledgements. Its private
`_Capture` helper in `capture/capturing.py` owns sampling, caption tasks,
persistence, and finalization.
`capture/captioning.py` defines the request and response contract used by both
recording and evaluation. Application cleanup explicitly drains capture work
before closing its frame endpoint and VLM client.

The worker keeps shared model, voice, image, and lifecycle wiring in `app.py`
and shared settings in `config.py`. The sibling `capture/` and `guide/` packages
own their capability-specific agents, settings, and helpers. Only the selected
agent is registered. With no mode flag, capture remains the default.

`guide/agent.py` handles connection events, foreground questions, and conversational
controls. Its `_Guide` backend in `guide/guiding.py` owns guide progress,
runtime timers, frame evidence, and background verification. Transitions clear
both local notices and pending voice-aggregation output under the session lock,
so obsolete completion notices cannot follow the new step's instruction.
The foreground receives detached guidance context and scoped tools. Observation
updates have a state revision; controls have a separate run-and-step revision.
Completing the same step during inference does not invalidate a stop or advance
request, but a concurrent step transition does. The backend rechecks those scopes
under its own lock when controls execute and answers finish. It owns accepted control tasks
through completion even if a new utterance cancels the foreground turn. The
cancelled turn drains accepted controls and delivers their still-current transition
response; new guidance snapshots also drain controls first. Departure and shutdown
drain those tasks without speaking to disconnected participants.
Stop guide cancels the observer and clears timer
state without unloading the selected guide. Connecting only loads an inactive
session; start guide activates it and starts its observer and any first-step timer.
Guide mode uses `VoiceAggregationAgent`, `ImageQueryTool`, native tools, and the
shared tool loop. The conversation prompt governs intent without an English
wording filter. Polite action requests can use controls; questions about controls
do not authorize them. This is model policy, not a deterministic intent guarantee.
State-changing tools independently check session freshness and completion.
Guide loading and validation live in `guide/guides.py`
and `guide/_workflow_spec.py`. Guide mode never constructs the capture backend or
writes demonstration packets. Both modes start the same hub and worker processes
and use the same separately launched model stack and `yaml/models.json`.

The always-on voice gate passes commands and narration through the same accepted
input path. Recording acknowledgements follow voice-input handling, so a command
cannot cancel its own response.

| Configuration | Settings |
|---|---|
| `yaml/worker.yaml` | Shared frame and speech detection settings; capability configuration paths |
| `yaml/capture-agent-config.yaml` | Task output directory, sampling rate, and caption interval |
| `yaml/guide-agent-config.yaml` | Local guide directory and visual verification timeout |
| `yaml/models.json` | Shared batch STT, LLM, VLM, and TTS services and endpoints |
| `yaml/voice_gate.yaml` | Silent, wake-word-free input with sample-owned recording commands |
| `yaml/voice_gate.guide.yaml` | Wake-word-free guide questions and guide controls |
| `yaml/device_io_hub.yaml` | Room and web server |

`worker.yaml` selects the capture YAML through `capture_config`, resolved relative
to the worker YAML. `artifacts_dir` resolves relative to the selected capture
YAML, even when it is in another directory. Model and voice configuration remain
beside the worker YAML. Absolute configuration and output paths are also supported.
The worker loads shared settings into `WorkerConfig` and capture-specific settings
into `CaptureConfig` under `capture/config.py`; `app.py` supplies each to its
owning component. Guide mode selects its YAML through `guide_config` and loads
`GuideConfig` from `guide/config.py`; its guide directory resolves relative to
that YAML. The required positional `GUIDE_NAME` selects `task.name` or `task.id`,
not a filesystem path. `--output-root` applies only to capture. Restart after
configuration edits.
Refer to {doc}`configuration` and {doc}`command-line` for fields and CLI options.

## Guide regression tests

From the repository root:

```bash
uv --config-file uv.toml run --project tests pytest -q \
  tests/test_procedure_capture_guide.py tests/test_sop_guide.py -m "not gpu"
```

These tests cover mode isolation, approval, guide validation, participant-local
progress, repeated runs, stale observations, timers, and runtime-to-voice delivery.
They use synthetic fixtures and validate the shipped hand-authored tea guide,
not private user recordings. Offline CLI tests exercise the evaluation runner
with controlled model responses, including visual-only and visual-and-timer guides.
These tests establish runtime behavior, not model accuracy. Run the standalone
`eval/guide.py` scenarios above against existing model servers to measure live
tool routing and visual judgments; the runner never manages their lifecycle.
