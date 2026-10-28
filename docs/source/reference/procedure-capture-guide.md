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
or generate guides automatically. Guide mode uses Omni for conversation, Cosmos for descriptive visual evidence,
and CLEF (`Cloudflare/clef-flash`, port 8120) to decide whether that evidence
establishes the step condition. The model-server launcher includes CLEF; guide
readiness waits for it, while capture does not construct or wait for CLEF.
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

On connection, capture mode says: "Say start capture to start the recording and
demonstrate your procedure. To finish recording, say stop capture." This welcome
is only sent on connection. Other speech while idle does not trigger a reply,
and the welcome is not repeated after "Capture stopped."

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
does not require NVENC for its snapshots. Session recording is a separate opt-in
feature described below.

## Session recording

To record the whole participant connection in either mode, run:

```bash
uv run main.py --mode capture --session-recording
uv run main.py --mode guide example-guides/tea-making/guide.yaml --session-recording
```

The shared recorder starts immediately after DeviceIOHub and records participant
video, device and agent audio, and data-channel traffic from join to leave.
Recording continues while the selected agent is idle, capturing a demonstration,
or following a guide. **start capture** and **stop capture** control only procedure
packets; **start guide** and **stop guide** control only guided execution.
Neither pair starts or stops session recording, and enabling session recording
does not start either procedure mode's work. Guide mode still waits for **start guide**
and retains its verification and timer rules. Disconnecting closes the raw bundle
and then starts rendering while the recorder remains running.

`yaml/media_capture.yaml` selects `session_mode: participant` and the `demo`
profile. It saves recordings under
`~/.local/share/xr-ai/captures/procedure-capture-guide/`, retaining raw media and
rendering a captioned MP4 after disconnect. This option requires an NVIDIA GPU with NVENC support
and `ffmpeg` on `PATH`. Refer to {doc}`../components/server-runtime` for the shared
recorder's output and encoding contract.

For a captioned video, disconnect the client first and wait for the bundle's
`video/session.mp4` and its `renderings.captioned_mp4` manifest entry before
pressing Ctrl+C. Stack shutdown commits active raw bundles without starting new
renders. A render already running can also exceed the launcher's 20-second
shutdown timeout and be interrupted. For a completed raw-only bundle, use
[device_io_capture_render](../components/server-runtime.md#media-hub-session-capture)
to render it separately after the stack stops.

Session recordings have a separate retention policy: the shared recorder
automatically prunes the oldest completed bundles under `out_dir` to meet its
default **10 GiB** limit. Set `max_total_bytes` in `yaml/media_capture.yaml` to
change that limit; `0` disables automatic pruning. Copy recordings you need to
keep outside that directory, and monitor disk usage if pruning is disabled.
This policy does not prune procedure evidence packets.

The flag does not enable the client camera or microphone. Enable live video for
continuous camera recording. In capture mode, `--output-root` can be combined with this flag and
changes only the procedure-packet destination; configure session recording output
through `out_dir` in `yaml/media_capture.yaml`.

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
is retained only when session recording is enabled.

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
Procedure packets are retained until manually removed, so monitor disk usage.

## Caption evaluation

With the configured VLM running, run the synthetic visual controls from the
sample directory. Choose a new output directory for each evaluation:

```bash
PYTHONPATH=worker uv run --project worker python evaluation/capture/evaluate_captions.py --output artifacts/caption-eval
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

## Generate a draft guide

Use a coding agent with local file and image access to follow
`guide-generation-skills/SKILL.md`. This orchestration skill coordinates the
initial draft, runtime evaluation, evidence-supported revisions, and human-review
handoff. Its `draft-guide/` child writes YAML directly; `evaluate-guide/` runs a
fixed draft against captured images with the configured XR-AI models and tools,
then reports findings without editing the guide. The orchestrator owns revisions.
These skills, their references, the validator, and packet-playback support belong
to the generation workflow. No separate sample regression suite is needed.

The authoring model observes runtime results and edits the
guide. It does not replace the runtime VLM, invent tool answers, compile a plan
into YAML, or approve the result. The capture worker does not invoke the skills.

From the repository root, give the agent this request:

```text
Read agent-samples/procedure-capture-guide/guide-generation-skills/SKILL.md
and follow its draft-guide and evaluate-guide child skills, shared schema,
and applicable examples.
Use agent-samples/procedure-capture-guide/artifacts/sessions/<session-id>/packet.json.
Create <slug>-guide/ under agent-samples/procedure-capture-guide/ and write
guide.yaml and review.md inside it. Use a lowercase, hyphen-separated slug
derived from the guide name, such as assemble-toy-guide/guide.yaml.
Keep evaluation.md and per-run evaluation artifacts in the same guide directory.
Keep the guide draft.
Reuse narration and captions; inspect original images where evidence is unclear.
Interpret all capture sources together as complementary, fallible evidence of
the intended task, not observations to translate literally into steps. Use
judgment for ordinary organization and wording. Document consequential assumptions,
their effects and questions needing human confirmation in review.md, while
continuing to draft sufficiently understood parts.
Preserve distinct actions, counts, orientation, order, and required durations.
Validate with the runtime parser, then follow
agent-samples/procedure-capture-guide/guide-generation-skills/evaluate-guide/SKILL.md to test the
draft with the captured images and configured local XR-AI models and tools.
Assess fidelity to the intended task and clarity for someone performing it
afresh, not just whether the recording passes. Distinguish recording coverage
gaps, uncertain requirements and actual runtime verification limitations.
After a failed run, return to revision within this request. Compare selected
source images, Cosmos descriptions and CLEF decisions; fix supported question
or condition defects and rerun. Inspect the first completion's frames for false
positives as well as missed completions. Do not stop at reporting a failed run
or let one unsupported check prevent refinement of independently testable steps.
Preserve unsupported requirements as blockers, and justify any stop without
revision using the source evidence and actual outputs.
Revalidate and rerun revised guides without weakening requirements to fit the
demonstration. Follow the skill's iteration and final-verification rules.
Report paths, actual results, revision rounds, coverage gaps and approval
blockers. Leave task.status draft. Do not edit source recordings, substitute
your own visual answers, or invent successful tests when models are unavailable.
```

No installation is required for this explicit-file request. To install the
workflow, copy the entire `guide-generation-skills/` directory, preserving its
top-level `SKILL.md`, child skills, references, and metadata. Start with
the top-level skill, which links to the child instructions; do not depend on
nested skills being independently discovered. Keep the target checkout and
packet paths explicit; the evaluator runs that checkout's implementation.

The outputs are local and Git-ignored:

- `<slug>-guide/guide.yaml`: model-authored draft.
- `<slug>-guide/review.md`: action-to-evidence mapping, capture ID, source
  frame references, corrections, revisions, consequential assumptions and
  questions needing human confirmation.
- `<slug>-guide/evaluation.md`: task fidelity and clarity assessment, runtime
  observations, evaluated hash, untested checks and findings for the next revision.

Source session IDs and recording timestamps belong in review notes, not
executable YAML. Update the request, orchestration and child skills, schema reference and examples
together when changing this workflow.

### Validate and refine

Run the read-only validator from the repository root:

```bash
uv run --project agent-samples/procedure-capture-guide/worker python -m procedure_capture_guide_worker._validate_guide agent-samples/procedure-capture-guide/<slug>-guide/guide.yaml
```

It imports the runtime parser directly; there is no authoring-specific schema.
It checks state and completion rules, step links, trigger arguments, permitted
tools, strings and duplicate keys. It neither rewrites nor approves a guide.
A schema-valid guide can still omit an action or accept an incorrect result.

With this sample's configured model servers running, evaluate a fixed draft:

```bash
PYTHONPATH=agent-samples/procedure-capture-guide/worker uv run --project agent-samples/procedure-capture-guide/worker python agent-samples/procedure-capture-guide/evaluation/guide/evaluate_recorded_packet.py --packet /path/to/packet.json --guide agent-samples/procedure-capture-guide/<slug>-guide/guide.yaml --output agent-samples/procedure-capture-guide/<slug>-guide/evaluation/run-001
```

The existing runtime evaluator supplies each captured frame once at its original
relative time and sends start and Next utterances through the conversation
model. It records exact guide bytes, source hashes, frame selections, actual
model and tool outputs, guidance state, speech and failures. Draft eligibility
is temporary and in memory. It does not write YAML or change approval status.
The camera endpoint and user text are simulated; hub IPC, browser, STT and TTS
audio are not tested. Refer to `guide-generation-skills/evaluate-guide/SKILL.md`
for artifact details and timing limits.

Use the findings to refine instructions and checks, then rerun from fresh state.
Default to at most three evaluation-and-update rounds and always verify the
final revision. Report actual rounds, not retries or formatting-only edits.
A failed run is an intermediate finding: diagnose the selected frames and both
model outputs before deciding whether a guide correction is possible. Clarifying
an observation question does not require changing the task requirements or
additional approval for the newly authored draft. Check first-completion frames
for premature acceptance; a higher completion count alone is not improvement.
Missing models or evidence are blocked tests, never a pass. A fixed recording
cannot react to guidance, so distinguish timing gaps from guide defects.
Missing visual evidence in the recording does not by itself invalidate an
appropriate completion check for a future worker. Preserve that requirement
and disclose the coverage gap rather than weakening the SOP to make the test pass.

The timer example verifies a visual setup, then starts the timer on entry to
the wait step. It accepts `duration_s` only; no model-written start timestamp.
Each step has its own initially false completion field. A current image cannot
prove a past action sequence, and elapsed time does not prove continuous contact.
Retain unsupported requirements as review blockers rather than weakening them.

Review the YAML and evidence notes before manually setting `task.status` to
`approved`. Resolve every blocker, including any draft step without a completion commit.
Switch to `agent-samples/procedure-capture-guide/` and run an approved guide with
`uv run main.py --mode guide my-procedure-guide/guide.yaml`, then say "start guide".

### Check generation changes

The sample's `evaluation/guide/authoring_cases.yaml` contains text-only authoring probes with
separate inputs and withheld review criteria. They are not image-based runtime
tests. This data stays in YAML so an external authoring model can receive each
capture excerpt without executing test code or seeing the separate reviewer criteria.
Tea execution scenarios cannot test this capture-to-guide authoring behavior.
Refer to the evaluation skill for the authoring comparison procedure.
The offline checks validate the fixture structure, all examples and shipped
guides, and the complete packet-runner invocation:

```bash
uv run --project tests python -m pytest tests/test_sop_guide_generation.py tests/test_sop_recorded_replay.py -q
```

Deterministic model doubles test runtime integration, not runtime-model
perception or authoring-model guide quality. Keep guide-specific refinement
results distinct from sample-level end-to-end regression results.

## Follow an approved guide

Select a reviewed guide YAML file or its containing directory by path. Generated guide packages live in
local, Git-ignored `<name>-guide/` directories; maintained examples live under
`example-guides/`. Guides can also live outside the sample. Set `task.status` to
`approved` only after reviewing the instructions, completion checks, timers,
and safety constraints. A directory must contain exactly one top-level `.yaml`
or `.yml` file; otherwise, provide a specific file path.
The worker loads the selected file and rejects
missing, draft, or invalid guides. It does not search by `task.name` or `task.id`.
The launcher resolves relative paths from `agent-samples/procedure-capture-guide/`
and passes an absolute path to the worker. Symbolic links are supported. Quote paths
containing spaces. Run from the sample directory:

```bash
uv run main.py --mode guide my-procedure-guide/guide.yaml
uv run main.py --mode guide example-guides/tea-making
```

Start the shared models first, as for capture. Guide mode uses the LLM, VLM, CLEF, STT,
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
requiring a user question. Advancement is explicit unless the terminal step
sets `finish_on_complete: true`, which ends the run automatically.

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

`example-guides/tea-making/guide.yaml` is a hand-authored, approved example with task
name **Tea Making** and ID `tea_making`. Its four stages guide brewing
chamomile tea: read the packet, verify the water temperature, put the tea in
water, and start the steeping timer. It is not generated from a user recording.

Start the sample's shared models, then run:

```bash
uv run main.py --mode guide example-guides/tea-making/guide.yaml
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
- Its three visual steps require two distinct fresh frames and two affirmative
  CLEF decisions. Cosmos describes the printed tea name, temperature display, or
  bag-water contact; CLEF compares that evidence with the step's condition. The
  runtime applies the predefined completion patch after confirmation. Neither
  a plain YES from Cosmos nor an incidental matching word completes a step.
  The foreground LLM handles conversation and controls. Both perception and
  decision models can make mistakes; inspect their separate outputs during
  evaluation rather than interpreting a well-formed reply as proof of accuracy.
- The heating check accepts a visible **100 C or 212 F** reading. It requires
  the value and temperature unit, but no CURRENT/ACTUAL label or immersed probe;
  TARGET/SET and unlabeled readings also qualify. This checks displayed text,
  not whether the displayed setting measures the water's actual temperature.
  Guide does not expose the tea sample's typed temperature tools.
- Immersion detection does not start the timer. Say **next** promptly after
  immersion is confirmed to enter the timer step. The runtime then measures a
  full 300 seconds; time spent waiting at the immersion step is not subtracted.
  Timer verification does not inspect images or prove continued immersion.
- The final step sets `finish_on_complete: true`, so expiry ends the active run,
  clears progress, and announces completion without another **next** command.
- Skipping the temperature check advances to tea immersion, then the full
  five-minute timer. The skipped temperature stays unverified, and the run ends
  with a skipped-steps announcement. Skipping packet identification or immersion
  ends the run. Skipping the timer also cannot produce a verified completion.
- The per-run `worker.log` records each visual observation, its step and frame
  identity, the condition, and CLEF's choice and confidence at DEBUG level.
  These records describe verification attempts; two fresh affirmative observations
  are still required for completion. Evidence is saved before the decision request
  so it remains available if CLEF fails.

Regression tests load this exact example and exercise its state transitions,
distinct-frame evidence, skip behavior, and timer with simulated observations.
Those tests do not establish accuracy on real tea-making images or appliances.

#### Local-model guide evaluation

`example-guides/tea-making/evaluation/run_scenario.py` exercises a manifest's selected guide through
the actual guide agent, runtime, frame selection, and model
tools. It uses the sample's configured LLM, VLM, and CLEF. Fixture images replace the
hub camera endpoint, and typed requests enter the same runtime topic as
transcribed speech. Microphones, STT, network camera delivery, and audible TTS
are not covered. The script never manages external model-server processes.

Both the scenario runner and the skill's recorded-packet runner use the production frame freshness and request
timeouts from `yaml/worker.yaml`, and the visual verification timeout from
`yaml/guide-agent-config.yaml` (15 seconds by default).
Scenarios continue scoring later probes after an answer or tool-call
mismatch unless `fail_fast: true`; a runtime exception or timeout still ends the run. Packet evaluation stops
after a failed check because later steps depend on that progress.

The tea step-verification and full-run scenarios cover visual progress, timers,
completion, and start/stop behavior. `example-guides/tea-making/evaluation/scenarios/routing.yaml`
exercises the composed foreground prompt with controls, polite requests,
quoted commands, negation, requests to explain guide content, unsupported capture
requests, and visual questions.
These scenarios run with the full permitted tool set; the evaluator does not preselect
tools based on the utterance.
Both guide evaluators tag model events with their origin and foreground turn ID.
Routing assertions check only calls belonging to the simulated user turn;
concurrent background observation calls remain in the trace but cannot count
as foreground routing successes or failures.

`example-guides/tea-making/evaluation/scenarios/label_reading.yaml` tests the included tea guide with generated label
cards: chamomile alone, longer chamomile names, other teas, and an unreadable
label. It checks real VLM responses, rejection of premature advancement, and
identification announcements. These cards test the response contract, not
photographic accuracy.

A YAML or JSON manifest selects a `guide_path` (relative to the manifest's
directory, or absolute) and lists ordered `actions`:

- `image`: show an image `path`, or a named synthetic `scene`. Optional
  `scenes` map names to a rectangle color, display text, or a tea scene.
  `scenes_file` loads a shared scene map relative to the manifest.
- `observe`: wait for `count` visual results (default two) from the displayed
  image and compare them to the allowed `answers`. Alternatively, name a `case`
  from `visual_cases_file` (relative to the manifest) to reuse its scene, step,
  and expected-answer regex. Recognition comparisons ignore whitespace layout
  and case; unavailable observations never count as correct answers.
- `say`: send `text` as a user turn. Optional `expect` checks the delivered
  answer against a regex; `tools` and `arguments` check model tool calls.
  Status assertions use **What is my status?** and `workflow__status`.
- `wait_speech`: wait for an `expect` regex in speech delivered since the last
  user turn, with optional `timeout_s`. This checks completion notices, including
  automatic timer completion, without reading private backend state.
  Optional `after_speech` and `minimum_elapsed_s` check the elapsed delivery time
  from the first matching origin to completion within the most recent restart.
  Later conversational paraphrases do not move that origin.
- `quiet`: check for additional speech since the preceding `say` response,
  including intervening image changes, and for a further `duration_s`. Use it
  after stopping the guide while a matching image remains visible. Only `image`
  actions may appear between that `say` and `quiet`; observation, speech-wait,
  or other silence checks require a new `say` before another `quiet`.

Timing fields are validated before model calls. `after_speech` and
`minimum_elapsed_s` must appear together; the minimum must be finite and
nonnegative. Timeouts and silence durations must be finite and positive.

The runner makes no assumptions about step names, step count, or a final timer.
Timers use real elapsed time. Include negative observations before positive
ones and explicitly test early advance, completion, stop, and restart as needed.

From the sample directory, with its model servers already running:

```bash
uv run --project worker python example-guides/tea-making/evaluation/run_scenario.py \
  --manifest example-guides/tea-making/evaluation/scenarios/step_verification.yaml --output artifacts/tea-verification-eval
uv run --project worker python example-guides/tea-making/evaluation/run_scenario.py \
  --manifest example-guides/tea-making/evaluation/scenarios/routing.yaml --output artifacts/tea-routing-eval
uv run --project worker python example-guides/tea-making/evaluation/run_scenario.py \
  --manifest example-guides/tea-making/evaluation/scenarios/label_reading.yaml --output artifacts/tea-identification-eval
```

Use a new output directory for each run. Supply `--manifest /path/to/scenario.yaml`
to evaluate another guide and its images.
Outputs include the exact guide, manifest, image hashes, model tool calls, visual
answers, speech text, and result JSON. Synthetic controls can isolate label or
setpoint mistakes but do not establish accuracy on a user's actual camera views.
The manifest's optional boolean `fail_fast` defaults to `false`: mismatched
answers or tool calls fail the run but do not hide later independent probes.
`scenarios/routing.yaml` uses this continuation policy and restarts between probes.
The dependent tea scenarios set `fail_fast: true`, so a failed
check stops the scenario and later actions are reported as not scored rather
than run against an assumed successful history. Runtime exceptions and timeouts
stop either mode; they do not count as ordinary answer mismatches.

#### Generated tea-image evaluation

Tea-specific evaluation data lives under `example-guides/tea-making/evaluation/`. The executable procedure is
`example-guides/tea-making/guide.yaml`; the YAML files below describe test inputs and
assertions, not additional guides.

| File under `example-guides/tea-making/evaluation/` | Purpose |
|---|---|
| `scenarios/step_verification.yaml` | Presents incorrect then correct scenes for each visual step; checks blocked advancement and verified completion, then stops before the timer |
| `scenarios/skip_heating.yaml` | Skips temperature verification, still verifies immersion, and checks that the full steeping timer starts |
| `scenarios/full_run.yaml` | Independent successful run through the five-minute timer, completion, restart, and stop |
| `scenarios/label_reading.yaml` | Checks chamomile name variations, other tea names, and unreadable labels in the identification step |
| `scenarios/routing.yaml` | Independent conversation probes for controls, negation, quoted instructions, and visual questions; restarts before each probe |
| `fixtures/visual_expectations.yaml` | Expected recognition answers and acceptance decisions for isolated and multi-turn visual checks |
| `fixtures/image_scenes.yaml` | Image descriptions for packets, kettle readings, and tea-bag placement |

The adjacent `run_scenario.py` executes the scenario files; `evaluate_visual_checks.py` runs isolated
VLM checks selected by `--manifest`. `_scene_rendering.py` draws their fixtures,
and `_visual_expectations.py` supplies shared scoring. Automated runtime and
evaluator regressions live directly under the repository's `tests/`.
`tests/test_procedure_indicator_guide.yaml` supplies a minimal visual-and-timer
guide for isolated evidence, timing, and failure-reporting checks, not a second
end-to-end scenario.
The sample's standalone `evaluation/capture/evaluate_captions.py` loads synthetic caption inputs
from `evaluation/capture/caption_cases.yaml`. `evaluation/guide/authoring_cases.yaml` contains the
text-only authoring probes described above; `tests/test_procedure_indicator_guide.yaml`
is the only procedure-specific YAML fixture left in `tests/`.

The sample-owned `evaluation/guide/_runtime_harness.py` is an evaluation harness,
not the production guide runtime. It provides simulated camera and speech
boundaries, production agent and tool setup, and trace collection. Its speech
boundary records each published output without running the voice floor, so
update combining and withdraws are not exercised: a notice that production
withdraws on a step transition still appears in the trace.
`tests/test_sop_guide.py` covers those paths through the real floor. Both the
recorded-packet runner and tea scenario runner consume this harness.
Recorded-packet playback and orchestration stay in
`evaluation/guide/`; tea-specific scene rendering,
scripted actions, and assertions stay with the example guide. Neither runner
imports the other's implementation or regression fixtures. The coding-agent
skill invokes `evaluation/guide/evaluate_recorded_packet.py` from the target
checkout rather than maintaining executable evaluation code inside the skill.
Capture evaluation and its cases live in `evaluation/capture/`; text-only
guide-authoring probes live in `evaluation/guide/authoring_cases.yaml`.

Keep the full run independent of step-verification failures so timer and
lifecycle checks remain reachable. Label-reading has its own scenario because
it isolates name variations using text cards without running the rest of the
procedure. Image descriptions and expected answers are separate so both
evaluators share the same fixtures and scoring contract without supplying
expected answers to the renderer or model.

The tea evaluation separates visual recognition from guide execution. All images
are schematic drawings generated by `example-guides/tea-making/evaluation/_scene_rendering.py` from
`example-guides/tea-making/evaluation/fixtures/image_scenes.yaml`. Packet labels contain only tea names and a generic TEA heading,
without brand names or logos. Inspect the drawings before interpreting model scores.
The manifests use `guide_path: ../../guide.yaml`, relative to their own directory,
loading `example-guides/tea-making/guide.yaml` through the production guide runtime.
They exercise the same procedure, prompts, and verification rules that users run,
without a separate copy of the tea guide. The purpose-built
`tests/test_procedure_indicator_guide.yaml` remains a separate, simplified test procedure.
Identification checks whether chamomile is visibly printed on the packet,
using descriptive Cosmos evidence and the shipped guide's CLEF condition.

| Layer | What it checks | What it does not establish |
|---|---|---|
| CPU regression tests | Reproducible images, shipped-guide selection, scoring, timeout handling, and existing guide state-machine guards | Live model accuracy |
| Isolated visual cases | Configured Cosmos descriptions and CLEF decisions for the shipped guide | Conversation, consecutive-frame enforcement, or task completion |
| Multi-turn guide scenario | Actual guide agent, frame tools, local LLM, VLM, and CLEF, controls, announcements, and the five-minute timer | Microphone, STT, network camera transport, or audible TTS |

From the sample directory, generate images for inspection without model calls:

```bash
uv run --project worker python example-guides/tea-making/evaluation/evaluate_visual_checks.py \
  --manifest example-guides/tea-making/evaluation/fixtures/visual_expectations.yaml \
  --render-only --output artifacts/tea-preview
```

With the existing model servers running, evaluate the 27 visual cases three
times each, then run the complete conversational scenario:

```bash
uv run --project worker python example-guides/tea-making/evaluation/evaluate_visual_checks.py \
  --manifest example-guides/tea-making/evaluation/fixtures/visual_expectations.yaml \
  --repeats 3 --output artifacts/tea-visual
uv run --project worker python example-guides/tea-making/evaluation/run_scenario.py \
  --manifest example-guides/tea-making/evaluation/scenarios/step_verification.yaml --output artifacts/tea-step-verification
uv run --project worker python example-guides/tea-making/evaluation/run_scenario.py \
  --manifest example-guides/tea-making/evaluation/scenarios/full_run.yaml --output artifacts/tea-full-run
```

The visual suite covers five tea blends, incomplete and
covered labels, cold through boiling readings, Celsius and Fahrenheit, target
versus current labels, competing readings, and dry, immersed, floating, and
obscured tea bags. Expected answers describe scene content and are used only for
scoring, not supplied to the VLM. Isolated and conversational checks share this
answer table. The suite reports recognition accuracy (allowing whitespace layout
and equivalent temperature notation),
evidence-gate accuracy, false accepts, false rejects, and unavailable requests
separately. An unavailable model cannot pass a negative case. Correctly rejecting
a scene while misreading its contents is not a recognition pass. Repetitions
measure response variability on the same images, not additional independent scenes.
Decision scores call the production CLEF verifier on the original description.
If CLEF fails, the case is unavailable for decision scoring, but its VLM answer
and recognition score are retained. `recognition_scored` counts usable VLM
answers; `scored` counts cases with both a usable answer and a CLEF decision.
Use each count as its corresponding accuracy denominator. Result rows record VLM and decision errors
separately; later cases continue and the final summary includes the failure.
Recognition regexes are offline diagnostics only and never authorize progress.
They can also miss valid paraphrases, so retain and inspect the raw descriptions.

The step-verification scenario tests early advance rejection, completion
announcements, and explicit advancement for the three visual steps, then stops.
The full-run scenario tests successful visual verification, the real five-minute
steep, return to inactive state, restart, stop, and silence after stop. Its completion check
requires at least 295 seconds between timer-entry and completion speech, allowing
five seconds for speech delivery while leaving the runtime's 300-second timer
unchanged. Allow more than five minutes for a complete run. Fixture pixels are
republished as a simulated live stream: repeated source pixels do not demonstrate
recognition across different camera views. Existing CPU tests separately enforce
distinct-frame identity and stale-observation rejection.
Post-stop checks include the image switch and ten additional seconds of silence.
This is a bounded observation window, not a guarantee against announcements
delayed beyond it by slow inference.

Both runners exit nonzero on failed checks and preserve evidence in a new output
directory. The visual runner saves PNGs, scene definitions, guide and source
hashes, guide questions, raw answers, timing, and per-step scores. Compare
runs using the same guide, fixtures, model configuration, and renderer versions.
A synthetic pass is
not evidence of safe operation with real appliances or general photographic accuracy.

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
- `voice` contains the foreground prompt and permitted tools. Visual steps omit
  `agent`: Cosmos describes the scene and CLEF judges the written evidence.
  Timer steps may use an `agent` policy when they do not directly commit evidence.
  Foreground workflow controls are supplied by the runner, not the guide.
- Visual `evidence` requires a prose `condition`, positive `consecutive` count,
  and a bounded `commit` mapping for approved guides. The question asks Cosmos
  for observable facts; the condition tells CLEF which facts must all hold.
  Do not ask Cosmos for a YES/NO or readiness token. CLEF receives the description
  and condition, without images or conversation history, and chooses yes or no.
  Only yes counts toward completion. Ambiguity, missing facts, or contradictory
  evidence should yield no. Confidence is logged, not used as another threshold.
  Visual steps reject `evidence.pattern` and observation `agent` policies.
- Timer `evidence` retains its full-match `pattern` (normally `^true$` on
  `expired`), `consecutive`, and direct `commit`. No CLEF judgment measures time.
- Every completion patch must satisfy `complete_when` and use only typed writable
  fields. Each step needs an initially unmet completion field that preceding
  steps cannot write. Visual confirmations require distinct source frames,
  identified by participant, track, timestamp, and sequence. Repeated or older
  frames do not increment the count. Unavailable evidence, failed decisions,
  and cancelled decisions break the streak; results from an old step or run
  cannot commit. A draft with an unresolved verification requirement may omit
  its visual commit; it cannot complete and cannot be approved until resolved.
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

The outer `main.py` launches the hub and worker, plus the shared recorder when
`--session-recording` is enabled. The worker's `__main__.py`
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
runtime timers, frame evidence, and background verification.
`guide/verification.py` uses the typed decision service to judge descriptive
evidence; its Relay trace retains the source frame, step, and revision. Transitions clear
local notices under the session lock, so obsolete completion notices cannot
follow the new step's instruction.
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
Guide mode gives `VoiceAgent` an LLM update combiner for unprompted guide
announcements, and uses `ImageQueryTool`, native tools, and the
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
| `yaml/guide-agent-config.yaml` | Visual and CLEF decision timeouts |
| `yaml/models.json` | Shared batch STT, LLM, VLM, TTS, and CLEF services and endpoints |
| `yaml/media_capture.yaml` | Opt-in session recording output, profile, and participant lifetime |
| `yaml/voice_gate.yaml` | Shared wake-word-free speech input; each mode handles its own commands |
| `yaml/device_io_hub.yaml` | Room and web server |

`worker.yaml` selects the capture YAML through `capture_config`, resolved relative
to the worker YAML. `artifacts_dir` resolves relative to the selected capture
YAML, even when it is in another directory. Model and voice configuration remain
beside the worker YAML. Absolute configuration and output paths are also supported.
The worker loads shared settings into `WorkerConfig` and capture-specific settings
into `CaptureConfig` under `capture/config.py`; `app.py` supplies each to its
owning component. Guide mode selects its YAML through `guide_config` and loads
`GuideConfig` from `guide/config.py`. The required positional `GUIDE_PATH` selects
a guide file or a directory containing exactly one top-level YAML file. The
launcher resolves sample-relative paths and passes an absolute path to the worker.
`task.name` and `task.id` remain guide metadata, not CLI selectors.
`--output-root` applies only to capture. Restart after configuration edits.
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
`example-guides/tea-making/evaluation/run_scenario.py` scenarios above against existing model servers to measure live
tool routing and visual judgments; the runner never manages their lifecycle.
