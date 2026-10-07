<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP sample

The SOP sample captures narrated demonstrations and provides a coding-agent skill
for turning saved evidence into draft procedural guides. The capture process
does not generate, approve, or execute guides automatically. Its only supported
mode is `--capture`. Guide replay is not implemented in this capture stack.

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
The sample creates a TTS client to satisfy the shared voice runtime contract,
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

## Generate a draft guide

After the packet is finalized, use a coding agent with local file and image
access to follow
[`recording-to-guide`](https://github.com/NVIDIA/xr-ai/blob/main/agent-samples/sop-sample/skills/recording-to-guide/SKILL.md).
This is the manual generation stage: the model reads the capture and skill,
writes the guide YAML directly, runs validation, and repairs its YAML if needed.
There is no Python YAML generator, plan compiler, automatic approval, or new
model-server requirement. Selecting GPT Luna happens in the coding agent, not
in the capture sample's `models.json`.

From the repository root, give the coding agent this request, replacing the
session placeholder:

```text
Read agent-samples/sop-sample/skills/recording-to-guide/SKILL.md and follow it.
Use agent-samples/sop-sample/artifacts/sessions/<session-id>/packet.json.
Read the referenced schema and examples. Reuse the saved captions and narration,
and inspect selected frames when they are insufficient. Write a draft guide
YAML directly and concise evidence review notes under agent-samples/sop-sample/guides/.
Run the documented validator and fix any errors. Do not approve the guide.
```

No skill installation is required for this explicit-file workflow. To install
it into an agent's skill directory, copy the **entire** `recording-to-guide/`
directory, including `references/` and `agents/`, rather than only `SKILL.md`.
Keep the target repository and sample directory explicit when using an installed
copy. An agent without an image tool can draft from available evidence, but it
must report uninspected visual ambiguities rather than claim they are resolved.

The outputs are local and Git-ignored:

- `guides/<slug>.guide.yaml`: authoritative, model-authored draft guide.
- `guides/<slug>.review.md`: evidence mapping, caption corrections, and concrete
  approval blockers; this is not executable guide state.

Generated task names use 1–5 punctuation-free words. The skill preserves narrated
actions, prerequisites, counts, and orientations; it does not use activity labels
as an exhaustive list of steps. It reads captions first and inspects selected
frames when necessary. Missing narration or occluded actions remain evidence
gaps. A schema-valid guide can still omit a physical requirement, so human review
must compare both the instructions and completion questions with the recording.

### Validate and review

From `agent-samples/sop-sample/`:

```bash
uv run --project worker python -m sop_sample_worker._validate_guide guides/<slug>.guide.yaml
```

The read-only validator checks the same guide contract used by the SOP replay
work: state types and write sets, linked steps, allowed tools, trigger arguments,
result fields, and the 500-character visual question limit. Text fields must be
actual strings. Duplicate mapping keys are rejected at every level, including
overrides introduced through YAML merges, instead of silently replacing earlier
requirements. It reports errors
with a nonzero exit status. It does not generate, rewrite, approve, or execute
the guide, and it does not certify evidence coverage or physical correctness.

Review the evidence notes and resolve every blocker before manually changing
`task.status` from `draft` to `approved`. Preserve the original capture. When
revising a guide, increment `task.version` and check that the title, step prompts,
and completion messages still describe the same procedure. Execution requires
the separate replay implementation; this stack only captures and validates.

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

The skill incorporates lessons from smaller-model evaluations: explicit action
coverage, selective image inspection for stale captions, concrete verification
questions, consistent state-field names, and block-style YAML. These instructions
reduce known failure opportunities; they are not a measured accuracy guarantee
for a particular model version.

Synthetic evaluation cases live in
`tests/fixtures/sop_guide_generation/cases.yaml`. They cover missed narrated
additions, prerequisites and counts, uncertain transcription, elapsed waits,
and unsupported temporal verification. Each contains capture excerpts and a
semantic review checklist, not a ready-made guide or third-party recording.

To evaluate a skill revision, give a fresh coding-agent session the skill and
one case's `input` only. Ask it to write a draft YAML and review notes in a
temporary local directory; do not supply `checks` until scoring the result.
Validate the produced guide, then review it against every case check. Record
the model, skill revision, schema errors, missed checks, and token usage when
the provider exposes it. Do not count guessed token usage or treat schema
validity as semantic accuracy. The text-only cases do not measure caption or
image-understanding accuracy; supplement them with local captured media for
those claims.

Run the offline schema and validator regressions from the repository root:

```bash
uv --config-file uv.toml run --project tests pytest tests/test_sop_guide_generation.py
```

## Composition and configuration

The orchestrator starts the hub, shared `device_io_capture` service, then the
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
