<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP sample

The SOP sample captures narrated demonstrations. It records the source evidence
for a later procedural guide; it does not generate, approve, or execute guides.
The only supported mode is `--capture`. Guide replay is not implemented.

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
also applies during shutdown. For accepted recordings, if no manifest arrives,
restart and graceful shutdown remain blocked until capture finalization can complete.
The worker also waits for the new bundle's recorded start event, retrying the
start command if capture is still closing the previous bundle. Frames and
captions for the new packet begin only after that acceptance check succeeds.
Departure and worker shutdown cancel pending starts without waiting behind the
camera-event queue. Queued starts for that connection are discarded. A packet
created for a start that was never accepted is finalized as `incomplete` without
waiting for a nonexistent media manifest. If capture accepted the start before
cancellation, normal manifest, narration, and pending-write draining still apply.

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
