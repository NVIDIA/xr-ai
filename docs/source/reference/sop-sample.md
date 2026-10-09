<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP sample

The SOP sample records narrated demonstrations between spoken commands. It saves
source evidence for a later procedural guide; it does not generate, approve,
or execute guides. Guide replay is not implemented in this capture-only branch.

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
