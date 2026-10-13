<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP sample

The SOP sample records narrated demonstrations between spoken commands. It saves
snapshots, visual captions, and transcribed narration for review and downstream
processing.

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

Use live video or on-demand image capture. Speak commands directly, without a
wake phrase. After **start recording**, the worker says "Recording started."
and stays silent during narration. After **stop recording**, it says
"Recording stopped." The SOP packet, transcript, and summary then finish saving
silently; processing errors are logged. This does not generate a guide or finalize
the independent shared media capture bundle. Disconnecting still finalizes the
packet without announcements.

To stop the sample, press Ctrl+C. The shared model servers remain running.

## Optional media capture

To additionally save participant-wide video, bidirectional audio, and data-channel
traffic for the whole connection, add `--capture`:

```bash
uv run main.py --capture
```

The shared capture bundle starts on connection and ends on disconnect or service
shutdown. One bundle may span several voice-controlled SOP recordings. The
default demo profile saves MP4 output when streamed video is available, along
with source media and a timeline. Audio, transcripts, and metadata can be saved
without streamed video; on-demand stills remain in the SOP packet as snapshots.

Refer to {doc}`/getting_started/requirements` for model and hub deployment and
{doc}`/components/server-runtime` for shared media capture. Optional video
recording uses the shared NVENC encoder and requires supported NVIDIA hardware.
SOP snapshot recording requires STT and VLM services, plus TTS for the recording
acknowledgements; the NVENC requirement
applies to optional shared video capture.

## Recording boundaries

- After connecting, the worker waits for **start recording** to open a packet.
  **stop recording** finalizes it. Repeat the pair to record another demonstration
  on the same connection; duplicate starts during recording are ignored.
- Say each command as a standalone utterance. Matching ignores case, surrounding
  whitespace, and trailing sentence punctuation. Other speech is saved as
  narration while recording, including phrases such as “stop” or “stop capture.”
  The two control commands are excluded from narration.
- Recording and narration continue while the camera is off or images are
  unavailable. When a fresh live frame is unavailable, the worker requests a
  still from clients supporting on-demand capture; this can briefly activate
  the camera. Captioning resumes when a distinct image is saved.
- Disconnect or graceful shutdown finalizes the active packet. Reconnecting
  returns the worker to waiting for **start recording**.

Commands take effect after speech recognition finishes. Keep the microphone
enabled and pause after the final narration before saying **stop recording** or
disconnecting. A late transcript can miss the recording boundary; optional shared
capture retains source audio independently.

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
| `artifacts/sessions/<session-id>/packet.json` | SOP status, participant, timestamps, counts, and activity hierarchy |
| `artifacts/sessions/<session-id>/frames/` | JPEG samples targeting 2 fps and timestamped `index.jsonl` |
| `artifacts/sessions/<session-id>/captions.jsonl` | Timestamped captions, activities, phases, and visible changes |
| `artifacts/sessions/<session-id>/transcript.jsonl` | Final user transcripts inside the SOP recording boundary |
| `artifacts/sessions/<session-id>/summary.md` | Human-readable activity and phase summary |
| `artifacts/sessions/<session-id>/errors.jsonl` | Frame, caption, or persistence errors, when present |
| `artifacts/captures/<bundle>/` | Optional participant-wide media, transcripts, timeline, and manifest from `--capture` |

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

The SOP transcript contains narration between the spoken commands. The optional
shared transcript covers the connection, including those commands. Match shared
bundles to SOP packets using participant identity and timestamps.

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

## Composition and configuration

The launcher starts the hub, optional shared capture service, and SOP worker.
The worker uses `VoiceAgent` for transcripts, `CurrentFrameTool` for snapshots,
and `MultiImageQueryTool` for captions.

The always-on voice gate passes commands and narration through the same accepted
input path. Recording acknowledgements follow voice-input handling, so a command
cannot cancel its own response. Shared media capture independently receives the
raw final transcripts.

| Configuration | Settings |
|---|---|
| `yaml/worker.yaml` | Output paths, frame and caption intervals, speech detection |
| `yaml/models.json` | STT, VLM, and recording-acknowledgement TTS endpoints |
| `yaml/voice_gate.yaml` | Silent, wake-word-free input with sample-owned recording commands |
| `yaml/media_capture.yaml` | Shared capture output, encoding, and retention; defaults to participant mode and the demo profile |
| `yaml/device_io_hub.yaml` | Room and web server |

Paths resolve relative to their owning YAML. Restart after configuration edits.
Refer to {doc}`configuration` and {doc}`command-line` for fields and CLI options.
