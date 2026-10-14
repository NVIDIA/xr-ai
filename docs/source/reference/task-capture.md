<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Task capture

Task capture records narrated demonstrations between spoken commands. It saves
snapshots, visual captions, and transcribed narration for review and downstream
processing.

## Run a demonstration

Run commands from `agent-samples/task-capture/`:

```bash
uv run --project ../../model-server-samples/model-servers model_servers --models yaml/models.json
uv run main.py
```

The sample's `yaml/models.json` configures both the worker's model clients and
the shared launcher's managed services: batch Parakeet STT on port 8103, Cosmos
VLM on 8100, and Pocket TTS on 8105. Batch STT accepts narration without speaker
enrollment. Use `--models yaml/models.json`: the launcher's default speech
service uses a different streaming protocol and requires speaker enrollment.

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
uv run --project ../../model-server-samples/model-servers model_servers --models yaml/models.json --stop
```

## Output destination

Choose an output root at launch without editing configuration:

```bash
uv run main.py --output-root /path/to/task-evidence
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

## Composition and configuration

The outer `main.py` launches the hub and worker. The worker's `__main__.py`
accepts configuration and readiness arguments; `task_worker/app.py` constructs
the hub endpoint, model clients, voice runtime, and frame/image tools.
The worker uses `VoiceAgent` for transcripts, `CurrentFrameTool` for snapshots,
and `MultiImageQueryTool` for captions.

The registered `CaptureAgent` in `task_worker/capture/agent.py` owns commands,
narration admission, participant generations, and acknowledgements. Its private
`_Capture` helper in `capture/capturing.py` owns sampling, caption tasks,
persistence, and finalization.
`capture/captioning.py` defines the request and response contract used by both
recording and evaluation. Application cleanup explicitly drains capture work
before closing its frame endpoint and VLM client.

The worker keeps shared wiring outside the capture subpackage. A future replay
capability can reuse that infrastructure with its own state; this sample
currently registers only voice I/O and capture.

The always-on voice gate passes commands and narration through the same accepted
input path. Recording acknowledgements follow voice-input handling, so a command
cannot cancel its own response.

| Configuration | Settings |
|---|---|
| `yaml/worker.yaml` | Shared frame and speech detection settings; capture configuration path |
| `yaml/capture-agent-config.yaml` | Task output directory, sampling rate, and caption interval |
| `yaml/models.json` | Managed batch STT, VLM, and acknowledgement TTS services and endpoints |
| `yaml/voice_gate.yaml` | Silent, wake-word-free input with sample-owned recording commands |
| `yaml/device_io_hub.yaml` | Room and web server |

`worker.yaml` selects the capture YAML through `capture_config`, resolved relative
to the worker YAML. `artifacts_dir` resolves relative to the selected capture
YAML, even when it is in another directory. Model and voice configuration remain
beside the worker YAML. Absolute configuration and output paths are also supported.
The worker loads shared settings into `WorkerConfig` and capture-specific settings
into `CaptureConfig` under `capture/config.py`; `app.py` supplies each to its
owning component. Restart after configuration edits.
Refer to {doc}`configuration` and {doc}`command-line` for fields and CLI options.
