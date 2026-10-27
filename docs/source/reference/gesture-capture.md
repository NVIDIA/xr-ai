<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Gesture capture

The `gesture-capture` sample reads hand landmarks from mono video. A held
camera-box gesture or a pinched circle selects a section of the camera view.
Both crop the selected streamed video frame by default and ask the configured
VLM to describe it. Optional aligned-photo mode uses a separate still photograph
that can have a higher resolution. The sample launches a hub and worker; the
VLM service runs separately. Start commands and common settings are in the
[sample README](https://github.com/NVIDIA/xr-ai/tree/main/agent-samples/gesture-capture).

## Compose the capabilities

```text
Live image → hand landmarks → sample region selector
           → selected video frame or aligned still → image crop → VLM query → gesture.response
```

`GestureCaptureAgent.step()` shows the composition. `CurrentFrameTool` selects
live images with photo fallback disabled. `HandTrackingTool` returns 21 landmarks
per hand and knows nothing about gestures or capture. The sample-owned
`RegionSelector` composes the two gestures into an `ImageRegion`; the agent
uses the same image-selection, crop, and VLM query sequence for either gesture.

`aligned_photo: false` selects the same frame used for the final gesture
observation. The crop uses that frame's actual streamed resolution and coordinate
system; this mode sends no photo request. It works with ordinary video streaming
and needs no photo calibration acknowledgement.

With `aligned_photo: true`, `StillCaptureTool` requests a new device photograph,
even with fresh video available. It returns the actual dimensions and an opaque image reference.
`ImageCropTool` crops that reference independently of hand tracking.
`ImageQueryTool` resolves the crop through the typed VLM service. Changing the
question does not require changing sensing or acquisition. Both gestures use
VLM inference only; no dedicated OCR model or OCR result parser is involved.
A question can still ask the VLM to read or explain text in the selected image.
The packaged system prompt treats framing hands as selectors, asks about visible
target content, and requires an obstruction response rather than inventing hidden
objects when the selected view is blocked.

The worker owns one polling task per participant and one shared local hand
detector. IMAGE mode keeps observations independent across participants. Native
calls are serialized off the event loop; teardown waits for actual native
inference after polling cancellation. Disconnect cancels the participant task
and releases its image handles. Tool execution uses Relay, with raw pixels
kept outside tool requests and results.

## Select a region

For the box gesture, form two opposing L shapes with the thumbs and index
fingers and hold the framing steady. The sample checks the finger angles and
matching corners before acquiring the hold. It then crops the estimated open
interior, immediately while the framing hands remain in place. Each finger's
inward clearance is estimated from its distal joint-to-tip length using
`finger_clearance_ratio`. MediaPipe landmarks are centerlines, not contours;
this is an approximation requiring camera-specific tuning, not segmentation.
The crop is the largest centered axis-aligned rectangle inside all four inferred
inward finger boundaries. For a rotated frame it excludes the outer bounding
rectangle's corner regions rather than accidentally including the fingers. Relax
both hands' pose while they are visible to rearm it. Missing hands alone do not count as an intentional
release of a held box.

For the circle gesture, pinch thumb and index finger and move the pinch point
around the desired section. Either direction works. A near-closed loop needs at
least 300 degrees of observed travel and endpoints within one fitted radius;
240/270-degree arcs, tiny loops, elongated shapes, scribbles, sparse motion and
excessive backtracking remain rejected. The chime freezes the trace's bounding
rectangle (no circular mask). Keep the camera and target steady and move the
drawing hand away from that rectangle. Stable unpinch afterward rearms detection.

Circle capture waits for that hand's landmark bounding rectangle to overlap at
most `circle.clear_overlap` (default 20%) of the smaller of the hand and selected
rectangles, continuously for `release_s`. It then crops a **newer video frame**.
After `circle.clear_timeout_s` (default 2 seconds), it captures the latest frame
anyway: this is a best-effort wait, not a guarantee of an unobstructed target.
Missing or ambiguous hands do not prove clearance; they use this same deadline.
Unrelated distant hands do not need to disappear. A bystander inside the selection
may still obscure the target; the VLM is instructed to report obstruction.
Box capture remains immediate, inside the framing fingers.

Grip memory is OPEN or PINCHED; missing observations supply unknown evidence,
not a new release or engagement. Two accepted closed observations enter
PINCHED; duplicates and missing/outlier observations cannot supply the second
sample. Strong closed candidate points are retained, so pinch-and-immediately-draw
keeps its first arc without a stationary hold. Once pinched, the gap between
`pinch_ratio` and `release_ratio` retains both grip and usable observed pointer
motion. Uncertain grip distance does not mean the pointer disappeared.

Only sustained positive open evidence over `release_s` (default 0.4s) releases
the grip. An open outlier suppresses that sample without releasing; intermediate
or missing evidence breaks release confirmation. Missing/ambiguous/outlier pointer
observations add no trajectory points. `circle.occlusion_grace_s` (default 2s)
retains confirmed grip and the current path while the hand is missing, ambiguous
or implausible on otherwise fresh video. The progress hum continues during this
bounded pause. Reacquisition must match the existing spatial continuity limits;
a distant/ambiguous hand cannot continue the path. A reliable return position is
retained for the fit, bounds and point count, but cannot recognize on that first
return. The unobserved segment to it contributes no angular travel; subsequent
observed segments count toward circle completion.

Longer occlusion **cancels the stroke attempt**, discarding its points; it never
means OPEN/release, recognizes a circle or submits a query. Confirmed grip memory
survives, so a recovered held pinch can retry a fresh path without a synthetic
new engagement. Only positive stable OPEN evidence disengages.
Once a circle has been **recognized**, its release latch survives tracking loss:
a held pinch cannot trigger another turn until visible, sustained open evidence
rearms it. Pending capture/query/response cooldown remains in force.

The ratio uses image-space thumb/index distance divided by palm width. Relative
z is deliberately excluded: the attempted 3D-distance/edge-on gate rejected
otherwise usable projected traces under depth-estimate jitter. No visibility or
per-joint confidence is available, and handedness score is not such confidence.
Projection overlap can still look pinched, and self-occluded/dorsal hands may
produce incorrect landmarks. These limitations require physical-camera testing.

Recognized circle selections and pending clearance are independent of subsequent
grip uncertainty: the existing spatial clear-or-timeout policy continues. Brief
absent, ambiguous or implausible observations retain the box candidate for
`max_gap_s`, and the circle path for `circle.occlusion_grace_s`, measured since
the last accepted hand observation. They add no trajectory
points and break release confirmation. Longer hand loss discards failed acquisition
but preserves deliberate grip state; recognized turns still require open to rearm. Duplicate
frames cannot advance hold, release, drawing or clearance. Source changes,
dimension changes and long video gaps cancel pending selections rather than
querying an old frame.

Spatial continuity uses image pixels and palm scale, ignoring handedness label
flicker and tolerating a distant second detection while tracing. Ambiguous nearby
hands retain only bounded grace. This is best-effort spatial association, not
person identity: nearby hand switches cannot always be detected. MediaPipe
IMAGE mode processes frames independently; changing a tracking-confidence setting
would not supply temporal tracking. The existing detector requests up to two
hands, including single-hand input; three or more visible hands can exceed its
capacity. No backend confidence controls or new SDK API are introduced here.

Defaults sample every 80 ms and require eight distinct points, allowing faster
natural traces than the previous 150 ms/twelve-point policy. This increases
potential CPU use; inference time also limits actual cadence. Thresholds are demo
tuning values, not measured recognition accuracy; qualify on the target camera.

The timeouts serve separate purposes: `circle.occlusion_grace_s: 2.0` tolerates
hand-observation loss before recognition, `max_gap_s: 0.5` still bounds actual
video/source continuity and box acquisition, and `circle.clear_timeout_s: 2.0`
still controls best-effort hand clearance **after** recognition. Increasing hand
grace does not permit stale camera frames or extend pending capture indefinitely.

## Audio feedback and cooldown

With `gesture_audio: true`, each participant hears a quiet hum while a valid
box hold is being acquired or a pinched circle is actively being drawn. A
stationary pinch stays quiet until its traced span shows movement. A single
wake-word-style perfect-fifth chime marks recognition: completed stable box hold,
or the first valid near-closed circle. The hum stops before the chime.

The sample sends paced float32 PCM through the existing hub return-audio path;
it does not start a voice or TTS stack. Browser audio playback must be permitted.
Cues stop on cancellation, observation errors, expired progress, disconnect,
and shutdown. The helper queues short chunks and drains its sends; it does not
flush the participant's shared audio queue or interrupt other producers. A short
tail already in the hub or client buffer can still finish playing.

A participant stays in cooldown from recognition through VLM response delivery
or failure cleanup, including the circle clearance wait. Hand observations continue
during that wait, but frames received during the turn cannot trigger another
query or chime. After the turn ends, the box still needs a visible relaxed pose
and the circle needs a stable open pinch before another deliberate gesture.
Set `gesture_audio: false` for silent operation.

## Diagnose a flaky gesture

Set `gesture_diagnostics: true` in `yaml/gesture_capture_worker.yaml` for a
reproduction on your next normal sample launch. It defaults to false. This adds
at most one INFO JSON summary per second per participant through the existing
worker logger, visible in the launch terminal and
`/tmp/log_gesture-capture_<run timestamp>/worker.log`. `XR_AI_LOG_ROOT` overrides
the default `/tmp` root. Filter the text for `"event": "gesture_diagnostics"`.
Allow several seconds of drawing and share those summary lines when diagnosing
an issue; enabling the option does not change gesture thresholds or behavior.

The summaries contain counts and numeric min/max ranges, not raw camera/audio,
landmark coordinates, image references, participant/track identifiers or absolute
frame timestamps. They use no extra recording files or telemetry sink. Anonymous
summaries from multiple participants can interleave.

- `hands_0`, `hands_1`, `hands_2` and `hands_3` (three or more) count actual
  HandLandmarker results. `no_hands_returned` means the overall MediaPipe task
  returned no hands; it cannot distinguish its internal palm-detection failure
  from landmark/presence rejection.
- `backend_ms` is wall time of the hand tool call, including decoding and any
  lock wait, not a pure model benchmark. `frame_interval_ms` is the interval
  between observed source-frame timestamps; `duplicate_frame` and
  `frame_unavailable` distinguish sensing gaps from absent hand detections.
- `pinch_ratio`, `tip_distance_px` and `palm_width_px` separate the normalized
  distance from its numerator/denominator; palm foreshortening can inflate the
  ratio. These ranges are independent; `pinch_peak` also retains the paired
  ratio/tip-distance/palm-width values from the window's maximum-ratio observation.
  `evidence_closed/open/band`, `grip`, `grip_to_*`,
  `entry_pointer_added` and `pointer_added` show whether grip admission or usable
  trajectory sampling is progressing. `association_ambiguous`,
  `association_no_usable_nearby_hand`, `pointer_jump` and `pointer_outside` count
  sample continuity rejections, rather than blaming the model for all misses.
- `circle_points`, `circle_min_points`, `circle_too_small`,
  `circle_elongated`, `circle_degenerate_fit`, `circle_endpoint_gap`,
  `circle_radial_error`, `circle_sparse_segment`, `circle_sweep` and
  `circle_backtrack` locate the first failed geometry check. Numeric
  `trace_width_px`, `trace_height_px` and `endpoint_gap_radius` expose span/closure
  geometry without logging coordinates. Many checks are expected
  while a partial circle is still being drawn; their counts alone do not prove
  poor recognition. `tracking_gap`, `circle_reset` and
  `selector_dimensions_or_frame_gap` show continuity resets.
  `fresh_attempt_after_loss` counts discarded acquisitions eligible for fresh
  entry; `rearm_wait_after_completion` identifies the recognized-turn release latch.
- `circle_recognized` / `box_recognized` and `pending_cleared` /
  `pending_timeout` separate recognition from the circle's later capture point.
  `pending_association_unknown`, `pending_source_or_frame_gap` and
  `pending_no_fresh_video` explain why waiting continued or was cancelled.

These observations help isolate detector/landmark availability, cadence and
sample decisions. They do not establish back-of-hand accuracy or fix an
unobserved physical-camera failure. No detector replacement is enabled here.

## Provision the hand model

The optional `hand-tracking` extra uses the reviewed MediaPipe 0.10.31 release.
Its wheels support Linux x86-64 with glibc 2.28+, macOS ARM64, and Windows x64.
There is no Linux ARM64 wheel for this release. The sample launcher and hub
retain their existing platform requirements; wheel availability alone does not
establish that the full sample runs on Windows or macOS.

The model is provisioned explicitly; the SDK does not download it. From
`agent-samples/gesture-capture/`:

```bash
mkdir -p models
curl --fail --location \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
  --output models/hand_landmarker.task
printf '%s  %s\n' \
  fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1 \
  models/hand_landmarker.task | sha256sum --check
```

Both the package and the official model card identify Apache 2.0 licensing.
The wheel also includes a Lucent permissive notice. Refer to the repository's
[third-party notices](https://github.com/NVIDIA/xr-ai/blob/main/THIRD_PARTY_NOTICES.md#mediapipe-hand-landmarks)
for artifact scope, transitive dependencies, and attribution files. The
[official model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf)
describes limitations with occlusion, gloves, motion blur and lighting; gesture
recognition here is a demonstration requiring target-camera tuning.

## Configure the image path

Select **Live** camera mode in the web client. The existing capture callback
handles optional aligned-photo requests through the browser's native
`ImageCapture.takePhoto()` API. It requests the maximum photo dimensions
reported by the camera without changing video constraints. A camera may return
less than 4K or pause video during exposure. Unsupported browsers reject the
request in aligned-photo mode; it does not silently change to a video snapshot.
Default video-frame mode does not require this browser photo API.

Custom device applications implement `onImageCaptureRequested` with their own
photo API. This permits image-only sensor resolutions while low-resolution
video continues to supply hand observations. The existing request, response,
and cancellation protocol carries the encoded image independently of video.
An aborted request suppresses delivery; a device exposure already started by
its native camera API may still finish. Native iOS and Android high-resolution
photo adapters for live mode are separate platform work; this sample's built-in
photo implementation is web-only.

Leave `aligned_photo: false` for video-frame queries at startup. To opt into
higher-resolution stills, set `aligned_photo: true` only after verifying that
video and stills come from the same camera, with the same optical framing and
orientation. Coordinates
refer to source pixels, independent of a mirrored preview. EXIF orientation is
normalized when processing stills. The flag is a manual assertion; the sample does
not calibrate, register, or transform coordinates between the two images. In
aligned-photo mode, the sample rejects different photo and video aspect ratios
before cropping or querying. Equal aspect ratios cannot detect digital zoom,
stabilization,
lens switching, changed field of view, or scene motion between gesture and
exposure; these need device-specific calibration or alignment. Keep the camera
and framed target steady until capture finishes.

## Settings and output

The orchestrator selects `yaml/device_io_hub.yaml` and
`yaml/gesture_capture_worker.yaml`. The latter resolves `model_path` and
`models` relative to its own directory. `models.json` owns the VLM adapter and
endpoint; it does not deploy that server. Restart the sample after editing
settings. Model-server endpoint changes must also match the separately started
service; refer to {doc}`/guides/customizing-model-servers`.

The worker YAML owns observation cadence, hold/release durations, the maximum
observation gap, movement and geometric tolerances, alignment acknowledgement,
the general VLM question, cue enablement, finger clearance, and circle settings. These are demonstration settings, not measured accuracy
claims. The box returns an axis-aligned rectangle inside the estimated finger-frame
interior; the circle returns the bounding rectangle of its observed path.

Results are UTF-8 messages on `gesture.response`, visible in the web client's
message log. A capture failure is logged and does not automatically repeat a
held box or resubmit a released circle. Perform a new deliberate gesture before
trying again. The `query` setting supplies the same VLM question for either
selection; its default asks for a description and what is notable in the region.
