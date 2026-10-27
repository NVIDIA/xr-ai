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

The worker owns one polling task per participant and one shared local hand
detector. IMAGE mode keeps observations independent across participants. Native
calls are serialized off the event loop; teardown waits for actual native
inference after polling cancellation. Disconnect cancels the participant task
and releases its image handles. Tool execution uses Relay, with raw pixels
kept outside tool requests and results.

## Select a region

For the box gesture, form two opposing L shapes with the thumbs and index
fingers and hold the framing steady. The sample checks the finger angles and
matching corners before acquiring the hold. Relax both hands' pose while they
are visible to rearm it. Missing hands alone do not count as an intentional
release of a held box.

For the circle gesture, pinch thumb and index finger with one hand visible,
move the pinch point around the desired section in a closed circle, and hold
the pinch open briefly to submit. Either drawing direction works. The sample
uses the trace's bounding rectangle, rather than masking the exact circle.
It rejects tiny loops, open arcs, elongated shapes, large radial deviations,
sparse motion, multiple revolutions, and excessive backtracking. Keep the target
and camera steady until the photograph finishes.

Pinch entry and exit have different palm-scaled distance thresholds. The
`release_s` setting debounces release for both gestures. A one-frame open pinch
cannot submit a trace. Brief absent or ambiguous detections preserve the box
candidate or circle path up to `max_gap_s`; they add no trajectory points and
break release confirmation. Longer loss cancels acquisition. A cancelled circle
requires an observed open pinch before another trace can begin. Duplicate frames
cannot advance a hold, release, or drawing.

Both geometry and trajectory continuity use source-image pixels, accounting for
aspect ratio. Circle input requires one hand; detected handedness changes and
implausible pointer jumps cancel the trace. IMAGE-mode handedness is a classifier
label, not a stable hand identity, so switching between nearby hands with the
same label cannot always be detected. Changed camera tracks or dimensions reset
source continuity. These heuristics require target-camera qualification.

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
the general VLM question, and the circle drawing settings. These are demonstration settings, not measured accuracy
claims. Each gesture returns an axis-aligned bounding rectangle: the framing corners
for the box, or the observed path for the circle.

Results are UTF-8 messages on `gesture.response`, visible in the web client's
message log. A capture failure is logged and does not automatically repeat a
held box or resubmit a released circle. Perform a new deliberate gesture before
trying again. The `query` setting supplies the same VLM question for either
selection; its default asks for a description and what is notable in the region.
