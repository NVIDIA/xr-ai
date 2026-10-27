<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Gesture capture

Frame a box with both hands' index fingers and thumbs, hold it steady, and
request a device still photograph for cropped OCR. This sample starts a hub
and worker and reuses a separately running VLM service.

Refer to the [gesture-capture guide](https://nvidia.github.io/xr-ai/main/reference/gesture-capture.html)
for model provisioning, licenses, browser requirements, alignment, and behavior.

## Configure

Working directory: `agent-samples/gesture-capture/`.

Provision `models/hand_landmarker.task` using the guide. Set camera alignment
only after verifying photo/video framing:

```yaml
# yaml/gesture_capture_worker.yaml
aligned_photo: true
hold_s: 0.6
```

That YAML also selects model paths, gesture timing, tolerances, and the OCR
question. `yaml/models.json` selects the VLM endpoint and adapter.
`yaml/device_io_hub.yaml` configures the hub and web client.

## Run

```bash
uv --config-file ../../uv.toml sync
uv --config-file ../../uv.toml sync --project worker
```

Start the shared models separately, or use an existing VLM endpoint:

```bash
uv --config-file ../../uv.toml run \
  --project ../simple-vlm-example/model-servers simple_vlm_example_model_servers
```

Then launch the sample:

```bash
uv --config-file ../../uv.toml run gesture_capture
```

Open the hub's authenticated web URL, select **Live** camera mode, and form
the gesture. A browser and camera supporting native still photography are
required. The client displays the text reply on `gesture.ocr`.
