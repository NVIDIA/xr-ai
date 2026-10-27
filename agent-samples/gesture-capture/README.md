<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Gesture capture

Hold a box with both hands' index fingers and thumbs, or pinch and draw a circle
then release, to ask the VLM about that section of the camera view. Both gestures
request a separate still photograph and query its selected crop. This sample starts a hub
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

That YAML also selects model paths, gesture timing, tolerances, and the general VLM
question. `yaml/models.json` selects the VLM endpoint and adapter.
`yaml/device_io_hub.yaml` configures the hub and web client. Refer to the
[sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/gesture-capture.html#settings-and-output)
and generated [configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html).

## Run

Run all commands from `agent-samples/gesture-capture/`.

```bash
uv --config-file ../../uv.toml sync
uv --config-file ../../uv.toml sync --project worker
```

Start the shared models separately, or use an existing VLM endpoint:

```bash
uv run --project ../simple-vlm-example/model-servers simple_vlm_example_model_servers
```

Wait for the model launcher to report ready and return. Then launch the sample
from the same terminal:

```bash
uv run gesture_capture
```

Alternatively, after synchronization:

```bash
uv run main.py
```

Open the hub's authenticated web URL, select **Live** camera mode, and form
the gesture. A browser and camera supporting native still photography are
required. The client displays the text reply on `gesture.response`.
