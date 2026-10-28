<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Gesture capture

Frame a rectangle with two opposing thumb/index L shapes (tips need not touch),
or pinch and draw a near-closed circle, to ask a VLM about the selected camera region. The sample
starts a hub and worker and reuses a separately running VLM service. It crops
streamed video by default; optional aligned-photo mode requests a separate
higher-resolution photograph.

Refer to the [gesture-capture guide](https://nvidia.github.io/xr-ai/main/reference/gesture-capture.html)
for model provisioning, licenses, browser requirements, alignment, and behavior.

## Configure

Working directory: `agent-samples/gesture-capture/`.

Before starting the sample, provision the required hand model. Run these commands
from `agent-samples/gesture-capture/`:

```bash
mkdir -p models
curl --fail --location \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
  --output models/hand_landmarker.task
printf '%s  %s\n' \
  fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1 \
  models/hand_landmarker.task | sha256sum --check
```

The model stays local and is ignored by Git. Refer to the guide for its licensing
and backend requirements. Default video-frame mode needs no photo alignment
configuration. For example, adjust the hold time:

```yaml
# yaml/gesture_capture_worker.yaml
hold_s: 0.6
```

That YAML also selects model paths, gesture timing, tolerances, and the general VLM
question. For a flaky-gesture reproduction, opt into `gesture_diagnostics: true`
in that YAML; bounded numeric summaries appear in the terminal and existing
worker log (see the guide for interpretation). `yaml/models.json` selects the VLM endpoint and adapter.
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
the gesture. A quiet progress hum becomes a chime at recognition; permit browser
audio playback to hear it. The client displays the text reply on `gesture.response`.
Wait for the answer, then relax the box pose or open the pinch to rearm.

To enable higher-resolution stills, verify matching photo and video framing,
then set `aligned_photo: true` in `yaml/gesture_capture_worker.yaml`. That mode
requires a browser and camera supporting native still photography.
