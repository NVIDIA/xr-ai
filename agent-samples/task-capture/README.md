<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Task capture

Record narrated demonstrations with spoken start and stop commands. The sample
saves sampled frames, visual captions, and narration as local evidence packets.

## Run

Run all commands from `agent-samples/task-capture/`. Start the shared model
servers and wait for them to report ready:

```bash
uv run --project ../../model-server-samples/model-servers model_servers
```

Then start the sample from the same terminal:

```bash
uv sync
uv run main.py
```

Alternatively, use the installed entry point:

```bash
uv run task_capture
```

Open the authenticated web-client URL printed by the hub and connect. Enable
the microphone and use live video or on-demand image capture. Say **start
capture**, narrate the demonstration, then say **stop capture**. Repeat
these commands to record another task without reconnecting. The worker announces
"Capture started." and "Capture stopped.", and stays silent during narration
and finalization.

To choose where packets are saved, launch with `uv run main.py --output-root PATH`.

Stop the sample with Ctrl+C. The shared models remain running; stop them with:

```bash
uv run --project ../../model-server-samples/model-servers model_servers --stop
```

## Configure

| File | Common settings |
|---|---|
| `yaml/worker.yaml` | Task output directory, frame and caption intervals, speech detection |
| `yaml/models.json` | STT, VLM, and recording-acknowledgement TTS endpoints |
| `yaml/voice_gate.yaml` | Silent, wake-word-free narration input |
| `yaml/device_io_hub.yaml` | Room, ports, and web client |

For example, increase the interval between captions in `yaml/worker.yaml`:

```yaml
caption_interval_s: 8.0
```

Restart the sample after configuration changes. Refer to the
[sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/task-capture.html#composition-and-configuration)
for recording boundaries, output files, prerequisites, and limitations.
Refer to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html)
for every checked-in field.
