<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP sample

Capture narrated demonstrations automatically while a participant's live camera
is on. The sample saves audio, video, sampled frames, captions, and narration.
It does not answer questions or run guides. A separate coding-agent skill turns
saved captures into draft guides for human review. Refer to
[guide generation](https://nvidia.github.io/xr-ai/latest/reference/sop-sample.html#generate-a-draft-guide)
for the prompt, validation command, and approval workflow.

## Run

Run all commands from `agent-samples/sop-sample/`. Start the shared model
servers and wait for them to report ready:

```bash
uv run --project ../../model-server-samples/model-servers model_servers
```

Then start capture from the same terminal:

```bash
uv sync
uv run main.py --capture
```

Alternatively, use the installed entry point:

```bash
uv run sop_sample --capture
```

Open the authenticated web-client URL printed by the hub, connect, enable the
microphone, and turn on **live video**. Turn the camera off to finalize the
recording; turn it on again for another recording. No spoken commands are
required. Disconnecting also finalizes the recording.

Stop the sample with Ctrl+C. The shared models remain running; stop them with:

```bash
uv run --project ../../model-server-samples/model-servers model_servers --stop
```

## Configure

| File | Common settings |
|---|---|
| `yaml/worker.yaml` | SOP output directory, frame and caption intervals, speech detection |
| `yaml/media_capture.yaml` | Shared media output directory, encoding, retention |
| `yaml/models.json` | STT and VLM endpoints; the voice runtime's unused TTS adapter |
| `yaml/voice_gate.yaml` | Silent, wake-word-free narration input |
| `yaml/device_io_hub.yaml` | Room, ports, and web client |

For example, increase the interval between captions in `yaml/worker.yaml`:

```yaml
caption_interval_s: 8.0
```

Restart the sample after configuration changes. Refer to the
[sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/sop-sample.html#composition-and-configuration)
for recording boundaries, output files, prerequisites, and limitations.
Refer to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html)
for every checked-in field.
