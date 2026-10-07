<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SOP sample

Capture narrated demonstrations while a participant's live camera is on, or
replay one approved local SOP with spoken guidance and step verification.
Capture and replay are separate, mutually exclusive launch modes.

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

To replay a reviewed guide instead, use its exact task name or ID:

```bash
uv run main.py --replay "Guide Name"
```

Place the local `*.guide.yaml` file under `guides/` and review it before setting
`task.status: approved`. Connect with microphone and camera access. The agent
starts step one, announces verified completion, and waits for “next.” Finishing
the final step with “next” resets the guide for another run. Refer to the
[replay reference](https://nvidia.github.io/xr-ai/latest/reference/sop-sample.html#replay-an-approved-guide)
for the schema, controls, and verification limits.

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
| `yaml/replay.yaml` | Local guide directory, replay models, voice gate, and vision timeouts |
| `yaml/models.replay.json` | Replay LLM, VLM, STT, and TTS endpoints |
| `yaml/voice_gate.replay.yaml` | Wake-word-free guide questions and controls |

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
