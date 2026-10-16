<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Procedure capture and replay

Record narrated demonstrations with spoken start and stop commands. The sample
saves sampled frames, visual captions, and narration as local evidence packets.
Replay mode provides spoken guidance and evidence checks for an approved local guide.

## Run

Run all commands from `agent-samples/procedure-capture-replay/`. Start the sample's model
servers and wait for them to report ready:

```bash
uv run --project model-servers procedure_capture_replay_model_servers
```

Then start the sample from the same terminal:

```bash
uv sync
uv run main.py --mode capture
```

Alternatively, use the installed entry point:

```bash
uv run procedure_capture_replay --mode capture
```

Open the authenticated web-client URL printed by the hub and connect. Enable
the microphone and use live video or on-demand image capture. Say **start
capture**, narrate the demonstration, then say **stop capture**. Repeat
these commands to record another task without reconnecting. The worker announces
"Capture started." and "Capture stopped.", and stays silent during narration
and finalization.

To choose where packets are saved, launch with `uv run main.py --mode capture --output-root PATH`.

To test replay with the included Numi Chamomile Lemon example:

```bash
uv run main.py --mode replay "Tea Making"
```

Use Numi Chamomile Lemon tea, boiling water, a five-minute steep, and a kettle
or thermometer that shows the current water temperature. For another locally
reviewed guide under `guides/`, replace `Tea Making` with its exact name or ID.
The guide must have `task.status: approved`. Replay uses the same model stack.
After connecting, say **Hey agent, let's start talking** to enroll your voice,
then **start replay** to begin and **stop replay** to cancel. Say **start replay**
again to begin a fresh run of the same command-line-selected guide.
Refer to the [replay reference](https://nvidia.github.io/xr-ai/latest/reference/procedure-capture-replay.html#replay-an-approved-guide)
for verification and step-advance controls. Capture and replay run separately.

Stop the sample with Ctrl+C. The shared models remain running; stop them with:

```bash
uv run --project model-servers procedure_capture_replay_model_servers --stop
```

## Configure

| File | Common settings |
|---|---|
| `yaml/worker.yaml` | Shared frame and speech detection settings; capture configuration path |
| `yaml/capture-agent-config.yaml` | Task output directory, sampling rate, and caption interval |
| `yaml/replay-agent-config.yaml` | Local guide directory and verification timeout |
| `yaml/models.json` | Client adapters and endpoints for batch STT, LLM, VLM, and acknowledgement TTS |
| `model-servers/yaml/<gpu-profile>.yaml` | Local model-server GPU placement and workload budgets |
| `yaml/voice_gate.yaml` | Silent, wake-word-free narration input |
| `yaml/voice_gate.replay.yaml` | Speaker enrollment, conversation controls, and speaker-STT endpoint |
| `yaml/device_io_hub.yaml` | Room, ports, and web client |

For example, increase the interval between captions in `yaml/capture-agent-config.yaml`:

```yaml
caption_interval_s: 8.0
```

Restart the sample after configuration changes. Refer to the
[sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/procedure-capture-replay.html#composition-and-configuration)
for recording boundaries, output files, prerequisites, and limitations.
Refer to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html)
for every checked-in field.
