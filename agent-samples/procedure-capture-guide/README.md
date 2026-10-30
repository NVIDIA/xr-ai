<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Procedure capture and guidance

Record narrated demonstrations with spoken start and stop commands. The sample
saves sampled frames, visual captions, and narration as local evidence packets.
Guide mode provides spoken guidance and evidence checks for an approved local guide.

To generate a guide from a captured packet, start with
`guide-generation-skills/SKILL.md`, which coordinates drafting, evaluation, and
revision. Refer to the
[guide-generation reference](https://nvidia.github.io/xr-ai/latest/reference/procedure-capture-guide.html#generate-a-draft-guide)
for the authoring request, validation, and human review workflow.

## Directory map

```text
main.py                         Sample launcher
worker/                         Capture and guide agents
yaml/                           Sample configuration
model-servers/                  Shared local model launcher
guide-generation-skills/        Coding-agent instructions and references
evaluation/
  capture/                      Caption evaluator and cases
  guide/                        Packet evaluator, shared harness, authoring probes
example-guides/tea-making/
  guide.yaml                    Maintained example procedure
  evaluation/                   Tea runners, fixtures/, and scenarios/
```

The skills invoke the checkout's evaluation tools; they do not contain a
separate implementation. Refer to the
[evaluation reference](https://nvidia.github.io/xr-ai/latest/reference/procedure-capture-guide.html#local-model-guide-evaluation)
for commands, coverage, and result interpretation.

## Run

Run all commands from `agent-samples/procedure-capture-guide/`. Start the sample's model
servers and wait for them to report ready:

```bash
uv run --project model-servers procedure_capture_guide_model_servers
```

Then start the sample from the same terminal:

```bash
uv sync
uv run main.py
```

Capture is the default mode; `--mode capture` selects it explicitly.
Alternatively, use the installed entry point:

```bash
uv run procedure_capture_guide --mode capture
```

Open the authenticated web-client URL printed by the hub and connect. Enable
the microphone and use live video or on-demand image capture. Say **start
capture**, narrate the demonstration, then say **stop capture**. Repeat
these commands to record another task without reconnecting. The worker announces
"Capture started." and "Capture stopped.", and stays silent during narration
and finalization. On connection, the worker explains these commands once.
Other speech while idle does not trigger a reply.

Each capture is saved inside this sample at
`artifacts/sessions/<session-id>/` (Git-ignored): `frames/` holds the JPEG
snapshots, and `captions.jsonl`, `transcript.jsonl`, `summary.md`, and
`packet.json` sit beside it. To choose another location, launch with
`uv run main.py --mode capture --output-root PATH`; packets then go under
`PATH/sessions/<session-id>/`.

To test guide mode with the included chamomile tea example:

```bash
uv run main.py --mode guide example-guides/tea-making/guide.yaml
```

Use tea labeled chamomile, boiling water, a five-minute steep, and a kettle
or thermometer that shows the current water temperature. For another locally
reviewed guide, pass its YAML file path, such as `assemble-toy-guide/guide.yaml`,
or a directory containing exactly one top-level YAML file.
Relative paths resolve from this sample directory; absolute paths are also
supported. Quote paths containing spaces. Generated guides and their review
artifacts stay in local `<name>-guide/` directories; checked-in examples live
under `example-guides/`.
The guide must have `task.status: approved`. Guide mode uses the same model servers;
say **start guide** to begin and **stop guide** to cancel. Say **start guide**
again to begin a fresh run of the same command-line-selected guide.
Refer to the [guide reference](https://nvidia.github.io/xr-ai/latest/reference/procedure-capture-guide.html#follow-an-approved-guide)
for verification and step-advance controls. Capture mode and guide mode run separately.

To also record the whole participant session in either mode, independently of
spoken capture or guide commands, use the opt-in flag below. These video
recordings are separate from the capture packets above and are saved under
`~/.local/share/xr-ai/captures/procedure-capture-guide/`; without the flag,
nothing is written there.

```bash
uv run main.py --mode capture --session-recording
uv run main.py --mode guide example-guides/tea-making/guide.yaml --session-recording
```

In capture mode, this flag can also be combined with `--output-root PATH`. Refer to the
[session recording guide](https://nvidia.github.io/xr-ai/latest/reference/procedure-capture-guide.html#session-recording)
for output locations and encoding requirements.

Stop the sample with Ctrl+C. The shared models remain running; stop them with:

```bash
uv run --project model-servers procedure_capture_guide_model_servers --stop
```

## Configure

| File | Common settings |
|---|---|
| `yaml/worker.yaml` | Shared frame and speech detection settings; capture configuration path |
| `yaml/capture-agent-config.yaml` | Task output directory, sampling rate, and caption interval |
| `yaml/guide-agent-config.yaml` | Visual verification timeout |
| `yaml/media_capture.yaml` | Opt-in session recording output and rendering profile |
| `yaml/models.json` | Client adapters and endpoints for batch STT, LLM, VLM, and acknowledgement TTS |
| `model-servers/yaml/<gpu-profile>.yaml` | Local model-server GPU placement and workload budgets |
| `yaml/voice_gate.yaml` | Shared wake-word-free speech input for capture and guide |
| `yaml/device_io_hub.yaml` | Room, ports, and web client |

For example, increase the interval between captions in `yaml/capture-agent-config.yaml`:

```yaml
caption_interval_s: 8.0
```

Restart the sample after configuration changes. Refer to the
[sample configuration guide](https://nvidia.github.io/xr-ai/latest/reference/procedure-capture-guide.html#composition-and-configuration)
for recording boundaries, output files, prerequisites, and limitations.
Refer to the generated
[configuration reference](https://nvidia.github.io/xr-ai/latest/reference/configuration.html)
for every checked-in field.
