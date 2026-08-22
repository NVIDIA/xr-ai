<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Lab instrument monitoring

This sample writes durable monitoring output to files and serves a bounded live
event viewer while a separate foreground agent answers voice or typed queries.
See the [sample guide](../../docs/source/reference/lab-instrument-monitoring.md)
for architecture, output and privacy contracts, marker setup, configuration,
and eval instructions.

## Run

By default, the sample uses Nemotron Omni for foreground tool routing and Cosmos
for image inference. It reuses those services, STT, and Piper TTS; the sample
never starts or stops model services.

Start the model server stack in one terminal:

```bash
uv sync --project agent-samples/model-servers
uv run --project agent-samples/model-servers model_servers
```

Start Piper TTS in a second terminal:

```bash
uv sync --project services/piper-tts
uv run --project services/piper-tts piper_tts_server \
  --config services/piper-tts/piper_tts_server.yaml
```

Then start the sample in a third terminal:

```bash
uv sync --project agent-samples/lab-instrument-monitoring
uv sync --project agent-samples/lab-instrument-monitoring/worker
uv run --project agent-samples/lab-instrument-monitoring \
  lab_instrument_monitoring

# Allow direct event-viewer access from a trusted private network.
uv run --project agent-samples/lab-instrument-monitoring \
  lab_instrument_monitoring --expose-web-events
```
Connect with the authenticated URL and token printed by the hub. The live event
viewer is `http://127.0.0.1:8092`; expose it only on a trusted network.
