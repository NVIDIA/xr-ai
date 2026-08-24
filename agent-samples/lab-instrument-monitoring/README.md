<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Lab instrument monitoring

This sample combines foreground voice and text assistance with participant-
controlled background observation. A separate QR and ArUco monitor associates
camera readings with instruments, reports meaningful changes, and records
durable participant-scoped output.

Nemotron-3 Nano Omni handles foreground tool routing, while Cosmos3 Nano
Reasoner handles image inference. The sample launches its hub, worker, and
application-specific processes, but reuses STT, TTS, LLM, and VLM endpoints
from the shared model stack.

## Configure

Edit the sample-owned files before starting the stack; the launcher reads them
automatically:

| File | Common changes |
|---|---|
| `yaml/lab_instrument_monitoring_worker.yaml` | Monitor cadence, lost-device timeout, frame freshness, VAD, artifacts, and event-viewer port and history |
| `yaml/device_map.yaml` | QR or ArUco identifiers and their instrument names |
| `yaml/voice_gate.yaml` | Wake phrases, listening chime, and follow-up window |
| `yaml/models.json` | Reused model adapters and endpoint addresses |
| `yaml/device_io_hub.yaml` | Room, ports, web client, and network behavior |

For example, change the two monitoring intervals without changing worker code:

```yaml
monitor_interval_s: 2.0
instrument_monitor_interval_s: 2.0
```

Restart the sample after an edit. Use `--expose-web-events` to change the
event-viewer bind address; that option overrides the YAML host. Refer to the
[sample configuration guide](../../docs/source/reference/lab-instrument-monitoring.md#configuration)
for how the files fit together and to the generated
[configuration reference](../../docs/source/reference/configuration.rst) for
every field and default.

## Run

Run all commands from `agent-samples/lab-instrument-monitoring/`. Start the
shared models first:

```bash
uv run --project ../model-servers model_servers
```

Wait for the launcher to report that all processes are ready and return. Then
start the sample from the same terminal:

```bash
uv sync
uv sync --project worker
uv run lab_instrument_monitoring
```

To make the unauthenticated event viewer reachable from a trusted private
network, use this alternative sample command:

```bash
uv run lab_instrument_monitoring --expose-web-events
```

Connect with the authenticated URL and token printed by DeviceIOHub. Durable
JSONL output is written below `artifacts/`, and the bounded live event viewer
is available at `http://127.0.0.1:8092` by default.

The shared models remain running after the sample stops. From this directory,
stop them with `uv run --project ../model-servers model_servers --stop`.

Refer to the
[sample reference](../../docs/source/reference/lab-instrument-monitoring.md)
for architecture, privacy and output contracts, marker setup, configuration,
and evaluation instructions.
