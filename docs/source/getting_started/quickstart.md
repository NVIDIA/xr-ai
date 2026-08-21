<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Quickstart

## Set up with a coding agent

The fastest path: paste this to your agent and it does the rest, including
walking you through the choices below. See {doc}`skills` for how it works.

```{literalinclude} /_snippets/agent-setup-prompt.txt
:language: text
```

The rest of this page is the manual path.

Every sample follows the same pattern: **start the server, then connect a
client.** Once it is ready, any supported client — web browser, Android app,
iOS/visionOS app, or AR glasses — can join the session using the token printed
on startup.

## Model servers (shared AI services)

`model-servers` starts the shared inference services used across demos and exits
immediately — the services keep running in the background with weights hot.
Start this once before running `simple-vlm-example`,
`lab-instrument-monitoring`, `tea-making-sample`, or `xr-render-demo`, or
whenever you want to pre-warm models:

:::{important}
After updating, stop any existing model servers before starting this stack:

```bash
uv run model_servers --stop
```

Persisted vLLM processes or containers may otherwise keep serving the previous
models and images even though the checked-in configuration now selects
Nemotron-3 Nano Omni and Cosmos3 Nano Reasoner.
:::

```bash
cd agent-samples/model-servers
uv sync
uv run model_servers
```

GPU profiles are auto-detected (`dual_48G_ada`, `spark`, `96G_blackwell`). The
profiles are presets for common configurations; to run on a different GPU, refer to
{doc}`Running on other GPUs </getting_started/requirements>`.
Detection inventories every visible device. If `nvidia-smi` fails, returns
malformed data, or reports a topology without an existing model-server
configuration, startup stops with the detected per-GPU capacity instead of
assuming a fallback.
On first run each model downloads from HuggingFace (tens of GB; can take
tens of minutes). On subsequent runs the containers restart in under a minute.

Which servers start is a deployment profile selected with
`--models <name|path>`. The default starts Nemotron-3 Nano Omni (8108,
serving both LLM roles), Cosmos3 Nano Reasoner (8100), STT (8103), Piper TTS
(8105), and embeddings (8109); `vlm_llm_nim` serves the LLM and VLM as self-hosted NIM
containers (Nemotron-3-Nano and Cosmos3-Nano Reasoner; requires docker and
`NGC_API_KEY`). Starting a profile stops persisted servers outside it first
and aborts if they cannot be stopped, avoiding GPU overcommit.

```bash
uv run model_servers --models vlm_llm_nim
```

`HF_TOKEN` is required by default: without it the large first-run download
can stall indefinitely. Refer to the
{doc}`credentials guide </getting_started/credentials>` for how to set it, or
pass `--allow-anonymous` to run without one.

To stop all model servers when done:

```bash
uv run model_servers --stop
```

`--stop` stops every model-server port, so it takes no profile selection.

## Simple VLM example (vision Q&A over voice + text)

End-to-end voice + vision sample. Speak into the mic or type into the data
channel; both routes use the same VLM pipeline against the latest video frame.
Replies arrive as streaming Piper TTS audio plus a `vlm.response` text message.

Uses the text-output Reasoner from `nvidia/Cosmos3-Nano` by default. Refer to
{doc}`AI services </components/ai-services>` for runtime-selection details.

The sample always reuses model services and never starts or stops them. Its
fixed `yaml/models.json` expects Parakeet STT on port 8103, Cosmos3-Nano on
port 8100, and Piper TTS on port 8105. Start the repository defaults first
from the repository root:

```bash
uv run --project agent-samples/model-servers model_servers
```

The command may download model weights on its first run and requires the
credentials described in the {doc}`credentials guide
</getting_started/credentials>`. The model services remain running across
sample restarts.

### Step 1 — Start the server

```bash
cd agent-samples/simple-vlm-example
uv sync
uv run simple_vlm_example
```

Only the DeviceIOHub and worker start. Worker readiness probes all three reused
services and includes a short 1280x720 streaming VLM warmup.
The hub prints:

```
[hub]   LiveKit URL : wss://localhost:8080
[hub]   Room        : xr-room
[hub]   Token       : eyJ…
[hub]   Web client  : https://localhost:8080
```

This banner appears as soon as the hub itself is ready, while the model
services and worker are still starting. Clients can connect as soon as it
appears, but the agent answers queries only after the launcher prints its
`All processes ready` banner.

### Step 2 — Connect a client

Open `https://localhost:8080` in a browser. The samples ship with HTTPS on by
default (a self-signed certificate is generated on first run at
`~/.local/share/xr-ai/web-server.crt`), so you'll see a "Your connection is not
private" warning the first time — click **Advanced → Proceed** (Chrome or Edge) or
**Accept the Risk and Continue** (Firefox). Refer to the networking guide for
trusting the certificate permanently or running over plain HTTP instead.

Leave **Token URL** blank — the web client fetches a token from the server
automatically. Click **Connect**.

You are now live in the XR session. To test the agent:

- Type any question → sent verbatim to the VLM.
- Speak into your mic → speech is transcribed and sent as a query.

A successful round trip: your query appears in the log, the agent responds after
a moment, and you hear the reply through your speakers.

To use compatible services at different locations, edit their endpoints in
`agent-samples/simple-vlm-example/yaml/models.json`:

```json
{
  "endpoint": {"base_url": "https://your-vlm.example.com"},
  "deployment": {"ownership": "reused", "service": "vlm"}
}
```

The sample does not offer deployment profiles; the referenced services remain
operator-owned regardless of endpoint.

Each sample has its own `device_io_hub.yaml` controlling the hub; refer to
`services/device-io-hub/device_io_hub.yaml` for the full option list.

## Lab instrument monitoring (marker-associated readings + foreground voice)

This sample offers one on-demand background visual observation task per
participant while a separate generic tool-calling agent answers voice or typed
queries and controls that task. A separate QR and ArUco instrument monitor tracks readings,
speaks only discovered, changed, or long-missing device updates, and persists
10-second full-state snapshots. The sample writes monitor, instrument,
final pre-gate transcript, foreground-turn, and Relay JSONL files under `artifacts/`
and intentionally serves no sample-specific monitoring web UI. Its fixed model
configuration uses Cosmos for visual inference.

Start the shared `model-servers` stack, which includes Piper TTS:

```bash
uv run --project agent-samples/model-servers model_servers
```

Then start the sample in another terminal:

```bash
cd agent-samples/lab-instrument-monitoring
uv sync
uv sync --project worker
uv run lab_instrument_monitoring
```

Connect an existing glasses or platform client using the authenticated URL,
room, and token printed by the hub. The token and signaling routes remain
available on port 8080 together with the shared connection web client. Refer to
the {doc}`lab instrument architecture guide </reference/lab-instrument-monitoring>`
for reusable agent patterns, marker setup, output contracts, and adaptation
recipes.

## Tea-making guidance (voice + visual workflow)

This sample combines an interactive tea guide with optional background change,
transcript, and video observation. Nemotron-3 Nano Omni supplies both language
and visual inference. Records are written as JSON Lines under the sample's
`artifacts/` directory. A separate live event viewer presents selected runtime
events without replacing those durable records.

Start the shared model services, including Piper TTS, then launch the sample:

```bash
uv run --project agent-samples/model-servers model_servers

uv run --project agent-samples/tea-making-sample tea_making_sample
```

Open the XR-Media-Hub connection page at `https://localhost:8080`, accept the
self-signed certificate on first use, allow camera and microphone access, and
connect. The checked-in voice-gate YAML requires “Agent” or “Hey Agent.” Set
`voice_gate_yaml: voice_gate.always-on.yaml` in `yaml/tea_making_worker.yaml`
to dispatch every finalized utterance without a wake phrase.

Open `http://127.0.0.1:8092` on the XR-AI host for the live event viewer. To
view it directly from another trusted machine, add `--expose-web-events` and
use `http://<xr-host>:8092`; restrict that unauthenticated port to the trusted
development network.

The tea workflow changes steps only after an explicit user command. Visual
observations can satisfy the current step's evidence requirements, but never
advance the workflow silently. Refer to the
{doc}`tea-making architecture guide </reference/tea-making-sample>` for
reusable workflow patterns, background-agent contracts, backend integration,
and adaptation recipes.

## XR render demo (voice-driven sphere in CloudXR)

Speak to the web client and a sphere in the streamed scene tracks your voice —
radius follows loudness, colour and position follow spoken commands ("make it
red", "put it to my left", "where I'm looking"). Runs against a Quest 3 or Vision
Pro on the same LAN, or the IWER emulator built into the web client for desktop
dev.

Under the hood, the orchestrator launches the hub, CloudXR runtime, model
endpoints, typed capability processes, and the worker. The worker calls those
processes through Relay-managed native tools. The voice runtime runs quick-acks
and a Nemotron-30B agentic tool-calling loop over scene, tracking,
spatial math, vision, and video-memory tools. Refer to the xr-render-demo guide for the
full process map, agentic-loop details, and the XR session lifecycle.

**Requires `model-servers` to be running first** — the demo does not start its
own model services.

### Step 1 — Start model servers (once)

```bash
cd agent-samples/model-servers
uv sync && uv run model_servers
```

This exits immediately once all configured services are ready. Weights stay loaded in
the background.

### Step 2 — Start the demo

This demo has two extra host prerequisites beyond the shared
{doc}`Requirements <requirements>`:

- **Vulkan loader + headers** — the CloudXR compositor and LOVR render through
  Vulkan, so install them before running the demo: `sudo apt install libvulkan-dev`
- **npm 18+** on PATH — the orchestrator builds the web vendor bundle on first
  run (skipped on subsequent runs).

Start the shared model stack, which includes Piper TTS:

```bash
uv run --project agent-samples/model-servers model_servers
```

Then start XR Render:

```bash
cd agent-samples/xr-render-demo
uv sync
uv run xr_render_demo
```

On first run the orchestrator automatically downloads the pinned LOVR version to
`deps/lovr/` inside the repository and builds the web vendor bundle (requires npm
and network access). Both steps are skipped on subsequent runs.

```{note}
On **DGX Spark** (aarch64), LOVR does not publish a prebuilt aarch64 Linux
binary, so the auto-download is not available: build LOVR from source and export
`LOVR_BIN`. Refer to the troubleshooting guide.
```

To use a custom LOVR build:

```bash
export LOVR_BIN=/path/to/your/lovr   # or set lovr_bin: in scene/scene_service.yaml
uv run xr_render_demo
```

**GPU pinning** for the XR side is controlled by `gpu_index` in
`agent-samples/xr-render-demo/yaml/cloudxr_runtime.yaml`. cloudxr-runtime applies
the pin to its own process and writes the selectors into `cloudxr.env`;
the scene process and LOVR inherit from that file. Refer to the xr-render-demo guide
for full details.

To stop the model servers when done:

```bash
cd agent-samples/model-servers
uv run model_servers --stop
```

XR Render uses the fixed reuse-only endpoints in `yaml/models.json`; it does
not select or own model deployment profiles.

## Hub only (standalone)

```bash
cd services/device-io-hub
uv sync
uv run device_io_hub
```

Useful for development or when running an agent in a separate terminal. The
DeviceIOHub auto-discovers `services/device-io-hub/device_io_hub.yaml`.
