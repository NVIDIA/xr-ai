<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Quickstart

## Overview

Before adapting XR AI to your own task, try a complete interaction: connect a
device, ask about its camera view, and hear the answer. A working sample gives
you a baseline for developing your application and checking how changes affect
the user's experience.

<!-- Compatibility anchor for the earlier setup heading. -->
<a id="set-up-with-a-coding-agent"></a>

## Run with a coding agent

Paste this to your coding agent to run the Simple VLM browser demo:

```{literalinclude} /_snippets/agent-demo-prompt.txt
:language: text
```

Refer to {doc}`skills` for `run-demo` or development-only preparation with `setup`.

The remainder of this quickstart is the manual path.

Before launching a voice sample, refer to {ref}`onnx-runtime-telemetry`.
If outbound ONNX Runtime telemetry is unwanted, export
`ORT_DISABLE_TELEMETRY=1` in the launch environment before starting the process.

Start only the local model services the sample needs, wait for their launcher
to report readiness and return, then start the sample from the same terminal.
For model services hosted externally, configure and validate those endpoints
before starting the sample; a mixed setup can use both local and external
services. Once the sample is ready, any supported client — web browser, Android
app, iOS/visionOS app, or AR glasses — can join the session using the token
printed on startup. Each procedure below starts with a `cd` from the repository
root; keep running that procedure's commands from the selected sample directory.

Each interactive procedure includes one concrete spoken request that exercises
the sample's main path. In the shipped voice-gated samples, first say
“Hey agent, let's start talking” and wait for “I'm listening to you now.”
Then speak questions without a wake phrase. After the first interaction succeeds,
you can ask “What can you do?” to explore further. Typed requests bypass the
voice gate. Treat the reply as a conversational introduction rather than an
exhaustive capability reference.

(model-servers-shared-ai-services)=
## Model servers for each sample

Complete {ref}`docker-host-setup` before the first local launch, including on
DGX Spark. Configure `HF_TOKEN` using {doc}`credentials`.

Each sample has a `model-servers/` convenience launcher for just the inference
services it needs. The commands below start that stack, wait for healthy
servers, and return to the same terminal. Models stay warm across agent
restarts. Compatible independently deployed endpoints can also serve the
sample; refer to {doc}`/guides/deploying-with-nim`.

GPU profiles are auto-detected (`dual_48G_ada`, `spark`, `96G_blackwell`). Use
`--gpu-profile NAME` to select an explicitly tuned sample profile. Unsupported
hardware or failed GPU inventory stops startup; refer to
{doc}`requirements` for custom profiles. First launch downloads missing weights
and can take tens of minutes.

Before switching samples, updating the checkout, or editing server settings,
stop the current agent and its model stack with that sample's model command
and `--stop`. Use the same explicit GPU profile, then start the destination
stack and agent. This releases unused models and applies the current settings
without keeping track of previous launches. Refer to
{doc}`/guides/customizing-model-servers` for configuration and the full flow.

## Conversation example

Start here for a new voice-and-vision application. The sample combines
conversation and questions about the current view with opt-in visual alerts
and note-taking. Refer to {doc}`/reference/conversation-example` for its
behavior, configuration, and architecture.

From the repository root, enter the sample directory and start its local models:

```bash
cd agent-samples/conversation-example
uv run --project model-servers conversation_example_model_servers
```

Wait for the model launcher to report readiness and return, then start the
sample from the same terminal:

```bash
uv sync
uv run conversation_example
```

Open the authenticated web-client URL printed by DeviceIOHub, allow microphone
and camera access, and connect. After the agent reports ready, say
“Hey agent, let's start talking,” then ask “What is in front of me?” while
pointing the camera at an object. Typed questions also work without opening a
voice conversation. Refer to {doc}`clients` for other client platforms.

Press Ctrl+C to stop the sample. Its local model services remain running;
stop them separately from the same sample directory:

```bash
uv run --project model-servers conversation_example_model_servers --stop
```

## Simple VLM example (vision Q&A over voice + text)

```{warning}
Simple VLM is deprecated and remains available as a small browser demo and
legacy reference. For a new voice-and-vision application, start with
{doc}`/reference/conversation-example`, which is the recommended foundation.
The `run-demo` workflow still uses Simple VLM to provide a quick end-to-end
demo.
```

Legacy end-to-end voice + vision demo. Speak into the mic or type into the data
channel; both routes use the same VLM pipeline against the latest video frame.
Replies arrive as streaming Pocket TTS audio plus a `vlm.response` text message.

Uses the text-output Reasoner from `nvidia/Cosmos3-Nano` by default. Refer to
{doc}`AI services </components/ai-services>` for runtime-selection details.

The sample always reuses model services and never starts or stops them. Its
fixed `yaml/models.json` configures Cosmos3-Nano on port 8100 and Pocket TTS on
port 8105; `yaml/voice_gate.yaml` configures speaker ASR on port 8102.
From the sample directory, start the
repository defaults first:

```bash
cd agent-samples/simple-vlm-example
uv run --project model-servers simple_vlm_example_model_servers
```

The command may download model weights on its first run. Refer to the
{doc}`credentials guide </getting_started/credentials>` for the required
credentials. The model services remain running across sample restarts.

### Step 1 — Start the server

```bash
uv sync
uv run simple_vlm_example
```

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

Only DeviceIOHub and the worker start. For the worker's startup warmup, refer
to {doc}`/reference/simple-vlm-example`. For readiness and required deployment
order, refer to {ref}`consumer-model-readiness`.
The hub prints:

```
[hub]   LiveKit URL : wss://localhost:8080
[hub]   Room        : xr-room
[hub]   Token       : eyJ…
[hub]   Web client  : https://localhost:8080
```

This banner appears as soon as the hub itself is ready, while the worker
is still starting. Clients can connect as soon as it
appears, but the agent answers queries only after the launcher prints its
`All processes ready` banner.

### Step 2 — Connect a client

Open `https://localhost:8080` in a browser. The samples ship with HTTPS on by
default (a development root CA and signed server leaf are generated on first
run under `~/.local/share/xr-ai/`), so you'll see a "Your connection is not
private" warning the first time — click **Advanced → Proceed** (Chrome or Edge) or
**Accept the Risk and Continue** (Firefox). Refer to the networking guide for
trusting the certificate permanently or running over plain HTTP instead.

Leave **Token URL** blank — the web client fetches a token from the server
automatically. Click **Connect**, then set **Camera Mode** to **Live video**
and allow camera access. For a spoken interaction, click **Start Microphone** and allow
microphone access.

### Step 3 — Meet the agent

First say “Hey agent, let's start talking” and wait for
“I'm listening to you now.”

Try this first spoken interaction:

```text
You: What am I looking at?
Agent: [Answers from the current camera frame.]
```

- Type any question → sent verbatim to the VLM.
- Speak into your mic → speech is transcribed and sent as a query.

A successful round trip: your query appears in the log, the agent responds after
a moment, and you hear the reply through your speakers.

To use compatible services at different locations, edit their endpoints in
`yaml/models.json`:

```json
{
  "endpoint": {"base_url": "https://your-vlm.example.com"}
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

Start the sample's `model-servers/` stack, which includes Pocket TTS:

```bash
cd agent-samples/lab-instrument-monitoring
uv run --project model-servers lab_instrument_monitoring_model_servers
```

Wait for the model launcher to report readiness and return, then start the
sample from the same terminal:

```bash
uv sync
uv run lab_instrument_monitoring
```

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

Connect an existing glasses or platform client using the authenticated URL,
room, and token printed by the hub. The token and signaling routes remain
available on port 8080 together with the shared connection web client. Refer to
the {doc}`lab instrument architecture guide </reference/lab-instrument-monitoring>`
for reusable agent patterns, marker setup, output contracts, and adaptation
recipes.

### Try the lab agent

Point the camera at instruments carrying the configured markers. Say
“Hey agent, let's start talking” and wait for “I'm listening to you now.”
Then try this spoken interaction:

```text
You: Read the instruments.
Agent: [Reports the visible marker-associated readings or their availability.]
```

Inspect the JSONL artifacts while trying additional interactions.

## Tea-making guidance (voice + visual workflow)

This sample combines an interactive tea guide with optional background change,
transcript, and video observation. Nemotron-3 Nano Omni supplies both language
and visual inference. Records are written as JSON Lines under the sample's
`artifacts/` directory. A separate live event viewer presents selected runtime
events without replacing those durable records.

Start the sample model services, including Pocket TTS:

```bash
cd agent-samples/tea-making-sample
uv run --project model-servers tea_making_sample_model_servers
```

Wait for the model launcher to report readiness and return, then launch the
sample from the same terminal:

```bash
uv sync
uv run tea_making_sample
```

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

Open the DeviceIOHub connection page at `https://localhost:8080`, accept the
development certificate on first use, allow camera and microphone access, and
connect. Say “Hey agent, let's start talking” to open the conversation, then
speak naturally; say “Hey agent, let's stop talking” to close it.
The Tea sample uses this `conversation` workflow. Set `mode: hybrid` with a wake
phrase to also ask addressed one-off questions while closed, or `always_on`
for an explicitly ungated microphone.

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

### Try the tea agent

Try this first spoken interaction:

```text
You: Hey Agent, help me make tea.
Agent: [Starts the tea guide and presents the first step.]
```

## XR render demo (voice-driven sphere in CloudXR)

Speak to the web client and a sphere in the streamed scene tracks your voice —
radius follows loudness, colour and position follow spoken commands ("make it
red", "put it to my left", "where I'm looking"). Runs against a Quest 3 or Vision
Pro on the same LAN, or the IWER emulator built into the web client for desktop
dev.

Under the hood, the orchestrator launches the hub, CloudXR runtime, typed
capability processes, and the worker alongside the reused model endpoints. The
worker calls those processes through Relay-managed native tools. The render
agent runs a Nemotron-3-Nano-Omni-30B-A3B-Reasoning tool-calling loop over
scene, tracking, spatial math, vision, and video-memory tools. Refer to the
{doc}`xr-render-demo reference </reference/xr-render-demo>` for the full process
map, agentic-loop details, and XR session lifecycle.

**Requires `model-servers` to be running first** — the demo does not start its
own model services.

### Step 1 — Start model servers (once)

```bash
cd agent-samples/xr-render-demo
uv run --project model-servers xr_render_demo_model_servers
```

This exits immediately once all configured services are ready. Weights stay
loaded in the background.

### Step 2 — Start the demo

Refer to the shared {doc}`Requirements <requirements>` first. This demo has two
additional host prerequisites:

- **Vulkan loader + headers** — the CloudXR compositor and LOVR render through
  Vulkan, so install them before running the demo: `sudo apt install libvulkan-dev`
- **Node.js 20.19.0+ with npm** on PATH for WebRTC profiles — the orchestrator
  builds the web vendor bundle on first run (skipped for native profiles and
  on subsequent runs).

Start XR Render:

```bash
uv sync
uv run xr_render_demo
```

Alternatively, run the source file directly after synchronization:

```bash
uv run main.py
```

On first run the orchestrator automatically downloads the pinned LOVR version to
`deps/lovr/` inside the repository. For WebRTC profiles it also builds the web
vendor bundle, which requires npm and network access. Existing artifacts are
reused on subsequent runs.

```{note}
On **DGX Spark** (aarch64), LOVR does not publish a prebuilt aarch64 Linux
binary, so the auto-download is not available: build LOVR from source and export
`LOVR_BIN`. Refer to {doc}`/guides/troubleshooting` for build instructions.
```

To use a custom LOVR build:

```bash
export LOVR_BIN=/path/to/your/lovr   # or set lovr_bin: in scene/scene_service.yaml
uv run xr_render_demo
```

**GPU pinning** for the XR side is controlled by `gpu_index` in
`yaml/cloudxr_runtime.yaml`. cloudxr-runtime applies
the pin to its own process and writes the selectors into `cloudxr.env`;
the scene process and LOVR inherit from that file. Refer to the
{doc}`xr-render-demo reference </reference/xr-render-demo>` for full details.

### Step 3 — Connect and meet the agent

Open the authenticated web-client URL printed by DeviceIOHub. Accept the
port-8080 development certificate on first use, leave **Token URL** blank, and
click **Connect**. Wait for the agent status to report ready, then click **Start
Microphone**. In the **XR Stream** section, follow the **accept the CloudXR
cert** link to accept the second development certificate, return to the client,
and click **Launch XR**.

After saying “Hey agent, let's start talking,” ordinary finalized speech is
dispatched without a wake phrase; stop commands interrupt the response without
closing the conversation. Try this first spoken interaction:

```text
You: Make the sphere red.
Agent: [Changes the sphere's color or explains why it could not.]
```

Speak naturally: the agent accepts questions, commands, and follow-up requests.
The scene draws two shapes, boxes and spheres; ask for either by name. Refer to
{doc}`Connecting clients <clients>` for headset and native-client setup.

To stop the model servers when done:

```bash
uv run --project model-servers xr_render_demo_model_servers --stop
```

XR Render uses the configured endpoints in `yaml/models.json`; it does
not select or own model deployment profiles.

## Hub only (standalone)

```bash
cd services/device-io-hub
uv sync
uv run device_io_hub --config device_io_hub.yaml
```

Useful for development or when running an agent in a separate terminal. The
explicit configuration is the repository reference copy with every field
documented beside its value.
