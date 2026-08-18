<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Quickstart

Every sample follows the same pattern: **start the server, then connect a
client.** Once it is ready, any supported client — web browser, Android app,
iOS/visionOS app, or AR glasses — can join the session using the token printed
on startup.

## Model servers (shared AI services)

`model-servers` starts the shared inference services used across demos and exits
immediately — the services keep running in the background with weights hot.
Start this once before running `xr-render-demo`, or whenever you want to
pre-warm models:

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

GPU profiles are auto-detected (`dual_48G_ada`, `spark`, `96G_blackwell`). These
are presets for common configurations; to run on a different GPU, refer to
{doc}`Running on other GPUs </getting_started/requirements>`.
On first run each model downloads from HuggingFace (tens of GB; can take
tens of minutes). On subsequent runs the containers restart in under a minute.

Which servers start is a deployment profile selected with
`--models <name|path>`. The default starts Nemotron-3 Nano Omni (8108,
serving both LLM roles), Cosmos3 Nano Reasoner (8100), STT (8103), and
embeddings (8109); `vlm_llm_nim` serves the LLM and VLM as self-hosted NIM
containers (requires docker + `NGC_API_KEY`); `vlm_speech_nim` serves
speech from Riva NIM containers. Starting a profile stops persisted servers
outside it first and aborts if they cannot be stopped, avoiding GPU
overcommit.

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

There are two ways to run it:

**Standalone** (~23 GB VRAM) — starts its own VLM and STT, owns them for the
session, and stops them when you exit:

```bash
cd agent-samples/simple-vlm-example
uv sync
uv run simple_vlm_example
```

On the very first run weights download from HuggingFace (tens of GB; can take
several minutes). `HF_TOKEN` is required by default; pass `--allow-anonymous`
to run without one (refer to the
{doc}`credentials guide </getting_started/credentials>`).

**With model-servers pre-running** — start `model_servers` to pre-warm VLM
(port 8100) and STT (port 8103). The demo detects and reuses them.
When you exit, those services keep running.

### Step 1 — Start the server

```bash
uv run simple_vlm_example
```

The XR-Media-Hub, VLM, STT, and TTS start together (or reuse running services).
When ready the hub prints:

```
[hub]   LiveKit URL : wss://0.0.0.0:8080
[hub]   Room        : xr-room
[hub]   Token       : eyJ…
[hub]   Web client  : https://localhost:8080
```

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

**Local model** — override the model weights or GPU settings by editing
`vlm_server.yaml` in the sample directory.

To use Cosmos-Reason1 instead, set `model: nvidia/Cosmos-Reason1-7B` in that
file and select `cosmos_vlm` as the VLM adapter preset in `models.local.json`.

**Remote model** — copy `yaml/models.hosted.json`, point its VLM endpoint at
your server, and select it in the worker config:

```json
{
  "endpoint": {"base_url": "https://your-remote-vlm.example.com"},
  "deployment": {"ownership": "external"}
}
```

```yaml
# yaml/simple_vlm_example_worker.yaml
models_config: models.remote.json
```

The external deployment declaration makes the orchestrator skip the local VLM
process automatically.

**Hosted NVIDIA NIM** — run the VLM on hosted NIM
([build.nvidia.com](https://build.nvidia.com)) instead of locally (STT/TTS stay
local) by setting **one key** in `simple_vlm_example_worker.yaml`:

```yaml
models_config: models.hosted.json
```

The same profile configures the worker and makes the orchestrator skip the
local VLM server. Pick the hosted model id in `models.hosted.json` and provide
an `NGC_API_KEY` as an
**environment variable** (or save it once via the launcher credential prompt) —
it is not stored in YAML; the overlay only names the env var via
`api_key_env: NGC_API_KEY`. Refer to the credentials and AI-services guides for
full details (and self-hosted NIM containers).

Each sample has its own `xr_media_hub.yaml` controlling the hub; refer to
`services/xr-media-hub/xr_media_hub.yaml` for the full option list.

## Lab instrument monitoring (marker-associated readings + foreground voice)

This sample offers one on-demand background visual observation task per
participant while a separate generic tool-calling agent answers voice or typed
queries and controls that task. A separate QR and ArUco instrument monitor tracks readings,
speaks only discovered, changed, or long-missing device updates, and persists
10-second full-state snapshots. The sample writes monitor, instrument,
final pre-gate transcript, foreground-turn, and Relay JSONL files under `artifacts/`
and intentionally serves no sample-specific monitoring web UI.

Start `model-servers`, then run:

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

**Hosted NVIDIA NIM** — run the LLMs and VLM on hosted NIM
([build.nvidia.com](https://build.nvidia.com)) instead of local vLLM (STT/TTS
stay local) by setting **one key** in `xr_render_demo_worker.yaml`:

```yaml
models_config: models.hosted.json     # default is models.local.json
```

The profile is consumed by both the worker and orchestrator, so there are
no `main.py`
edits, and the stack owns its own speech servers, so the model-servers stack
isn't needed at all. Provide an `NGC_API_KEY` as an **environment variable** (or via the
launcher credential prompt, not in YAML). Refer to the AI-services guide.

## Hub only (standalone)

```bash
cd services/xr-media-hub
uv sync
uv run xr_media_hub
```

Useful for development or when running an agent in a separate terminal. The
XR-Media-Hub auto-discovers `services/xr-media-hub/xr_media_hub.yaml`.
