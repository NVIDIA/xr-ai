<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- TODO: hero image -->

# xr-ai

Agentic AI for XR — a sample blueprint for multi-modal, real-time conversational AI
within the CloudXR ecosystem.

## Early Access Notice

This project is currently in early access and is under active development.
Features, APIs, documentation, and behavior may change without notice.
Expect bugs, incomplete functionality, and breaking changes as the project
evolves. Use at your own discretion, and please report issues or feedback
to help improve the project.

## What is XR-AI?

XR-AI is a developer stack for building powerful XR and AI systems across devices, platforms, and deployment environments. It connects web, iOS/visionOS, AR glasses, and XR headset clients to GPU-accelerated AI services, tool-using agents, and the CloudXR stack for remote rendering.

With XR-AI, developers can build agents that see and hear what the user experiences, reason over live physical context, call external tools through MCP, and respond with audio or data in the same XR session. The stack provides an end-to-end foundation for multimodal spatial computing applications: real-time media routing, participant-aware response handling, agent interfaces, AI service integration, remote rendering, and sample applications that show the pieces working together.

The value is speed without lock-in. XR-AI is designed to work quickly with NVIDIA open models for vision, language, speech, and speech synthesis, while still giving developers the flexibility to bring their own models, services, tools, and application logic. Because it is built around NVIDIA GPU infrastructure, the same architecture can be deployed where the workload needs to run: cloud, data center, workstation, or edge.

XR-AI also gives developers a practical path across product categories. Teams can start with AI glasses-style experiences that use live camera, audio, and agent responses, then extend the same framework to richer AR glasses or XR headset experiences that use CloudXR remote rendering. This lets developers build for today's lightweight AI devices while keeping a clear path to immersive, GPU-rendered spatial applications.

XR-AI is especially useful when you need to:

- **Build multimodal XR agents** that can see, hear, reason, use tools, and respond in real time.
- **Target multiple client platforms** including web, iOS/visionOS, AR glasses, and XR headsets.
- **Use NVIDIA open models out of the box** while preserving the flexibility to bring your own models and services.
- **Deploy wherever NVIDIA GPUs are available**, from cloud and data center to workstation and edge.
- **Start with AI glasses-style experiences and scale to CloudXR remote rendering** for richer AR and XR applications.
- **Keep transport, rendering, model services, tools, and agent logic separated** so teams can evolve each layer independently.


## Requirements

**Hardware**

XR-AI samples are designed for a single NVIDIA RTX PRO 6000 Blackwell workstation GPU or an
NVIDIA DGX Spark.  Both provide enough VRAM to run the
full model stack locally.  If you prefer not to run models on local hardware,
model endpoints are plain URLs — point the worker config at a cloud NIM or model
endpoint and no local GPU is required for the agent or hub.

| Sample | Local VRAM needed |
|---|---|
| model-servers (all 4 models) | ~70 GB |
| simple-vlm-example (standalone) | ~23 GB |
| xr-render-demo (requires model-servers) | ~70 GB (models) + ~2 GB (hub/TTS) |
| Hub only | none |

**Software**

| Requirement | Version | Notes |
|---|---|---|
| OS | Linux | Ubuntu 22.04 / 24.04 recommended |
| Python | 3.11 or 3.12 | 3.10 and 3.13 are not supported |
| [uv](https://docs.astral.sh/uv/) | latest | dependency manager used by all samples |
| NVIDIA driver | 570+ | required for local model inference |
| npm | 18+ | only needed to rebuild the web vendor bundle |

`uv` handles all Python dependencies per-sample — no global `pip install`
or virtual-environment setup needed.  If you do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**GPU-profile prerequisites** — install before `uv sync` for these targets:

- **DGX Spark** (`xr-render-demo/yaml/spark/`): `sudo apt install python3-dev`
- **RTX PRO 6000 Blackwell** (`xr-render-demo/yaml/96G_blackwell/`): NVIDIA Container Toolkit + CUDA NVCC (`sudo apt install nvidia-cuda-toolkit`)

If `uv sync` or the VLM fails on first run, see
[`docs/troubleshooting.md`](docs/troubleshooting.md).

**Network** — open the firewall ports listed in
[`docs/networking.md`](docs/networking.md) before connecting from another
machine.  UDP 7882 is a silent-failure path: signaling succeeds but media
frames are dropped if it is closed.

## Architecture

| Layer | Directory | Description |
|---|---|---|
| Clients | `client-samples/` | Android, iOS/visionOS, Web clients |
| Server runtime | `server-runtime/` | XR-Media-Hub + LiveKit internal transport |
| Launcher | `utils/xr-ai-launcher/` | stdlib-only process manager used by samples |
| Logging | `utils/xr-ai-logging/` | shared loguru sink + stdlib bridge for every process |
| Agent interfaces | `agent-mcp-servers/` | MCP adapters for XR data & rendering |
| Agent demos | `agent-samples/` | End-to-end agent pipelines |
| Tests | `tests/` | Multi-client / multi-agent integration tests |

Lightweight samples (`simple-vlm-example`) are self-contained — one command
starts everything.  Heavier demos (`xr-render-demo`) split model loading from
the demo itself: start `model-servers` once, then run the demo as many times
as you like without reloading weights.

Every sample worker depends on `agent-sdk/xr-ai-models` — one SDK that
abstracts the OpenAI-compatible HTTP wire format for LLM / VLM / STT / TTS
behind four service protocols.  Each sample ships a `yaml/models.yaml` that
names the logical models the worker needs (`llm`, `vlm`, `stt`, …) with
preset references that pre-fill model-specific quirks (reasoning-field
aliasing, `chat_template_kwargs`, served-model-name strings).  Workers call
`make_llm(config, "llm")` / `make_vlm(config, "vlm")` / `make_stt(config,
"stt")` / `make_tts(config, "tts")` — no hand-rolled httpx clients, no model
quirks leaking out of the SDK.  Full quickstart and the built-in preset
table: [`agent-sdk/xr-ai-models/README.md`](agent-sdk/xr-ai-models/README.md).

## Quickstart

Every sample follows the same pattern: **start the server, then connect a
client.**  Once it is ready, any supported client — web browser, Android app,
iOS/visionOS app, or AR glasses — can join the session using the token printed
on startup.

### Model servers (shared AI services)

`model-servers` starts the four inference services used across demos and exits
immediately — the services keep running in the background with weights hot.
Start this once before running `xr-render-demo`, or whenever you want to
pre-warm models:

```bash
cd agent-samples/model-servers
uv sync
uv run model_servers
```

GPU profiles are auto-detected (`dual_48G_ada` / `spark` / `96G_blackwell`).
On first run each model downloads from HuggingFace (~50 GB total; can take
tens of minutes).  On subsequent runs the containers restart in under a minute.

To stop all model servers when done:

```bash
uv run model_servers --stop
```

### Simple VLM example (vision Q&A over voice + text)

End-to-end voice + vision sample.  Speak into the mic, type into the data
channel, or send the literal text `"ping"` — all routes go through the
same VLM pipeline against the latest video frame.  Replies arrive as
streaming Piper TTS audio plus a `vlm.response` text message.

Uses `nvidia/Cosmos-Reason1-7B` (NVIDIA Open Model License + Apache 2.0).

There are two ways to run it:

**Standalone** (~23 GB VRAM) — starts its own VLM and STT, owns them for the
session, and stops them when you exit:

```bash
cd agent-samples/simple-vlm-example
uv sync
uv run simple_vlm_example
```

On the very first run weights download from HuggingFace (~23 GB; can take
several minutes).

**With model-servers pre-running** — if VLM (port 8100) and STT (port 8103)
are already up from `model-servers`, the demo detects them at startup and
reuses them.  No extra flags needed.  When you exit, those services keep
running.

#### Step 1 — Start the server

```bash
uv run simple_vlm_example
```

The hub, VLM, STT, and TTS start together (or reuse running services).
When ready the hub prints:

```
[hub]   LiveKit URL : wss://0.0.0.0:8080
[hub]   Room        : xr-room
[hub]   Token       : eyJ…
[hub]   Web client  : https://localhost:8080
```

#### Step 2 — Connect a client

Open `https://localhost:8080` in a browser.  The samples ship with HTTPS
on by default (a self-signed cert is generated on first run at
`~/.local/share/xr-ai/web-server.crt`), so you'll see a "Your connection
is not private" warning the first time — click **Advanced → Proceed**
(Chrome/Edge) or **Accept the Risk and Continue** (Firefox).  See
[`docs/networking.md`](docs/networking.md) for trusting the cert
permanently or running over plain HTTP instead.

Leave **Token URL** blank — the web client fetches a token from the server
automatically.  Click **Connect**.

You are now live in the XR session.  To test the agent:

- Type `ping` in the data channel → the agent describes what the camera sees.
- Type any question → sent verbatim to the VLM.
- Speak into your mic → speech is transcribed and sent as a query.

A successful round trip: your query appears in the log, the agent responds
after a moment, and you hear the reply through your speakers.

**Local model** — override the model weights or GPU settings by editing
`vlm_server.yaml` in the sample directory.

**Remote model** — to use a model on another machine or a cloud NIM
endpoint, set `vlm_server` (and optionally `vlm_model_name`) in the
sample's worker YAML (e.g., `simple_vlm_example_worker.yaml`):

```yaml
vlm_server:     http://192.168.1.42:8100   # or https://your-nim-endpoint
vlm_model_name: vlm                        # served-model-name on that host
```

When pointing at a remote model, `vlm_server.yaml` is unused — you can
remove the `vlm_server` entry from the launcher's process list so no local
vLLM process is started.

Each sample has its own `xr_media_hub.yaml` controlling the hub; see
[`server-runtime/xr_media_hub.yaml`](server-runtime/xr_media_hub.yaml)
for the full option list.

---

### XR render demo (voice-driven sphere in CloudXR)

Speak to the web client and a sphere in the streamed scene tracks your
voice — radius follows loudness, colour and position follow spoken commands
("make it red", "put it to my left", "where I'm looking"). Runs against a
Quest 3 / Vision Pro on the same LAN, or the IWER emulator built into the
web client for desktop dev.

Under the hood, the orchestrator launches twelve concurrent processes —
hub, CloudXR runtime, STT / TTS / VLM / two LLM servers, four MCP servers,
and the worker — wired together by a Pipecat pipeline that pairs a fast
Llama-8B for quick-acks with a Nemotron-30B agentic tool-calling loop over
`render-mcp` / `oxr-mcp` / `vlm-mcp` / `video-mcp`. Full process map,
agentic-loop details, and the XR session lifecycle:
[`docs/xr-render-demo.md`](docs/xr-render-demo.md).

**Requires `model-servers` to be running first** — the demo does not start
its own model services.

#### Step 1 — Start model servers (once)

```bash
cd agent-samples/model-servers
uv sync && uv run model_servers
```

This exits immediately once all four services are ready.  Weights stay loaded
in the background.

#### Step 2 — Start the demo

```bash
cd agent-samples/xr-render-demo
uv sync
uv run xr_render_demo
```

On first run the orchestrator automatically downloads LOVR v0.18.0 to
`deps/lovr/` inside the repo and builds the web vendor bundle (requires npm
and network access). Both steps are skipped on subsequent runs.

**DGX Spark (aarch64):** LOVR does not publish a prebuilt aarch64 Linux
binary, so the auto-download is not available — build LOVR from source and
export `LOVR_BIN`. See
[`docs/troubleshooting.md`](docs/troubleshooting.md#dgx-spark--lovr-auto-download-is-not-supported).

To use a custom LOVR build:

```bash
export LOVR_BIN=/path/to/your/lovr   # or set lovr_bin: in render_mcp.yaml
uv run xr_render_demo
```

To stop the model servers when done:

```bash
cd agent-samples/model-servers
uv run model_servers --stop
```

---

### Hub only (server-runtime standalone)

```bash
cd server-runtime
uv sync
uv run xr_media_hub
```

Useful for development or when running an agent in a separate terminal.
The hub auto-discovers `server-runtime/xr_media_hub.yaml`.

## Clients

### Web

Open `https://localhost:8080` in a browser.  The samples ship with HTTPS
on by default; the first connection shows a self-signed cert warning that
you click through (or trust permanently — see
[`docs/networking.md`](docs/networking.md)).  Leave **Token URL** blank to
use the server's built-in `/token` endpoint, or paste the printed token
directly.

The page's import map loads `livekit-client` and `@nvidia/cloudxr` from
`client-samples/web/vendor/` (same-origin, so XR headsets and offline LANs
work).  Both bundles are gitignored build output.  The xr-render-demo
orchestrator builds them automatically on first run (requires npm on PATH).
For a manual rebuild after an SDK bump, see
[`client-samples/web-xr-build/README.md`](client-samples/web-xr-build/README.md).

### Android

See [`client-samples/android/README.md`](client-samples/android/README.md) for
full setup. Quick steps:

1. Open `client-samples/android/` in Android Studio (Hedgehog or later).
2. Let Gradle sync finish — it downloads the LiveKit Android SDK automatically.
3. Run on a device or emulator (API 24+).
4. Enter the server IP, port (`8080` — the hub's web-server port, *not* LiveKit's internal 7880), and paste the printed token.

Permissions (`RECORD_AUDIO`, `CAMERA`) are requested at runtime on first use.

### iOS / visionOS

See [`client-samples/ios-visionos/README.md`](client-samples/ios-visionos/README.md)
for full Xcode setup. Quick connection settings:

| Field | Value |
|---|---|
| Host | IP of the machine running the server |
| Port | `8080` (the hub web-server port; *not* LiveKit's internal 7880) |
| Token | Paste the token printed on server startup |

The token is valid for 24 hours. To get a fresh one restart the server or call
`GET https://<host>:8080/token?identity=<name>`.

> **One-time per device:** the LiveKit Swift SDK does not expose a
> server-trust hook, so iOS rejects the hub's self-signed cert until you
> install it as a trusted profile. On the device, open
> `https://<host>:8080/cert` in Safari → bypass the warning → install →
> enable **Settings → General → About → Certificate Trust Settings →
> Enable Full Trust**. Full walkthrough plus recovery for the common
> failure modes is in
> [`client-samples/ios-visionos/README.md`](client-samples/ios-visionos/README.md)
> under "Trusting the hub's self-signed cert".

## Networking

The hub and CloudXR runtime use a small set of TCP/UDP ports (web client +
wss /rtc proxy on 8080, WebRTC fallbacks on 7881/TCP + 7882/UDP, CloudXR
WSS proxy on 48322). LiveKit's native 7880 stays on loopback — clients
connect through the same-origin wss proxy, not directly. Full table and
distro-specific `ufw` / `firewall-cmd` recipes are in
[`docs/networking.md`](docs/networking.md). The same doc covers HTTPS for
the web client and self-signed certificate trust on each browser.

## Tests

`tests/` contains the multi-client / multi-agent integration suite. The
core IPC tests run without Docker or LiveKit — they spin up real
`HubEndpoint` / `ConnectorEndpoint` / `ProcessorEndpoint` instances over
`ipc://` sockets and verify routing, isolation, and the
`ReturnAudioFlush` control path.

```bash
cd tests
uv sync
uv run pytest -v
```

See [`tests/README.md`](tests/README.md) for the full breakdown. CI runs
the suite on every push and pull request via
[`.github/workflows/tests.yml`](.github/workflows/tests.yml) on Python 3.11
and 3.12.

## Deeper docs

For engineers and agents working in the repo:

| Doc | Topic |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Working contract — hard rules every change must satisfy |
| [`DEPENDENCIES.md`](DEPENDENCIES.md) | Authoritative dependency map (update with every `pyproject.toml` change) |
| [`docs/architecture.md`](docs/architecture.md) | Hub ↔ transport ↔ agent boundaries; known limitations |
| [`docs/process-model.md`](docs/process-model.md) | `Process` / `run_stack` mechanics; ready-file protocol |
| [`docs/ai-services.md`](docs/ai-services.md) | VLM / STT / TTS / LLM server reference + worker call examples |
| [`docs/xr-render-demo.md`](docs/xr-render-demo.md) | xr-render-demo architecture: 12-process stack, agentic loop, XR lifecycle |
| [`docs/adding-a-sample.md`](docs/adding-a-sample.md) | Boilerplate for scaffolding a new sample |
| [`docs/adding-cloudxr.md`](docs/adding-cloudxr.md) | Wiring CloudXR into a sample |
| [`docs/credentials.md`](docs/credentials.md) | HF / NGC token management |
| [`docs/networking.md`](docs/networking.md) | Firewall ports + HTTPS for the web client |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Known frictions and runtime symptoms |
| [`docs/spdx-headers.md`](docs/spdx-headers.md) | SPDX header style and enforcement |
| [`docs/changelog.md`](docs/changelog.md) | Significant design decisions, reverse chronological |

## Project meta

- [`LICENSE`](LICENSE) — Apache-2.0.
- [`SECURITY.md`](SECURITY.md) — how to report a vulnerability.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution process and DCO.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — community standards.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — bundled third-party
  components and their licenses.
