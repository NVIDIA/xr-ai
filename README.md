<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- TODO: hero image -->

# xr-ai

**Agentic AI for XR.** A blueprint for multi-modal, real-time conversational
AI inside the CloudXR ecosystem — talk to a vision model about what's in
front of you, drive 3D scenes with your voice, run multiple agents over a
single XR session.

Each sample is **self-contained**: after `uv sync`, a single `uv run` starts
the hub, the AI inference servers, and the agent worker together. No
separate server step, no Docker required, no cloud dependency.

## What you can build with it

- **Voice + vision Q&A** over a live XR camera feed (`simple-vlm-example`).
- **Voice-driven 3D scene control** in CloudXR — colour, position, size of
  objects in the streamed scene (`xr-render-demo`).

---

## Prerequisites

Read this before the first `uv run`. Most first-time setup pain comes from
one of these. If you hit something not covered here, check
[`docs/troubleshooting.md`](docs/troubleshooting.md).

### System

- **Linux** with an NVIDIA GPU and a working driver (NVDEC + NVENC are
  required — the hub fails fast if either is missing).
- **Python 3.11 or 3.12** and [`uv`](https://docs.astral.sh/uv/).
- **Open firewall ports** before connecting from another machine — see
  [`docs/networking.md`](docs/networking.md). LiveKit uses UDP 7882 for
  WebRTC media; signaling-only connections "appear to work" with no audio
  if 7882 is closed.

### GPU profile-specific

`xr-render-demo` ships three pre-tuned YAML profiles under
`agent-samples/xr-render-demo/yaml/` — pick the one that matches your
hardware. Each has its own host prerequisite:

| Profile | Hardware | Extra system prereq |
|---|---|---|
| `dual_48G_ada/` | 2× RTX 6000 Ada (or similar 48 GB Ada) | none beyond the system list above |
| `spark/` | NVIDIA Spark / Jetson Thor | `sudo apt install python3-dev` **before** `uv sync` |
| `96G_blackwell/` | B200, RTX PRO 6000, Jetson Thor (Blackwell) | NVIDIA Container Toolkit + CUDA NVCC (`sudo apt install nvidia-cuda-toolkit`) |

`simple-vlm-example` runs on any single GPU with ~16 GB VRAM and has no
profile-specific prereq.

### First-run downloads

Model weights download into `models/` at the repo root (gitignored, ~16 GB
for the default VLM). The first `uv run` may appear to hang for several
minutes while this happens. Subsequent runs use the cache and start in
~30–60 s.

---

## Quickstart

### Simple VLM example (vision Q&A over voice + text)

End-to-end voice + vision sample. Speak into the mic, type into the data
channel, or send the literal text `"ping"` — all routes go through the
same VLM pipeline against the latest video frame. Replies arrive as
streaming Piper TTS audio plus a `vlm.response` text message.

Requires ~16 GB VRAM (default VLM is `nvidia/Cosmos-Reason1-7B`, NVIDIA
Open Model License + Apache 2.0).

```bash
cd xr-ai/agent-samples/simple-vlm-example
uv sync
uv run simple_vlm_example
```

On startup you will see hub output (prefixed `[hub]`) followed by the
worker connecting. The hub prints connection details:

```
[hub]   LiveKit URL : ws://0.0.0.0:7880
[hub]   Room        : xr-room
[hub]   Token       : eyJ…   ← paste this into the client
[hub]   Web client  : http://localhost:8080
```

**Client protocol** — anything you send via the data channel is treated
as a query:

- `"ping"` → uses the configured default prompt (`"Describe what you see."`).
- Any other UTF-8 text → used as the query verbatim.
- Audio from the mic → STT (parakeet) → query.

Override the VLM model by editing `vlm_server.yaml` in the sample directory.
Each sample has its own `xr_media_hub.yaml` controlling the hub; see
[`server-runtime/xr_media_hub.yaml`](server-runtime/xr_media_hub.yaml) for
the full option list. The VLM and TTS models are loaded at startup
(~30–60 s after the first-run download) and ready before the first query.

### XR render demo (voice-driven sphere in CloudXR)

Speak to the web client and a sphere in the streamed scene tracks your
voice — radius follows loudness, colour and position follow spoken commands
("make it red", "put it to my left", "where I'm looking"). Runs against a
Quest 3 / Vision Pro on the same LAN, or the IWER emulator built into the
web client for desktop dev.

```bash
cd xr-ai/agent-samples/xr-render-demo
uv run xr_render_demo
```

On first run the orchestrator automatically downloads LOVR v0.18.0 to
`deps/lovr/` inside the repo and builds the web vendor bundle (requires npm
and network access). Both steps are skipped on subsequent runs.

To use a custom LOVR build instead:

```bash
export LOVR_BIN=/path/to/your/lovr   # or set lovr_bin: in render_mcp.yaml
uv run xr_render_demo
```

Plan ~17 GB VRAM for the LLM + STT models on first run (weights download
on demand). To free VRAM after exit see
[`docs/troubleshooting.md`](docs/troubleshooting.md) — vLLM-backed servers
intentionally survive stack restarts.

### Hub only (server-runtime standalone)

```bash
cd xr-ai/server-runtime
uv sync
uv run xr_media_hub
```

Useful for development or when running an agent in a separate terminal.
The hub auto-discovers `server-runtime/xr_media_hub.yaml`.

---

## Sample tour

| Sample | What it shows | VRAM | Run |
|---|---|---|---|
| [`simple-vlm-example`](agent-samples/simple-vlm-example/) | Voice + vision Q&A over the latest XR frame | ~16 GB | `uv run simple_vlm_example` |
| [`xr-render-demo`](agent-samples/xr-render-demo/) | Voice-driven 3D scene in CloudXR | ~17 GB | `uv run xr_render_demo` |

---

## Clients

The web client is bundled and served by the hub on `http://localhost:8080`.
Native clients live in `client-samples/`:

| Client | Setup |
|---|---|
| Web | Open `http://localhost:8080`. Token URL blank to use the server endpoint, or paste the printed token. |
| Android | [`client-samples/android/README.md`](client-samples/android/README.md) — Android Studio Hedgehog+, API 24+ |
| iOS / visionOS | [`client-samples/ios-visionos/README.md`](client-samples/ios-visionos/README.md) — Xcode |

For HTTPS web client setup (required for camera access from remote devices)
and firewall rules, see [`docs/networking.md`](docs/networking.md).

### Web vendor bundle

The page's import map loads `livekit-client` and `@nvidia/cloudxr` from
`client-samples/web/vendor/` (same-origin, so XR headsets and offline LANs
work). Both bundles are gitignored build output. The xr-render-demo
orchestrator builds them automatically on first run (requires npm on PATH).
For a manual rebuild after an SDK bump, see
[`client-samples/web-xr-build/README.md`](client-samples/web-xr-build/README.md).

---

## Tests

`tests/` contains the multi-client / multi-agent integration suite. The
core IPC tests run without Docker or LiveKit — they spin up real
`HubEndpoint` / `ConnectorEndpoint` / `ProcessorEndpoint` instances over
`ipc://` sockets and verify routing, isolation, and the
`ReturnAudioFlush` control path.

```bash
cd xr-ai/tests
uv sync
uv run pytest -v
```

See [`tests/README.md`](tests/README.md) for the full breakdown. CI runs
the suite on every push and pull request via
[`.github/workflows/tests.yml`](.github/workflows/tests.yml) on Python 3.11
and 3.12.

---

## Deeper docs

For engineers and agents working in the repo:

| Doc | Topic |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Working contract — hard rules every change must satisfy |
| [`DEPENDENCIES.md`](DEPENDENCIES.md) | Authoritative dependency map (update with every `pyproject.toml` change) |
| [`docs/architecture.md`](docs/architecture.md) | Hub ↔ transport ↔ agent boundaries; known limitations |
| [`docs/process-model.md`](docs/process-model.md) | `Process` / `run_stack` mechanics; ready-file protocol |
| [`docs/ai-services.md`](docs/ai-services.md) | VLM / STT / TTS / LLM server reference + worker call examples |
| [`docs/adding-a-sample.md`](docs/adding-a-sample.md) | Boilerplate for scaffolding a new sample |
| [`docs/adding-cloudxr.md`](docs/adding-cloudxr.md) | Wiring CloudXR into a sample |
| [`docs/credentials.md`](docs/credentials.md) | HF / NGC token management |
| [`docs/networking.md`](docs/networking.md) | Firewall ports + HTTPS for the web client |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Known frictions and runtime symptoms |
| [`docs/spdx-headers.md`](docs/spdx-headers.md) | SPDX header style and enforcement |
| [`docs/changelog.md`](docs/changelog.md) | Significant design decisions, reverse chronological |

---

## Project meta

- [`LICENSE`](LICENSE) — Apache-2.0.
- [`SECURITY.md`](SECURITY.md) — how to report a vulnerability.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution process and DCO.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — community standards.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — bundled third-party
  components and their licenses.
