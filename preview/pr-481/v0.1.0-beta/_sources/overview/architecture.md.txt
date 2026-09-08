<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Architecture

This page explains how XR-Media-Hub, the transport, and agents fit together.

## Top-level layout

```
client-samples/     # Platform clients (Android, iOS/visionOS, Web)
server-runtime/     # XR-Media-Hub core + LiveKit transport
agent-sdk/          # xr-ai-agent (IPC client), xr-ai-models (model seam), xr-ai-pipecat (voice pipeline)
utils/              # Shared infra: launcher, logging, vad, vllm, voicegate
cloudxr-runtime/    # NVIDIA CloudXR integration: OpenXR runtime + WSS proxy, opt-in per sample
ai-services/        # OpenAI-compatible AI inference servers (VLM, STT, TTS, LLM)
agent-mcp-servers/  # MCP adapters: oxr, render, transcript, vec, video, vlm
agent-samples/      # End-to-end agent demos
tests/              # Multi-client / multi-agent integration tests
docs/               # Design docs and topic deep-dives
```

## Key design decisions

- **One hub, many clients, many agents.** A single hub instance fans the
  inbound stream out to every connected `ProcessorEndpoint` (agent) and
  routes return traffic back to the originating client only — never to peers.
- **XR-Media-Hub** is transport-agnostic at its IPC boundary. Agents connect
  via IPC only.
- **LiveKit** is an internal transport detail — not exposed to the agent layer.
  When LiveKit is the transport, return audio is published as one track per
  participant (`xr-hub-return-{pid}`) with subscribe permissions restricted to
  that participant; return data uses `destination_identities` for the same
  reason. Agents never need to know.
- **`agent-sdk/`** (`xr-ai-agent`) contains only the agent-facing IPC layer.
  Its sole runtime dependencies are `pyzmq` and `msgpack` — no LiveKit,
  FastAPI, or uvicorn.
- **MCP servers are the agent's only interface to XR data and rendering.**
- **No API keys or tokens in source files** — use environment variables or
  `xr_media_hub.yaml` (refer to {doc}`/getting_started/credentials`).

Refer to {doc}`/components/server-runtime` for more on the hub and transport.

## Multi-user sessions

A single XR-Media-Hub session can carry several participants at once, each fully
isolated. The hub is not a routing switch between participants: media and data
flow only between a participant and the agent, never from one participant to
another. The supported path is always:

```
participant → hub → agent → hub → same participant
```

The hub enforces this per participant:

- Return audio is published as one LiveKit track per participant, with subscribe
  permission restricted to that participant.
- Return data is addressed to the originating participant's identity.
- Return-traffic topics are matched on a terminated identity segment, so one
  participant id cannot be a prefix of another (`user1` never matches `user10`).

Because every response is addressed back to the participant it came from, a
single agent process can serve several participants concurrently without
cross-talk: it receives each participant's audio, video, and data tagged with
that participant's id, and addresses its replies to the same id. How richly a
given sample uses this is up to the sample — the `glasses-agent` and
`simple-vlm-example` workers are written around one active speaker, while the
transport and isolation guarantees hold for any number of connected participants.

Refer to the {doc}`Isolation contract </components/server-runtime>` for the
enforcement details.

## Hub configuration

Each sample provides its own `xr_media_hub.yaml` in its `yaml/` directory
(e.g. `agent-samples/simple-vlm-example/yaml/xr_media_hub.yaml`).
`server-runtime/` also contains a reference copy documenting all available
fields.

Paths inside the YAML (e.g. `web_client_dir`) resolve relative to the YAML
file's own directory, not CWD. `HubLauncher` finds the YAML automatically by
searching upward from CWD when the orchestrator runs.

## Known limitations

Refer to {doc}`/guides/troubleshooting` for runtime symptoms and fixes that
aren't architectural.

### LiveKit signaling is fronted by a same-origin wss:// proxy

LiveKit-server itself still runs plain `ws://` on the loopback interface
(`127.0.0.1:7880`). The hub's web server (`_web_server.py`) terminates TLS
on `web_server_port` (`8080` by default) and exposes a same-origin
`wss://<host>:8080/rtc` route that proxies LiveKit signaling
bidirectionally (`_lk_proxy.py`). Every external client — browser, web-xr,
Android, iOS, visionOS — connects only to that wss URL; nothing reaches
LiveKit's 7880 from off-box.

The `/token` endpoint returns `url: wss://<host>:<web_server_port>` when
`web_server_tls: true` (the default), so the URL the client SDK uses comes
straight from the server — no client-side toggle needed.

WebRTC media (7881/TCP fallback, 7882/UDP) is DTLS/SRTP regardless, so no
extra encryption is needed on those ports.

To run a fully plain stack for `localhost` development, set
`web_server_tls: false` — `/token` then returns `ws://`, and the same-origin
proxy serves plain WebSocket. `localhost` is the only context where browsers
grant camera and microphone permissions without HTTPS.
