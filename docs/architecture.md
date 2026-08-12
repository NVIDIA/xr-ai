<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Architecture

Read this when working across module boundaries or onboarding to the
overall design. For day-to-day rules see `AGENTS.md`; for historical
design decisions see `docs/changelog.md`.

## Top-level layout

```
client-samples/     # Platform clients (Android, iOS/visionOS, Web)
agent-sdk/          # Agent runtime, IPC, model, NAT-function, capability, and voice SDK packages
utils/              # Shared infra: stdlib-only launcher + loguru logging bridge
services/           # XR hub, CloudXR, model-serving, and typed XR capability services
agent-mcp-servers/  # Optional MCP compatibility adapters for non-NAT consumers
agent-samples/      # End-to-end agent demos
tests/              # Multi-client / multi-agent integration tests
docs/               # Design docs and topic deep-dives
```

## Key design decisions

- **One hub, many clients, many agents.** A single hub instance fans the
  inbound stream out to every connected ``ProcessorEndpoint`` (agent) and
  routes return traffic back to the originating client only — never to peers.
- **XR-Media-Hub** is transport-agnostic at its IPC boundary. Agents connect
  via IPC only.
- **LiveKit** is an internal transport detail — not exposed to the agent layer.
  When LiveKit is the transport, return audio is published as one track per
  participant (`xr-hub-return-{pid}`) with subscribe permissions restricted to
  that participant; return data uses ``destination_identities`` for the same
  reason. Agents never need to know.
- **`agent-sdk/xr-ai-hub-client`** contains only the agent-facing IPC layer. Its
  sole runtime dependencies are `pyzmq` and `msgpack` — no LiveKit, FastAPI,
  or uvicorn.
- **`agent-sdk/xr-ai-agent-runtime`** provides typed publish/subscribe routing.
  Agents expose existing `Tool` and `AsyncTool` instances from `xr-ai-tools`;
  direct callers and model loops use those tools without a second runtime
  dispatch API. Each agent owns its resources, background tasks, lifecycle,
  and concurrency policy. Model loops, planning, memory, and raw media
  transport remain outside the runtime.
- **Native agents compose typed tools in process.** Model-backed tools call
  typed capability services, while deterministic tools run locally. MCP
  adapters only republish selected tools for MCP consumers.
- **No API keys or tokens in source files** — use env vars or
  `xr_media_hub.yaml` (see `docs/credentials.md`).

## Hub config

Each sample provides its own `xr_media_hub.yaml` in its `yaml/` directory
(e.g. `agent-samples/simple-vlm-example/yaml/xr_media_hub.yaml`).
`services/xr-media-hub/` also contains a reference copy documenting all available
fields.

Paths inside the YAML (e.g. `web_client_dir`) resolve relative to the YAML
file's own directory, not CWD. `HubLauncher` finds the YAML automatically by
searching upward from CWD when the orchestrator runs.

## Known limitations

For runtime symptoms and fixes that aren't architectural, see
[`docs/troubleshooting.md`](troubleshooting.md).

### LiveKit signaling is fronted by a same-origin wss:// proxy

LiveKit-server itself still runs plain `ws://` on the loopback interface
(`127.0.0.1:7880`). The hub's web server (`_web_server.py`) terminates TLS
on `web_server_port` (8080 by default) and exposes a same-origin
`wss://<host>:8080/rtc` route that proxies LiveKit signaling
bidirectionally (`_lk_proxy.py`). Every external client — browser, web-xr,
Android, iOS, visionOS — connects only to that wss URL; nothing reaches
LiveKit's 7880 from off-box.

The `/token` endpoint returns `url: wss://<host>:<web_server_port>` when
`web_server_tls: true` (the default), so the URL the client SDK uses comes
straight from the server — no client-side toggle needed.

WebRTC media (7881/TCP fallback, 7882/UDP) is DTLS/SRTP regardless, so no
extra encryption is needed on those ports.

To run a fully plain stack for `localhost` dev, set `web_server_tls: false`
— `/token` then returns `ws://`, and the same-origin proxy serves plain
WebSocket. `localhost` is the only context where browsers grant
camera/mic permissions without HTTPS.
