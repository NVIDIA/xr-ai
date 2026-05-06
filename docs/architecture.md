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
server-runtime/     # XR-Media-Hub core + LiveKit transport
agent-sdk/          # xr-ai-agent: IPC client library (pyzmq + msgpack only)
utils/              # Shared infra: stdlib-only launcher + loguru logging bridge
cloudxr-runtime/    # Shared CloudXR OpenXR runtime + WSS proxy (opt-in per sample)
ai-services/        # OpenAI-compatible AI inference servers (VLM, STT, TTS, LLM)
agent-mcp-servers/  # MCP adapters: oxr, render, transcript, video, vlm
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
- **`agent-sdk/`** (`xr-ai-agent`) contains only the agent-facing IPC layer.
  Its sole runtime dependencies are `pyzmq` and `msgpack` — no LiveKit,
  FastAPI, or uvicorn.
- **MCP servers are the agent's only interface to XR data and rendering.**
- **No API keys or tokens in source files** — use env vars or
  `xr_media_hub.yaml` (see `docs/credentials.md`).

## Hub config

Each sample provides its own `xr_media_hub.yaml` in its `yaml/` directory
(e.g. `agent-samples/simple-vlm-example/yaml/xr_media_hub.yaml`).
`server-runtime/` also contains a reference copy documenting all available
fields.

Paths inside the YAML (e.g. `web_client_dir`) resolve relative to the YAML
file's own directory, not CWD. `HubLauncher` finds the YAML automatically by
searching upward from CWD when the orchestrator runs.

## Known limitations

For runtime symptoms and fixes that aren't architectural, see
[`docs/troubleshooting.md`](troubleshooting.md).

### LiveKit always uses plain `ws://` (no TLS)

The web server (`web_server_tls: true`) and token endpoint both support HTTPS,
but LiveKit itself always runs over plain WebSocket (`ws://`).  This means:

- The `/token` response returns `url: ws://<host>:<lk_port_ws>`.
- Browsers loaded over HTTPS will block the `ws://` connection as mixed content.
- Native clients (iOS, visionOS, Android) are unaffected — they accept both.

**Workarounds until LiveKit TLS is added:**

1. Use a reverse proxy (nginx, Caddy) in front of LiveKit to terminate TLS and
   forward as plain WebSocket internally.  Point `web_client_dir` at a build
   that targets the proxy URL.
2. Run the web client over plain HTTP (`web_server_tls: false`), which avoids
   the mixed-content restriction.  Camera/mic access requires a secure context,
   so this only works on `localhost` or with a browser flag override.
3. Add native LiveKit TLS: set `tls.cert` and `tls.key` in the generated
   `livekit.yaml` (see `_docker.py`) and change the token URL scheme to `wss://`.
   This is the correct long-term fix but has not been implemented yet.
