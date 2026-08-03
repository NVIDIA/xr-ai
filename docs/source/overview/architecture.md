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
agent-sdk/          # IPC, model, NAT-function, capability, and voice SDK packages
utils/              # Shared infra: launcher, logging, vad, vllm, voicegate
cloudxr-runtime/    # NVIDIA CloudXR integration: OpenXR runtime + WSS proxy, opt-in per sample
ai-services/        # OpenAI-compatible AI inference servers (VLM, STT, TTS, LLM)
services/           # Long-running typed XR capability services
agent-mcp-servers/  # Optional MCP compatibility adapters for non-NAT consumers
agent-samples/      # End-to-end agent demos
tests/              # Multi-client / multi-agent integration tests
docs/               # Design docs and topic deep-dives
models/             # Gitignored model-weight cache
deps/               # Gitignored downloaded runtime binaries
```

## Dependency diagram

The labels on the arrows are part of the contract: an HTTP or typed-RPC
connection is a runtime boundary, not a Python package dependency.

```text
 client-samples/
      ├── LiveKit media + data ──▶ server-runtime/ (XR-Media-Hub)
      │                                      │
      │                                      └── hub IPC via xr-ai-hub-client ──┐
      │                                                                          ▼
      └── CloudXR / WebRTC ──▶ cloudxr-runtime/                 agent-samples/<sample>/worker
                                      │                              │       │       │
                                      │                              │       │       └── imports voice runtime + utils
                                      │                              │       └── imports xr-ai-models ── HTTP ──▶ ai-services/
                                      │                              └── imports xr-ai-nat
                                      │                                      ├── invokes deterministic functions in process
                                      │                                      └── typed RPC ──▶ services/
                                      │                                                       or
                                      │                                              agent-samples/<sample>/<capability>
                                      │                                                       ▲
                                      └──────────── OpenXR, for example ──────────────────────┘

 external MCP consumer ── MCP ─▶ agent-mcp-servers/
                                      │
                                      └─ republishes selected NAT or
                                         sample-local capabilities
```

The worker reaches XR-Media-Hub through `xr-ai-hub-client`. The diagram leaves
that package on the IPC edge instead of drawing another box through the middle.
MCP adapters are not on a native sample's execution path.

## Folder ownership

| Folder | Owns | Dependency boundary |
|---|---|---|
| `client-samples/` | Platform UI, device capture, and client SDK integration | Talks to the hub or CloudXR over network protocols; contains no agent or model-service implementation |
| `server-runtime/` | Media fan-out, same-participant return routing, recording, and LiveKit transport | Depends on `xr-ai-hub-client`; workers never import it |
| `agent-sdk/` | Reusable, in-process APIs used by agent applications | Must not depend on a sample; keep optional framework and capability dependencies in their own distributions or extras |
| `utils/` | Process-agnostic launcher, logging, VAD, vLLM, and voice-gate infrastructure | Must not own application capabilities; dependency limits are defined in `DEPENDENCIES.md` |
| `ai-services/` | OpenAI-compatible LLM, VLM, STT, and TTS server processes | Called through `xr-ai-models`, never through hand-written worker HTTP clients |
| `services/` | Reusable long-running capability providers, such as OpenXR tracking and recorded video memory | Exposes typed service contracts consumed by NAT functions; a process boundary does not imply MCP |
| `agent-samples/` | End-to-end applications: orchestration, workers, configuration, prompts, and sample-specific capabilities | May compose SDK packages and services; reusable capabilities must move to `agent-sdk/` or `services/` once a second application needs them |
| `agent-mcp-servers/` | Optional outward compatibility adapters | May publish selected native functions or sample-local capabilities; native workers do not depend on or launch them |
| `cloudxr-runtime/` | Shared CloudXR OpenXR runtime and signaling proxy | Opt-in managed process; rendering application state remains with its sample |
| `tests/` | Cross-package, multi-client, and compatibility integration tests | May exercise several distributions; package-local behavior should still be testable at its owning boundary |
| `docs/` | Current architecture, guides, and historical decisions | `docs/changelog.md` records decisions; topic pages describe the current contract |
| `models/`, `deps/` | Downloaded weights and runtime binaries | Gitignored caches only; never the source of importable code |

### Agent SDK organization

```text
agent-sdk/
├── xr-ai-hub-client/    # Minimal pyzmq + msgpack IPC client
├── xr-ai-models/        # Model protocols, presets, and OpenAI-compatible clients
├── xr-ai-nat/           # Typed NAT functions and framework adapters
├── xr-ai-voice/         # VoiceSession runtime
├── xr-ai-pipecat/       # Optional Pipecat pipeline bridge
└── xr-ai-capabilities/  # Framework-agnostic reusable capabilities
```

### Agent sample organization

```text
agent-samples/<sample>/
├── main.py              # Orchestrator: declares managed processes
├── pyproject.toml       # Orchestrator dependencies only
├── yaml/                # Hub, worker, model, and process configuration
├── worker/
│   ├── pyproject.toml   # Worker SDK and task dependencies
│   └── ...              # Agent logic, prompts, and worker entry point
└── <capability>/        # Optional sample-specific service/function slice
```

Put code in a sample-local capability directory when it is inseparable from
that application, as `xr-render-demo/scene/` is from LOVR scene state. Put a
reusable in-process contract in `agent-sdk/`; put its reusable long-running
provider in `services/`. `agent-mcp-servers/` is only the publication boundary
for consumers that cannot invoke NAT functions directly.

### Placement rules

When adding a component, choose its folder by ownership:

1. Put platform UI or device integration in `client-samples/`.
2. Put media routing and transport implementation in `server-runtime/`.
3. Put an in-process API needed by more than one application in `agent-sdk/`.
4. Put a reusable long-running capability provider in `services/`, with its
   typed agent-facing function or client in `xr-ai-nat`.
5. Keep application-specific functions, services, assets, and prompts under
   `agent-samples/<sample>/`.
6. Put model-serving processes in `ai-services/` and call them through
   `xr-ai-models`.
7. Put process-management and cross-cutting infrastructure in `utils/`.
8. Add an `agent-mcp-servers/` adapter only to publish an existing capability
   to MCP-only consumers; do not move the capability implementation there.

Do not add a new top-level folder when one of these owners already fits.

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
- **`agent-sdk/xr-ai-hub-client`** contains only the agent-facing IPC layer. Its
  sole runtime dependencies are `pyzmq` and `msgpack` — no LiveKit, FastAPI,
  or uvicorn.
- **Native agents compose typed NAT functions in process.** Runtime-backed
  functions call typed capability services, while deterministic functions run
  locally. MCP adapters only republish selected functions for MCP consumers.
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
