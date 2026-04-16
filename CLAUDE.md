# xr-ai — Claude Working Instructions

## Project Overview

Agentic AI for XR (Extended Reality), built on CloudXR ecosystem.
XR-Media-Hub is the primary server runtime; LiveKit is the internal transport layer.

## Architecture

```
client-samples/        # Platform clients (Android, iOS/visionOS, Web)
server-runtime/        # XR-Media-Hub core + LiveKit transport
agent-mcp-servers/     # MCP adapters: oxr, render, client, xr-media
agent-samples/         # E2E agent demos (Pipecat + NAI, Pipecat + LLM)
docs/                  # Design docs
```

## Key Design Decisions

- **XR-Media-Hub** is transport-agnostic at its IPC boundary. AI frameworks connect via IPC only.
- **LiveKit** is the internal transport between clients and the hub. It is an implementation detail — not exposed to the agent layer.
- MCP servers are the agent's only interface to XR data and rendering.
- No API keys or tokens in source; use env vars or secret stores.

## Module Conventions

- `server-runtime/xr_media_hub/ipc/` — client-side and server-side IPC endpoints
- `server-runtime/xr_media_hub/transport/livekit/` — LiveKit connector (internal)
- Each `agent-mcp-servers/<name>/` is an independent MCP server process
- Client samples wrap LiveKit SDK with a thin, platform-neutral API surface

## Coding Norms

- Prefer clarity over cleverness — XR code runs on constrained hardware.
- Note latency / memory budget constraints in commit messages when touching inference code.
- Don't add abstractions until needed by two concrete use-cases.
