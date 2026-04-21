# xr-ai — Working Conventions

Guidelines for developers and AI assistants working in this repo.

## Architecture

```
client-samples/     # Platform clients (Android, iOS/visionOS, Web)
server-runtime/     # XR-Media-Hub core + LiveKit transport
launcher/           # stdlib-only process manager (used by samples)
agent-mcp-servers/  # MCP adapters: oxr, render, client, xr-media
agent-samples/      # End-to-end agent demos
docs/               # Design docs
```

Key design decisions:
- **XR-Media-Hub** is transport-agnostic at its IPC boundary. Agents connect via IPC only.
- **LiveKit** is an internal transport detail — not exposed to the agent layer.
- MCP servers are the agent's only interface to XR data and rendering.
- No API keys or tokens in source files — use env vars or `xr_media_hub.yaml`.

## Process model

Every sample is self-contained: running it starts the hub and all required
processes automatically. No separate server launch step.

- `xr_media_hub` always runs as its own process — never embedded in-process.
- Process management lives in `launcher/`, not inside any process it manages.
- Agents connect to the hub via IPC (`xr_media_hub.ipc.ProcessorEndpoint`).

## Adding a new sample

1. Create `agent-samples/<name>/` with its own `pyproject.toml`.
2. Declare local dependencies by name in `[project.dependencies]` and resolve
   them with `[tool.uv.sources]` — **not** with inline `@ ../../path` syntax,
   which breaks at wheel build time:
   ```toml
   [project]
   dependencies = ["xr-media-hub", "xr-ai-launcher"]

   [tool.uv.sources]
   xr-media-hub  = { path = "../../server-runtime", editable = true }
   xr-ai-launcher = { path = "../../launcher",      editable = true }
   ```
3. Wrap `main()` with `async with HubLauncher():` to start the full stack.
4. Update `README.md` — architecture table and quickstart section.

## Adding a new managed process type

Add `launcher/xr_ai_launcher/_<name>.py` following the pattern in `_hub.py`.
Use `ManagedProcess` as the base. Export from `__init__.py`.

## Documentation rule

**Update `README.md` (and relevant sub-repo docs) in the same task as the code
change.** A change is not done until the docs reflect it. This applies to: new
packages, changed entry points, new quickstart flows, renamed commands, new
config files.

## Dependency discipline

- `launcher/` has zero runtime dependencies — stdlib only. Keep it that way.
- Samples depend on `xr-media-hub` (IPC types) + `xr-ai-launcher` (launch) only.
- Don't add abstractions until needed by two concrete use-cases.

## Config

`xr_media_hub.yaml` at the repo root is the primary config. Paths inside it
resolve relative to the file's location. `HubLauncher` finds it automatically
by searching upward from CWD.

---

## Decisions & change log

Significant decisions, in reverse-chronological order. Update this whenever a
non-trivial architectural or design decision is made so the rationale is
preserved and not re-litigated.

### 2026-04-21 — Process management moved to `launcher/`

`HubLauncher` lives in `launcher/xr_ai_launcher/`, not in `server-runtime`.
**Why:** process management should not be part of the processes it manages.
The launcher will eventually start MCP servers, CloudXR runtime, and other
components — keeping it separate keeps dependency chains lean and the boundary
clean. `launcher/` has zero runtime dependencies (stdlib only).

### 2026-04-21 — `AgentEndpoint` + `ConsumerEndpoint` → `ProcessorEndpoint`

`ipc/_agent.py` and `ipc/_consumer.py` are deleted. Both are replaced by a
single `ProcessorEndpoint` in `ipc/_processor.py`.
**Why:** `ConsumerEndpoint` was unused scaffolding; `AgentEndpoint` was too
narrow a name (the endpoint suits analytics, recording, etc. — not just agents).
`ProcessorEndpoint` auto-maintains `connected_participants: frozenset[str]` so
processors always know who is present without manual event tracking.

### 2026-04-21 — Agent return path through hub

Agents push `RETURN_DATA`/`RETURN_AUDIO` on the hub's PULL socket.
The hub's `_dispatch` routes them to `send_return_data`/`send_return_audio`,
which PUBs them on `return_data.<pid>` / `return_audio.<pid>` topics.
The `ConnectorEndpoint` SUBs these topics and calls registered callbacks
→ `RoomClient` → LiveKit → client.
**Why:** closes the loop so agents can send audio and data back to participants.

### 2026-04-21 — Echo-agent sample added

`agent-samples/echo-agent/` — echoes audio back to the originating participant
and sends a JSON stats ping (`topic="agent.stats"`) every 5 s to each
connected participant. Demonstrates `ProcessorEndpoint` usage end-to-end.

### 2026-04-20 — Track task management keyed by track SID

`RoomClient._track_tasks` changed from `list[Task]` to `dict[str, Task]`
keyed by track SID. A `track_unsubscribed` handler cancels the exact task.
**Why:** without this, stop/start camera caused a new streaming task to start
while the old one kept running, doubling (then tripling) fps counts.

### 2026-04-20 — Audio format: float32 on the wire, int16 in LiveKit

LiveKit delivers audio as int16 PCM. The hub's IPC layer (`AudioChunk`) uses
float32 LE interleaved. Conversion happens in `_room_client.py`:
- Inbound: `int16 / 32768.0 → float32`
- Outbound (return audio): `clip(float32, -1, 1) * 32767 → int16`

### 2026-04-20 — `xr_media_hub.yaml` config file

Flat YAML at repo root. Fields map 1:1 to `LiveKitConnectorConfig` dataclass.
Relative paths (e.g. `web_client_dir`) resolve relative to the YAML file's
own directory, not CWD. `HubLauncher` searches upward from CWD to find it
automatically.
