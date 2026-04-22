# xr-ai — Working Conventions

Guidelines for developers and AI assistants working in this repo.

## Architecture

```
client-samples/     # Platform clients (Android, iOS/visionOS, Web)
server-runtime/     # XR-Media-Hub core + LiveKit transport
agent-sdk/          # xr-ai-agent: IPC client library (pyzmq + msgpack only)
launcher/           # stdlib-only process manager (used by samples)
agent-mcp-servers/  # MCP adapters: oxr, render, client, xr-media
agent-samples/      # End-to-end agent demos
docs/               # Design docs
```

Key design decisions:
- **XR-Media-Hub** is transport-agnostic at its IPC boundary. Agents connect via IPC only.
- **LiveKit** is an internal transport detail — not exposed to the agent layer.
- **`agent-sdk/`** (`xr-ai-agent`) contains only the agent-facing IPC layer. Its sole
  runtime dependencies are `pyzmq` and `msgpack` — no LiveKit, FastAPI, or uvicorn.
- MCP servers are the agent's only interface to XR data and rendering.
- No API keys or tokens in source files — use env vars or `xr_media_hub.yaml`.

## Process model

Every sample is self-contained: running it starts the hub and all required
processes automatically. No separate server launch step.

Each sample has **two sub-projects**:

| Sub-project | Role | Dependencies |
|---|---|---|
| `<sample>/` | Orchestrator — declares process list in code, launches all | `xr-ai-launcher` only (stdlib) |
| `<sample>/worker/` | Agent worker — connects to hub via IPC, runs agent logic | `xr-ai-agent`, numpy, etc. |

**Launchable convention** — every sub-project that can be run is self-describing:
it has an entry-point command and optionally a YAML config named `<command>.yaml`
that lives in the sample root.  The launcher discovers the YAML automatically
and passes it as `--config`.  No separate launcher config file exists.

The orchestrator declares the process sequence in code:
```python
_BASE = Path(__file__).resolve().parents[1]   # sample root

PROCESSES = [
    Process("hub",    "../../server-runtime", "xr_media_hub"),
    Process("worker", "worker",               "my_agent_worker"),
    # future: Process("cloudxr", "../../cloudxr-runtime", "cloudxr_runtime"),
    #         Process("mcp",     "../../agent-mcp-servers/oxr", "oxr_mcp"),
]

def run() -> None:
    asyncio.run(run_stack(PROCESSES, _BASE))
```

Rules:
- **All processes start concurrently** — no ordering is required or expressed.
  Every process must tolerate its peers not being ready at startup.
  ZMQ reconnects automatically; `ProcessorEndpoint` works regardless of hub startup order.
- `xr_media_hub` always runs as its own process — never embedded in-process.
- The worker never imports anything from `server-runtime` or `launcher/`.
- Process management lives in `launcher/`, not inside any process it manages.
- `run_stack` is fail-fast: if any process exits, the rest are terminated.

## Adding a new sample

1. Create `agent-samples/<name>/` — the orchestrator project:
   ```toml
   # pyproject.toml
   dependencies = ["xr-ai-launcher"]

   [tool.uv.sources]
   xr-ai-launcher = { path = "../../launcher", editable = true }

   [project.scripts]
   my_agent = "my_agent.__main__:run"
   ```

2. Create `agent-samples/<name>/worker/` — the agent worker project:
   ```toml
   # worker/pyproject.toml
   dependencies = ["xr-ai-agent"]   # add numpy, torch, etc. as needed

   [tool.uv.sources]
   xr-ai-agent = { path = "../../../agent-sdk", editable = true }

   [project.scripts]
   my_agent_worker = "my_agent_worker.__main__:run"
   ```

3. Write `my_agent/__main__.py` as the orchestrator:
   ```python
   _BASE = Path(__file__).resolve().parents[1]
   PROCESSES = [
       Process("hub",    "../../server-runtime", "xr_media_hub"),
       Process("worker", "worker",               "my_agent_worker"),
   ]
   def run(): asyncio.run(run_stack(PROCESSES, _BASE))
   ```

4. Write `worker/my_agent_worker/__main__.py` — import only from `xr_ai_agent`.

5. Copy or symlink a `xr_media_hub.yaml` into `agent-samples/<name>/`.

6. Update `README.md` — architecture table and quickstart section.

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
- `agent-sdk/` (`xr-ai-agent`) depends only on `pyzmq` + `msgpack`. No server deps.
- Agent workers import only from `xr_ai_agent` (and task-specific libs like numpy/torch).
- Agent workers must never import from `xr_media_hub` or `xr_ai_launcher`.
- Don't add abstractions until needed by two concrete use-cases.

## Config

Each sample provides its own `xr_media_hub.yaml` in its project directory
(e.g. `agent-samples/echo-agent/xr_media_hub.yaml`). `server-runtime/` also
contains a reference copy documenting all available fields.

Paths inside the YAML (e.g. `web_client_dir`) resolve relative to the YAML
file's own directory, not CWD. `HubLauncher` finds the YAML automatically by
searching upward from CWD when the orchestrator runs.

---

## Decisions & change log

Significant decisions, in reverse-chronological order. Update this whenever a
non-trivial architectural or design decision is made so the rationale is
preserved and not re-litigated.

### 2026-04-22 — Launchable convention + StackLauncher

Each runnable sub-project (hub, worker, future CloudXR runtime, MCP servers) is a
**launchable**: an entry-point command + an optional `<command>.yaml` config.
The launcher discovers YAML files automatically by convention — no separate
launcher config file (the previous `stack.toml` idea was dropped).

The orchestrator code declares the process sequence using `Process` + `run_stack`.
All processes start concurrently; startup order does not matter because every
launchable must be resilient to peers not being ready (ZMQ reconnects, etc.).
`run_stack` is fail-fast: any process exit terminates the whole stack.

`launcher/` gained `Process`, `StackLauncher`, and `run_stack` (all stdlib-only).
`HubLauncher` / `ProjectLauncher` remain as lower-level building blocks.

### 2026-04-21 — Agent-SDK extracted; samples use orchestrator + worker subprocess model

`agent-sdk/` (`xr-ai-agent`) was extracted as a standalone package with only
`pyzmq` + `msgpack` as runtime dependencies. The four IPC client modules
(`_types`, `_codec`, `_shm`, `_processor`) moved there from `server-runtime`.
`server-runtime/xr_media_hub/ipc/__init__` re-exports everything for backwards compat.

Each sample now has two entry points:
- **Orchestrator** (`<name>`): stdlib + `xr-ai-launcher` only. Uses `HubLauncher`
  (which runs the hub via `uv run --project server-runtime`) and `ProjectLauncher`
  (which runs the worker via `uv run --project .`). Waits for the worker to exit.
- **Worker** (`<name>_worker`): imports only from `xr_ai_agent`. Contains all
  agent logic. Launched as a subprocess by the orchestrator.

`launcher/` gained `ProjectLauncher` — a generic context manager that runs any
uv project command as a managed subprocess in its own isolated venv, yielding
the `asyncio.subprocess.Process` for lifecycle control.

**Why:** complete venv isolation between hub (server-runtime), agent (sample), and
orchestrator (launcher-only). No cross-contamination of server deps into agent
venvs and vice versa. `uv run --project` is the mechanism — uv resolves and caches
each project's venv independently.

### 2026-04-21 — VLM agent sample added

`agent-samples/vlm-agent/` — answers natural-language queries about live XR
video using a locally-hosted vision-language model.
**Model:** `nvidia/Cosmos-Reason1-7B` (NVIDIA Open Model License + Apache 2.0,
commercial use permitted; ~16 GB VRAM at BF16). Architecture:
`Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` + `qwen-vl-utils`.
**Protocol:** client sends `vlm.query` data message (raw text or
`{"query":"…","track_id":"…"}`); agent replies on `vlm.response`.
**Frame flow:** `on_frame()` tracks latest `FrameSignal` per (participant,
track); on query, `request_frame(signal)` pulls a pixel copy, converts to PIL
via numpy (I420/NV12/RGB24/RGBA/BGRA), then calls `_VlmBackend.infer()` in a
thread pool so the asyncio loop is not blocked. Model is loaded lazily on the
first query. Override model via `VLM_MODEL` env var.

### 2026-04-21 — Process management moved to `launcher/`

`HubLauncher` lives in `launcher/xr_ai_launcher/`, not in `server-runtime`.
**Why:** process management should not be part of the processes it manages.
The launcher will eventually start MCP servers, CloudXR runtime, and other
components — keeping it separate keeps dependency chains lean and the boundary
clean. `launcher/` has zero runtime dependencies (stdlib only).

### 2026-04-21 — NVDEC/NVENC required; OpenH264 must not be used

`LiveKitConnector.start()` calls `require_nvidia_video_codecs()` before doing
anything else. It checks for `libnvcuvid.so` (NVDEC) and `libnvidia-encode.so`
(NVENC) via ctypes and raises `RuntimeError` if either is absent (Linux only).
**Why:** `livekit-rtc` bundles `libwebrtc` which includes OpenH264 as a software
fallback. OpenH264 is royalty-bearing for end users and must not ship in this
product. The guard prevents silent fallback at the cost of a hard startup failure.
In Docker: `--gpus all` or `--device /dev/nvidia*` must be passed.

### 2026-04-21 — Video frame delivery: metadata push, pixel pull

Processors receive `FrameSignal` metadata at full frame rate via `on_frame()`.
Pixel data is only copied when the processor calls `await ep.request_frame(signal)`.
The hub holds one SHM slot per (participant, track) — always the latest frame.
The slot stays `_STATE_READY` (not released to the connector) until the next frame
arrives for the same track, so `bytes(view.data)` in FRAME_REQUEST is safe.
**Why:** avoids copying every frame over IPC; agents sample at their own rate.
Concurrent `request_frame()` calls for the same track are coalesced into one
FRAME_REQUEST; all waiters receive the same FRAME_DATA response.

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
